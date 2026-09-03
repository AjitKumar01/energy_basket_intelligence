#!/usr/bin/env python3
"""Exact synthetic recovery audit for the Version-4 interaction mechanism.

The catalogue is deliberately small enough to enumerate every supported basket.  This
removes Smolyak/QMC error completely and asks a clean question: when data are generated
from a known low-rank Version-4 Gram interaction, can fresh maximum-likelihood fitting
recover a held-out advantage over multinomial and the matched additive parent?
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
from pathlib import Path

import numpy as np
import torch


torch.set_default_dtype(torch.float64)


def enumerate_support(items: int, nmax: int, categories: int):
    baskets = [combination for size in range(1, nmax + 1)
               for combination in itertools.combinations(range(items), size)]
    membership = torch.zeros(len(baskets), items)
    for row, basket in enumerate(baskets):
        membership[row, list(basket)] = 1.0
    sizes = membership.sum(1).long()
    item_category = torch.arange(items) % categories
    category_indicator = torch.nn.functional.one_hot(
        item_category, categories).double()
    category_count = membership @ category_indicator
    category_pairs = category_count * (category_count - 1.0) * 0.5
    return baskets, membership, sizes, item_category, category_count, category_pairs


def gram_pair_energy(membership: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    basket_sum = membership @ phi
    selected_norm = membership @ phi.square().sum(1)
    return 0.5 * (basket_sum.square().sum(1) - selected_norm)


def joint_log_probability(b, rho_size, rho_category, phi, membership, sizes,
                          category_pairs):
    energy = b @ membership.T
    energy = energy - rho_size[sizes - 1][None, :]
    energy = energy - (category_pairs @ rho_category)[None, :]
    if phi is not None:
        energy = energy + gram_pair_energy(membership, phi)[None, :]
    return energy - torch.logsumexp(energy, dim=1, keepdim=True)


class FittedLaw(torch.nn.Module):
    def __init__(self, kind, contexts, items, nmax, categories, rank, seed):
        super().__init__()
        self.kind = kind
        self.nmax = nmax
        self.b = torch.nn.Parameter(torch.zeros(contexts, items))
        if kind != "multinomial":
            self.rho_size_tail = torch.nn.Parameter(torch.zeros(nmax - 1))
            self.rho_category = torch.nn.Parameter(torch.zeros(categories))
        if kind.startswith("interaction"):
            generator = torch.Generator().manual_seed(seed)
            self.phi = torch.nn.Parameter(
                0.05 * torch.randn(items, rank, generator=generator))

    def rho_size(self):
        if self.kind == "multinomial":
            raise RuntimeError("multinomial has an empirical size law")
        return torch.cat([torch.zeros(1), self.rho_size_tail])

    def log_probability(self, membership, sizes, category_pairs, log_size=None):
        additive = self.b @ membership.T
        if self.kind == "multinomial":
            if log_size is None:
                raise ValueError("multinomial requires a fitted size law")
            answer = torch.empty_like(additive)
            for size in range(1, self.nmax + 1):
                selected = sizes == size
                block = additive[:, selected]
                answer[:, selected] = (
                    block - torch.logsumexp(block, dim=1, keepdim=True)
                    + log_size[size - 1])
            return answer
        return joint_log_probability(
            self.b, self.rho_size(), self.rho_category,
            self.phi if self.kind.startswith("interaction") else None,
            membership, sizes, category_pairs)


def counts(context, subset, contexts, subsets):
    flat = np.asarray(context) * subsets + np.asarray(subset)
    return torch.as_tensor(
        np.bincount(flat, minlength=contexts * subsets).reshape(contexts, subsets),
        dtype=torch.float64)


def empirical_log_size(train_subset, sizes, nmax, smoothing=0.5):
    observed = sizes[torch.as_tensor(train_subset)].numpy()
    count = np.bincount(observed, minlength=nmax + 1)[1:nmax + 1].astype(float)
    probability = (count + smoothing) / (count.sum() + smoothing * nmax)
    return torch.as_tensor(np.log(probability))


def fit_law(kind, train_count, valid_context, valid_subset, membership, sizes,
            category_pairs, log_size, rank, seed, steps, evaluation_every,
            patience, learning_rate, base_model=None):
    contexts, subsets = train_count.shape
    model = FittedLaw(kind, contexts, membership.shape[1], int(sizes.max()),
                      category_pairs.shape[1], rank, seed)
    # Training-only marginal incidence gives every model the same useful starting point.
    item_count = train_count @ membership
    exposure = train_count.sum(1, keepdim=True).clamp_min(1.0)
    with torch.no_grad():
        probability = ((item_count + 0.5) / (exposure + 1.0)).clamp(1e-4, 1-1e-4)
        model.b.copy_(torch.logit(probability))
        if base_model is not None:
            model.b.copy_(base_model.b)
            model.rho_size_tail.copy_(base_model.rho_size_tail)
            model.rho_category.copy_(base_model.rho_category)
    if kind == "interaction_frozen":
        model.b.requires_grad_(False)
        model.rho_size_tail.requires_grad_(False)
        model.rho_category.requires_grad_(False)
    optimized = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(optimized, lr=learning_rate)
    total = train_count.sum()
    best, best_step, stale, best_state = -float("inf"), 0, 0, None
    history = []
    vc = torch.as_tensor(valid_context, dtype=torch.long)
    vs = torch.as_tensor(valid_subset, dtype=torch.long)
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        logp = model.log_probability(membership, sizes, category_pairs, log_size)
        loss = -(train_count * logp).sum() / total
        penalty = 1e-5 * model.b.square().mean()
        if kind != "multinomial":
            penalty = penalty + 1e-5 * (
                model.rho_size_tail.square().mean()
                + model.rho_category.square().mean())
        if kind.startswith("interaction"):
            penalty = penalty + 1e-5 * model.phi.square().mean()
        (loss + penalty).backward()
        torch.nn.utils.clip_grad_norm_(optimized, 10.0)
        optimizer.step()
        if step % evaluation_every == 0 or step == steps:
            with torch.no_grad():
                valid_logp = model.log_probability(
                    membership, sizes, category_pairs, log_size)
                score = float(valid_logp[vc, vs].mean())
            history.append({"step": step, "train_nll": float(loss.detach()),
                            "validation_loglik": score})
            if score > best + 1e-5:
                best, best_step, stale = score, step, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    return model.eval(), {"best_step": best_step, "best_validation": best,
                          "terminal_step": step, "history": history}


def paired_summary(first, second):
    difference = np.asarray(first, dtype=float) - np.asarray(second, dtype=float)
    standard_error = float(difference.std(ddof=1) / math.sqrt(len(difference)))
    mean = float(difference.mean())
    return {"mean": mean, "standard_deviation": float(difference.std(ddof=1)),
            "standard_error": standard_error,
            "interval_95": [mean - 1.96 * standard_error,
                            mean + 1.96 * standard_error]}


def summarize(values):
    value = np.asarray(values, dtype=float)
    return {"mean": float(value.mean()),
            "standard_deviation": float(value.std(ddof=1)),
            "standard_error": float(value.std(ddof=1) / math.sqrt(len(value)))}


def add_one_scores(model, context, observed, item_category):
    candidates = np.asarray([j for j in range(model.b.shape[1])
                             if j not in observed], dtype=np.int64)
    score = model.b[context, torch.as_tensor(candidates)].detach().numpy().copy()
    if model.kind != "multinomial":
        category_count = np.bincount(
            item_category[np.asarray(observed, dtype=np.int64)].numpy(),
            minlength=len(model.rho_category))
        score -= (model.rho_category.detach().numpy()[
            item_category[candidates].numpy()] * category_count[
                item_category[candidates].numpy()])
    if model.kind.startswith("interaction") and observed:
        observed_sum = model.phi[torch.as_tensor(observed)].sum(0)
        score += (model.phi[torch.as_tensor(candidates)] @ observed_sum).detach().numpy()
    return candidates, score


def recommendation(model, contexts, subsets, baskets, item_category, seed):
    rng = np.random.default_rng(seed)
    reciprocals, ranks = [], []
    for context, subset in zip(contexts, subsets):
        basket = list(baskets[int(subset)])
        hidden = int(basket[rng.integers(len(basket))])
        observed = [item for item in basket if item != hidden]
        candidate, score = add_one_scores(model, int(context), observed, item_category)
        hidden_score = score[np.flatnonzero(candidate == hidden)[0]]
        rank = 1.0 + float(np.sum(score > hidden_score)) \
            + 0.5 * float(np.sum(score == hidden_score) - 1)
        ranks.append(rank); reciprocals.append(1.0 / rank)
    reciprocal = np.asarray(reciprocals)
    rank = np.asarray(ranks)
    return {"mrr": float(reciprocal.mean()),
            "mrr_standard_error": float(reciprocal.std(ddof=1) / math.sqrt(len(rank))),
            "recall_at_5": float(np.mean(rank <= 5)),
            "recall_at_10": float(np.mean(rank <= 10)),
            "median_rank": float(np.median(rank))}


def off_diagonal_kernel_correlation(true_phi, fitted_phi):
    true = true_phi @ true_phi.T
    fitted = fitted_phi @ fitted_phi.T
    index = torch.triu_indices(len(true), len(true), offset=1)
    a = true[index[0], index[1]].detach().numpy()
    b = fitted[index[0], index[1]].detach().numpy()
    if np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def one_experiment(args, strength, replicate, support):
    baskets, membership, sizes, item_category, _category_count, category_pairs = support
    seed = args.seed + 1009 * replicate + round(1000 * strength)
    rng = np.random.default_rng(seed)
    generator = torch.Generator().manual_seed(seed)
    base = -1.45 + 0.35 * torch.randn(args.items, generator=generator)
    context_shift = 0.22 * torch.randn(
        args.contexts, args.items, generator=generator)
    true_b = base[None, :] + context_shift
    raw_phi = torch.randn(args.items, args.rank, generator=generator) \
        / math.sqrt(args.rank)
    raw_phi = raw_phi - raw_phi.mean(0, keepdim=True)
    true_phi = float(strength) * raw_phi
    true_rho_size = 0.08 * (torch.arange(1, args.nmax + 1).double() - 2.5).square()
    true_rho_size = true_rho_size - true_rho_size[0].clone()
    true_rho_category = torch.zeros(args.categories)
    oracle_logp = joint_log_probability(
        true_b, true_rho_size, true_rho_category, true_phi,
        membership, sizes, category_pairs).detach().numpy()

    def sample(count):
        context = rng.integers(args.contexts, size=count)
        subset = np.empty(count, dtype=np.int64)
        for h in range(args.contexts):
            selected = np.flatnonzero(context == h)
            subset[selected] = rng.choice(
                len(baskets), size=len(selected), p=np.exp(oracle_logp[h]))
        return context, subset

    train_context, train_subset = sample(args.train)
    valid_context, valid_subset = sample(args.validation)
    test_context, test_subset = sample(args.test)
    train_count = counts(train_context, train_subset, args.contexts, len(baskets))
    log_size = empirical_log_size(train_subset, sizes, args.nmax)
    fitted, training = {}, {}
    for position, kind in enumerate(("multinomial", "additive")):
        fitted[kind], training[kind] = fit_law(
            kind, train_count, valid_context, valid_subset, membership, sizes,
            category_pairs, log_size, args.rank, seed + 41 + position,
            args.steps, args.eval_every, args.patience,
            args.interaction_lr if kind == "interaction" else args.lr)
    for kind in ("interaction_frozen", "interaction"):
        fitted[kind], training[kind] = fit_law(
            kind, train_count, valid_context, valid_subset, membership, sizes,
            category_pairs, log_size, args.rank, seed + 43,
            args.steps, args.eval_every, args.patience, args.interaction_lr,
            base_model=fitted["additive"])
    tc = torch.as_tensor(test_context, dtype=torch.long)
    ts = torch.as_tensor(test_subset, dtype=torch.long)
    per_trip = {"oracle": oracle_logp[test_context, test_subset]}
    with torch.no_grad():
        for kind, model in fitted.items():
            logp = model.log_probability(membership, sizes, category_pairs, log_size)
            per_trip[kind] = logp[tc, ts].numpy()
    result = {
        "strength": float(strength), "replicate": replicate, "seed": seed,
        "test_log_likelihood": {name: summarize(value)
                                for name, value in per_trip.items()},
        "paired_gains": {
            "interaction_minus_multinomial": paired_summary(
                per_trip["interaction"], per_trip["multinomial"]),
            "interaction_minus_additive": paired_summary(
                per_trip["interaction"], per_trip["additive"]),
            "frozen_interaction_minus_additive": paired_summary(
                per_trip["interaction_frozen"], per_trip["additive"]),
            "joint_minus_frozen_interaction": paired_summary(
                per_trip["interaction"], per_trip["interaction_frozen"]),
            "oracle_minus_additive": paired_summary(
                per_trip["oracle"], per_trip["additive"]),
        },
        "recommendation": {kind: recommendation(
            model, test_context, test_subset, baskets, item_category,
            seed + 9001) for kind, model in fitted.items()},
        "kernel_recovery": off_diagonal_kernel_correlation(
            true_phi, fitted["interaction"].phi),
        "frozen_kernel_recovery": off_diagonal_kernel_correlation(
            true_phi, fitted["interaction_frozen"].phi),
        "training": training,
    }
    result["recommendation"]["interaction_minus_additive_mrr"] = (
        result["recommendation"]["interaction"]["mrr"]
        - result["recommendation"]["additive"]["mrr"])
    result["recommendation"]["frozen_interaction_minus_additive_mrr"] = (
        result["recommendation"]["interaction_frozen"]["mrr"]
        - result["recommendation"]["additive"]["mrr"])
    return result


def markdown_report(result):
    lines = ["# Exact synthetic interaction recovery audit", "",
             "All supported baskets were enumerated exactly; there is no quadrature or "
             "Monte Carlo normalizer error.", "",
             "| Strength | Replicate | Interaction − multinomial | Interaction − additive | "
             "Frozen − additive | MRR gain vs additive | Kernel correlation |",
             "|---:|---:|---:|---:|---:|---:|---:|"]
    for row in result["experiments"]:
        im = row["paired_gains"]["interaction_minus_multinomial"]
        ia = row["paired_gains"]["interaction_minus_additive"]
        fa = row["paired_gains"]["frozen_interaction_minus_additive"]
        lines.append(
            f"| {row['strength']:.3g} | {row['replicate']} | "
            f"{im['mean']:+.4f} ± {im['standard_error']:.4f} | "
            f"{ia['mean']:+.4f} ± {ia['standard_error']:.4f} | "
            f"{fa['mean']:+.4f} ± {fa['standard_error']:.4f} | "
            f"{row['recommendation']['interaction_minus_additive_mrr']:+.4f} | "
            + (f"{row['kernel_recovery']:+.3f} |"
               if row['kernel_recovery'] is not None else "N/A |"))
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=int, default=14)
    parser.add_argument("--contexts", type=int, default=6)
    parser.add_argument("--categories", type=int, default=4)
    parser.add_argument("--nmax", type=int, default=5)
    parser.add_argument("--rank", type=int, default=3)
    parser.add_argument("--train", type=int, default=20000)
    parser.add_argument("--validation", type=int, default=5000)
    parser.add_argument("--test", type=int, default=10000)
    parser.add_argument("--strengths", type=float, nargs="+", default=[0.0, 0.35, 0.7])
    parser.add_argument("--replicates", type=int, default=2)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--interaction-lr", type=float, default=0.015)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=41173)
    parser.add_argument("--output", type=Path,
                        default=Path("reports/synthetic_interaction_audit.json"))
    args = parser.parse_args()
    if args.rank > args.items or args.nmax > args.items:
        raise ValueError("rank and nmax cannot exceed item count")
    torch.set_num_threads(args.threads)
    support = enumerate_support(args.items, args.nmax, args.categories)
    experiments = []
    for strength in args.strengths:
        for replicate in range(args.replicates):
            print(f"[synthetic] strength={strength:g} replicate={replicate}", flush=True)
            row = one_experiment(args, strength, replicate, support)
            experiments.append(row)
            gain = row["paired_gains"]["interaction_minus_additive"]
            kernel = (f"{row['kernel_recovery']:+.3f}"
                      if row['kernel_recovery'] is not None else "N/A")
            print(f"[synthetic] interaction-additive {gain['mean']:+.5f} "
                  f"+/- {gain['standard_error']:.5f}; kernel r={kernel}", flush=True)
    result = {"schema": 1, "estimator": "exact enumeration",
              "support_baskets": len(support[0]), "config": vars(args),
              "experiments": experiments}
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    serializable = copy.deepcopy(result)
    serializable["config"]["output"] = str(args.output)
    output.write_text(json.dumps(serializable, indent=2) + "\n")
    output.with_suffix(".md").write_text(markdown_report(serializable))
    print(f"[synthetic] report: {output}", flush=True)


if __name__ == "__main__":
    main()
