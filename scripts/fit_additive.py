#!/usr/bin/env python3
"""Stage 3: fit the exact additive base model (phi = 0) by maximum likelihood.

With phi = 0 the normaliser is an exact dynamic program over all offered products and all
basket sizes 1..nmax, so this stage needs no sampling.  It learns every utility block
(popularity, taste, price, discount, promotion, season, store), the within-category terms
rho_c and the size potential rho_0.

Objective, per minibatch of training trips (AdamW, gradient clipping):

    - mean log p(S)                                       exact likelihood
    + size_kl * [ cross-entropy(data size law, model size law)
                  + rkl_w * KL(model size law || smoothed data size law) ]
    + elast_w * (aggregate price elasticity - elast_target)^2
    + pool_prod * sum over (mu, delta), (zeta, xi), (alpha, theta) of a factor-pooling term

The elasticity anchor keeps the level of price response near the data's aggregate
elasticity while the household x product slopes are learned.  After each step: context
gauges are re-centred, lam is centred (its mean moves into rho_0, which leaves every basket
energy unchanged), and rho_c is projected onto its safety bounds (v4/constraints.py).

Validation: every --eval-every updates on --validation-trips validation trips.  The
learning rate halves after --lr-patience evaluations without a gain of
--validation-min-delta; the fit has converged when it is at --min-lr and
--convergence-patience evaluations have passed without such a gain.

Writes, in --output-dir:  v3_LABEL_best.pt (best validation), v3_LABEL.pt (latest, for
--resume), v3_LABEL_iterN.pt (milestones), v3_LABEL_history.json and v3_LABEL.log.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_default_dtype(torch.float64)

from v4.batches import Batcher  # noqa: E402
from v4.bundle import data_root, file_sha256, load_bundle, supported_trips  # noqa: E402
from v4.checkpoints import load_initialization  # noqa: E402
from v4.constraints import category_capacities, project_category_reward_  # noqa: E402
from v4.exact import additive_log_size  # noqa: E402
from v4.features import PRICE_CONTRACTS, Features  # noqa: E402
from v4.price import uniform_price_elasticity  # noqa: E402
from v4.reporting import strict_json_dumps  # noqa: E402
from v4.training import add_pooling  # noqa: E402

ESTIMATOR = "exact_version4_no_gram_dynamic_program"
FITTED = ("lam", "alpha", "theta", "rho_c", "rho_0_free", "price_kappa", "gamma", "beta",
          "w_dsp", "w_mlr", "mu", "delta", "zeta", "xi", "disc_kappa", "disc_common")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--artifact", type=Path, required=True, help="initialization (initialize.py)")
    p.add_argument("--data-root", type=Path, help="model-data bundle (default $ENERGY_MODEL_DATA_ROOT)")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--label", default="pipeline_additive")
    p.add_argument("--resume", type=Path, help="continue from v3_LABEL.pt; --iters is the final update")
    p.add_argument("--warm-start", type=Path,
                   help="start the fitted blocks from another additive fit (same products, possibly another "
                        "partition); rho_c, which belongs to the partition, starts from the initialization")
    p.add_argument("--iters", type=int, default=30000, help="safety ceiling on updates")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.002)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--clip", type=float, default=10.0)
    p.add_argument("--validation-trips", type=int, default=1024)
    p.add_argument("--validation-chunk", type=int, default=128)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--milestone-every", type=int, default=500)
    p.add_argument("--lr-patience", type=int, default=4)
    p.add_argument("--lr-factor", type=float, default=0.5)
    p.add_argument("--min-lr", type=float, default=6.25e-5)
    p.add_argument("--convergence-patience", type=int, default=8)
    p.add_argument("--convergence-min-updates", type=int, default=4000)
    p.add_argument("--validation-min-delta", type=float, default=1e-3)
    p.add_argument("--require-convergence", action="store_true",
                   help="exit with status 2 if --iters is reached first")
    p.add_argument("--seed", type=int, default=29001)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--size-kl", type=float, default=1.0)
    p.add_argument("--rkl-w", type=float, default=10.0)
    p.add_argument("--rkl-eps", type=float, default=1e-4)
    p.add_argument("--elast-w", type=float, default=20.0)
    p.add_argument("--elast-target", type=float, default=-0.121)
    p.add_argument("--elast-step", type=float, default=1e-3)
    p.add_argument("--pool-prod", type=float, default=1.45)
    p.add_argument("--rho-c-max-category-reward", type=float, default=1.5)
    p.add_argument("--price-feature-contract", choices=PRICE_CONTRACTS, default="chain_product_week_only")
    p.add_argument("--discount-panel", type=Path, help="centred discount panel (build_discount_panel.py)")
    p.add_argument("--disc-kappa", action="store_true", help="own relative multiplier for the discount part")
    p.add_argument("--disc-common", action="store_true", help="a discount slope common to all, from zero")
    return p.parse_args()


class Tee:
    """Write to the terminal and to the log file."""

    def __init__(self, stream, path, append):
        self.stream, self.file = stream, Path(path).open("a" if append else "w", buffering=1)

    def write(self, text):
        self.stream.write(text)
        self.file.write(text)
        return len(text)

    def flush(self):
        self.stream.flush()
        self.file.flush()


def atomic_save(path, payload):
    tmp = Path(str(path) + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def loglik_and_size_law(model, batch):
    """Exact log p(S) of each observed basket and each trip's size law P(n), n = 1..nmax."""
    log_size = additive_log_size(model, batch.ix)
    energy = model.additive_energy(batch.line_item, batch.line_trip, batch.line_cat, batch.ix.B,
                                   batch.line_ctx)
    return energy - torch.logsumexp(log_size, dim=-1), torch.softmax(log_size, dim=-1)


@torch.no_grad()
def validate(model, batcher, trips, chunk):
    model.eval()
    values, sizes, laws = [], [], []
    for s in range(0, len(trips), chunk):
        batch = batcher.make(trips[s:s + chunk])
        batch.bind(model)
        ll, law = loglik_and_size_law(model, batch)
        values.append(ll)
        laws.append(law)
        sizes.append(torch.bincount(batch.line_trip, minlength=batch.ix.B).double())
    model.train()
    value, observed, law = torch.cat(values).numpy(), torch.cat(sizes).numpy(), torch.cat(laws).numpy()
    grid = np.arange(1, law.shape[1] + 1, dtype=np.float64)
    expected = law @ grid
    return {"basket_loglik": float(value.mean()),
            "standard_error": float(value.std(ddof=1) / math.sqrt(len(value))),
            "observed_size_mean": float(observed.mean()),
            "observed_size_variance": float(observed.var(ddof=1)),
            "expected_size_mean": float(expected.mean()),
            "expected_size_median": float(np.median(expected)),
            "expected_size_p95": float(np.quantile(expected, .95)),
            "expected_size_max": float(expected.max()),
            "model_population_size_variance": float((law @ grid ** 2).mean()) - float(expected.mean()) ** 2,
            "trips": int(len(value))}


def main():
    args = parse_args()
    if (args.disc_kappa or args.disc_common) and args.discount_panel is None:
        raise SystemExit("--disc-kappa and --disc-common need --discount-panel")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = {k: output / f"v3_{args.label}{s}" for k, s in
             (("log", ".log"), ("latest", ".pt"), ("best", "_best.pt"), ("history", "_history.json"))}
    sys.stdout = Tee(sys.stdout, paths["log"], append=args.resume is not None)

    root = data_root(args.data_root)
    data = load_bundle(root)
    model, meta = load_initialization(args.artifact, data, root)
    model.train()
    nmax = int(meta["nmax"])
    panel = args.discount_panel.resolve() if args.discount_panel else None
    features = Features(root, int(data["n_item"]), int(data["n_store"]), recency=False,
                        price_contract=args.price_feature_contract, discount_panel=panel)
    # The discount split starts exactly at the one-slope model.
    if args.disc_kappa:
        model.disc_kappa = torch.nn.Parameter(model.price_kappa.detach().clone())
    if args.disc_common:
        model.disc_common = torch.nn.Parameter(torch.zeros(()))
    discount_record = None if panel is None else {
        "discount_panel": str(panel), "discount_panel_sha256": file_sha256(panel),
        "disc_kappa": bool(args.disc_kappa), "disc_common": bool(args.disc_common)}
    batcher = Batcher(data, features)
    if args.warm_start is not None:
        if args.resume is not None:
            raise SystemExit("--warm-start and --resume exclude each other")
        state = torch.load(args.warm_start, map_location="cpu", weights_only=False)["model"]
        taken = []
        with torch.no_grad():
            for name in FITTED:
                if name == "rho_c" or name not in state or getattr(model, name) is None:
                    continue
                if state[name].shape != getattr(model, name).shape:
                    raise SystemExit(f"--warm-start: {name} has shape {tuple(state[name].shape)}, "
                                     f"the model {tuple(getattr(model, name).shape)}")
                getattr(model, name).copy_(state[name])
                taken.append(name)
        print(f"[additive] warm start from {args.warm_start}: {', '.join(taken)}", flush=True)

    train = supported_trips(data, 0, nmax)
    counts = np.bincount(np.clip(data["trip_nlines"][train], 1, nmax), minlength=nmax + 1)[1:].astype(np.float64)
    data_size_law = torch.as_tensor(counts / counts.sum())
    valid_all = supported_trips(data, 1, nmax)
    valid = valid_all[np.random.default_rng(args.seed + 1).permutation(len(valid_all))[:args.validation_trips]]
    rng = np.random.default_rng(args.seed + 2)
    parameters = [getattr(model, n) for n in FITTED if getattr(model, n) is not None]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    capacities = category_capacities(data, nmax)

    # ---- resume ------------------------------------------------------------------------------
    start, resumed = 0, None
    if args.resume is not None:
        resumed = torch.load(args.resume, map_location="cpu", weights_only=False)
        prior = resumed["config"]
        checks = {"estimator": (resumed.get("estimator"), ESTIMATOR),
                  "data": (resumed.get("data_fingerprint_sha256"), meta["data_fingerprint_sha256"]),
                  "batch": (int(prior["batch"]), args.batch), "seed": (int(prior["seed"]), args.seed),
                  "price contract": (resumed.get("price_feature_contract"), args.price_feature_contract),
                  "discount split": (resumed.get("discount_split"), discount_record),
                  "category reward": (float(prior.get("rho_c_max_category_reward", 0.0)),
                                      args.rho_c_max_category_reward)}
        for name, (was, now) in checks.items():
            if was != now:
                raise SystemExit(f"cannot resume: {name} changed ({was} -> {now})")
        model.load_fitted_state(resumed["model"])
        optimizer.load_state_dict(resumed["optimizer"])
        start = int(resumed["iter"])
        if args.iters <= start:
            raise SystemExit(f"--iters must exceed the resumed update {start}")
        for _ in range(start):          # replay the minibatch draws so the stream continues
            rng.choice(len(train), size=args.batch, replace=False)

    print(f"[additive] J={model.J}, sizes 1..{nmax}, batch={args.batch}, lr={args.lr:g}, "
          f"price contract {args.price_feature_contract}", flush=True)
    if args.rho_c_max_category_reward > 0:
        project_category_reward_(model, capacities, args.rho_c_max_category_reward, optimizer=optimizer)

    started = time.perf_counter()
    current = validate(model, batcher, valid, args.validation_chunk)
    if resumed is None:
        initial, evaluations, records = current, [{"iter": 0, **current}], []
        best, best_iter = current["basket_loglik"], 0
        anchor, plateau, since_gain = best, 0, 0
        print(f"[additive] initial validation LL={best:.6f}", flush=True)
    else:
        evaluations, records = list(resumed["evaluations"]), list(resumed["records"])
        initial = {k: v for k, v in evaluations[0].items() if k != "iter"}
        best, best_iter = float(resumed["best_validation"]), int(resumed["best_iteration"])
        sched = resumed["scheduler"]
        anchor, plateau = float(sched["plateau_anchor"]), int(sched["plateau_evaluations"])
        since_gain = int(sched["evaluations_since_best"])
        saved = next(x for x in reversed(evaluations) if int(x["iter"]) == start)
        if abs(current["basket_loglik"] - saved["basket_loglik"]) > 5e-10:
            raise RuntimeError("the resumed model does not reproduce its saved validation score")
        print(f"[additive] resumed at update {start}: LL={current['basket_loglik']:.6f}", flush=True)

    def payload(iteration):
        return {"format": 2, "estimator": ESTIMATOR, "price_response_estimator": "exact_dp_fourth_order",
                "price_feature_contract": args.price_feature_contract, "discount_split": discount_record,
                "data_fingerprint_sha256": meta["data_fingerprint_sha256"],
                "initialization": str(args.artifact.resolve()), "iter": iteration,
                "model": model.state_dict(), "optimizer": optimizer.state_dict(), "config": vars(args),
                "best_validation": best, "best_iteration": best_iter,
                "scheduler": {"plateau_anchor": anchor, "plateau_evaluations": plateau,
                              "evaluations_since_best": since_gain,
                              "learning_rate": float(optimizer.param_groups[0]["lr"])},
                "evaluations": evaluations, "records": records}

    atomic_save(paths["best"], payload(start))
    converged = False
    for iteration in range(start + 1, args.iters + 1):
        tick = time.perf_counter()
        batch = batcher.make(train[rng.choice(len(train), size=args.batch, replace=False)])
        batch.bind(model)
        optimizer.zero_grad(set_to_none=True)
        ll, size_law = loglik_and_size_law(model, batch)
        score = ll.mean()
        loss = -score
        size_penalty = torch.zeros(())
        if args.size_kl > 0:
            model_law = size_law.mean(0).clamp_min(1e-12)
            model_law = model_law / model_law.sum()
            target = data_size_law[:model_law.numel()]
            size_penalty = -(target * model_law.log()).sum()
            if args.rkl_w > 0:
                smooth = (target + args.rkl_eps) / (target + args.rkl_eps).sum()
                size_penalty = size_penalty + args.rkl_w * (model_law * (model_law.log() - smooth.log())).sum()
            loss = loss + args.size_kl * size_penalty
        elasticity = torch.full((), float("nan"))
        if args.elast_w > 0:
            elasticity = uniform_price_elasticity(model, batch.ix, size_law, step=args.elast_step)
            loss = loss + args.elast_w * (elasticity - args.elast_target).square()
        if args.pool_prod > 0:
            loss = add_pooling(loss, model, args.pool_prod)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite loss at update {iteration}")
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(parameters, args.clip))
        optimizer.step()
        model.project_context_gauges()
        with torch.no_grad():               # centre lam; n * mean moves into rho_0(n)
            mean = model.lam.mean().clone()
            model.lam.sub_(mean)
            model.rho_0_free.sub_(mean * torch.arange(1, model.rho_0_free.numel() + 1, dtype=model.lam.dtype))
        model.project_rho_c(-1.5)
        max_reward = float("nan")
        if args.rho_c_max_category_reward > 0:
            max_reward = project_category_reward_(model, capacities, args.rho_c_max_category_reward,
                                                  optimizer=optimizer)
        records.append({"iter": iteration, "train_loglik": float(score.detach()), "grad_norm": grad_norm,
                        "size_penalty": float(size_penalty.detach()), "elasticity": float(elasticity.detach()),
                        "lam_sd": float(model.lam.std().detach()),
                        "house_size_sd": float(model.theta_c()[:, -1].std().detach()),
                        "maximum_category_reward": max_reward, "seconds": time.perf_counter() - tick})
        if iteration % args.log_every == 0 or iteration == args.iters:
            w = records[-min(args.log_every, len(records)):]
            print(f"[additive] step {iteration:5d} LL={np.mean([x['train_loglik'] for x in w]):.5f} "
                  f"grad={np.mean([x['grad_norm'] for x in w]):.3f} lam.sd={records[-1]['lam_sd']:.3f} "
                  f"elast={records[-1]['elasticity']:+.3f} {np.mean([x['seconds'] for x in w]):.3f}s/it",
                  flush=True)
        if iteration % args.eval_every == 0 or iteration == args.iters:
            current = validate(model, batcher, valid, args.validation_chunk)
            evaluations.append({"iter": iteration, **current})
            print(f"[additive] validation {iteration}: LL={current['basket_loglik']:.6f}, "
                  f"change={current['basket_loglik'] - initial['basket_loglik']:+.6f}, "
                  f"E[n]={current['expected_size_mean']:.2f}", flush=True)
            if current["basket_loglik"] > best:
                best, best_iter = current["basket_loglik"], iteration
                atomic_save(paths["best"], payload(iteration))
                print(f"[additive] new best at {iteration}", flush=True)
            if current["basket_loglik"] > anchor + args.validation_min_delta:
                anchor, plateau, since_gain = current["basket_loglik"], 0, 0
            else:
                plateau += 1
                since_gain += 1
            if args.lr_patience > 0 and plateau >= args.lr_patience:
                old = float(optimizer.param_groups[0]["lr"])
                new = max(args.min_lr, old * args.lr_factor)
                if new < old:
                    for group in optimizer.param_groups:
                        group["lr"] = new
                    print(f"[additive] validation plateau: lr {old:.3g} -> {new:.3g}", flush=True)
                plateau = 0
            atomic_save(paths["latest"], payload(iteration))
            if args.milestone_every and iteration % args.milestone_every == 0:
                atomic_save(output / f"v3_{args.label}_iter{iteration}.pt", payload(iteration))
            paths["history"].write_text(strict_json_dumps({
                "initial_validation": initial, "latest_validation": current, "best_validation": best,
                "best_iteration": best_iter, "evaluations": evaluations, "records": records,
                "wall_seconds": time.perf_counter() - started}))
            if (iteration >= args.convergence_min_updates and since_gain >= args.convergence_patience
                    and float(optimizer.param_groups[0]["lr"]) <= args.min_lr):
                converged = True
                print(f"[additive] converged at {iteration}", flush=True)
                break
    print(f"[additive] {'converged' if converged else 'reached --iters'}; best LL={best:.6f} at {best_iter}",
          flush=True)
    if args.require_convergence and not converged:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
