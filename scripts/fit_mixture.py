"""Fit the mixture form of the Version-4 basket law (v4/mixture.py, docs/mixture_theory.pdf).

    python scripts/fit_mixture.py --init runs/commodity/v4_comp_promo.pt --out runs/commodity/v4_mixture \
        --epochs 2 --threads 4

Starts from a Version-4 checkpoint's parameters (both forms share them) and maximises, per training
basket, the K-sample importance bound on log p(S) plus the production priors on psi, delta = phi - psi_c,
rho_c and f_extra.  Trained: psi_cat, phi, rho_0, rho_c, f_extra, lam; context, price, promotion,
competition and history stay at the checkpoint.  Every --every steps:

    moment matching  one damped step of lam, rho_0, rho_c along their averaged gradient divided by the
                     data rate (the rule of scripts/fit_interactions.py)
    whitening        parameter-expanded EM: rescale phi, psi by the square root of the pooled posterior
                     second moment of z (mixture.whiten_), estimated with --whiten-K draws on
                     --whiten-trips fixed training trips (the K training draws overstate it)

Each trip's posterior peak is kept and starts its search on the next epoch.  Validation (every
--val-every steps and at the end): the mixture log-likelihood of the first --val-eval held-out trips
(K = --val-K draws; a lower bound in expectation) and the size of exact draws for --val-gen trips.
The held-out trips are the fit's validation split of scripts/fit_interactions.py (seed + 99).

Writes OUT.pt (final), OUT_best.pt (best validation), OUT.log (one JSON row per validation).
The checkpoints are Version-4 checkpoints with "form": "mixture".
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_default_dtype(torch.float64)

from v4 import mixture  # noqa: E402
from v4.bundle import data_root, load_bundle, supported_trips  # noqa: E402
from v4.checkpoints import load_version4  # noqa: E402
from v4.constraints import category_capacities  # noqa: E402
from v4.training import project  # noqa: E402

TRAINED = ("psi_cat", "phi", "rho_0_free", "rho_c", "f_extra", "lam")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--init", type=Path, required=True, help="Version-4 checkpoint to start from")
    p.add_argument("--data-root", type=Path, help="model-data bundle (default $ENERGY_MODEL_DATA_ROOT)")
    p.add_argument("--out", type=Path, required=True, help="output path without suffix")
    p.add_argument("--epochs", type=float, default=1.0, help="passes over the training trips (fractions allowed)")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--K", type=int, default=8, help="importance draws per training basket")
    p.add_argument("--mode-iters", type=int, default=8, help="peak-search iterations per step")
    p.add_argument("--spread-draws", type=int, default=32, help="exact draws for the posterior spread")
    p.add_argument("--every", type=int, default=30, help="steps between moment matching and whitening")
    p.add_argument("--val-trips", type=int, default=1024, help="held-out trips (the fit's validation split)")
    p.add_argument("--val-eval", type=int, default=64, help="held-out trips scored at each validation")
    p.add_argument("--val-K", type=int, default=256)
    p.add_argument("--val-gen", type=int, default=1024, help="held-out trips for the generated-size check")
    p.add_argument("--val-every", type=int, default=120, help="steps between validations")
    p.add_argument("--whiten-trips", type=int, default=256, help="fixed training trips for the whitening moment")
    p.add_argument("--whiten-K", type=int, default=256,
                   help="draws per trip for the whitening moment (K=8 overstates it by ~6%%, so the scale creeps up)")
    p.add_argument("--lr-phi", type=float, default=0.003)
    p.add_argument("--lr-rho0", type=float, default=0.05)
    p.add_argument("--lr-rho", type=float, default=0.03)
    p.add_argument("--lr-f-extra", type=float, default=0.006)
    p.add_argument("--lr-lam", type=float, default=0.008)
    p.add_argument("--phi-prec", type=float, default=20.0)
    p.add_argument("--delta-prec", type=float, default=200.0)
    p.add_argument("--rho-c-prec", type=float, default=0.2)
    p.add_argument("--f-extra-prec", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=41001)
    p.add_argument("--start-step", type=int, default=0,
                   help="resume a stopped fit: --init its _best.pt saved at this step, same seed and epochs")
    p.add_argument("--threads", type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    started = time.time()
    root = data_root(args.data_root)
    data = load_bundle(root)
    model, batcher, blob, _ = load_version4(args.init, data, root)
    if model.rank == 0:
        raise SystemExit("--init must be a Version-4 checkpoint (rank > 0)")
    for p in model.parameters():
        p.requires_grad_(False)
    nmax, r = int(model.nmax), model.rank
    J, C = int(model.lam.shape[0]), int(model.rho_c.shape[0])
    capacities = category_capacities(data, nmax)

    held = np.random.default_rng(args.seed + 99).permutation(supported_trips(data, 0, nmax))
    val = held[:args.val_trips]
    train = held[args.val_trips:]
    N = len(train)
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(N)                                   # positions in `train`
    batches = [np.sort(order[s:s + args.batch]) for s in range(0, N, args.batch)]
    steps = int(round(args.epochs * len(batches)))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(args.out.with_suffix(".log"), "a")

    def emit(row, to_log=True):
        print(json.dumps(row), flush=True)
        if to_log:
            log_file.write(json.dumps(row) + "\n")
            log_file.flush()

    emit({"event": "start", "train_trips": N, "val_trips": len(val), "batches": len(batches), "steps": steps,
          "args": {k: str(v) for k, v in vars(args).items()}})

    # Data rates for moment matching, from every training trip.
    with torch.no_grad():
        k_counts, pair_sum, item_counts = torch.zeros(nmax + 1), torch.zeros(C), torch.zeros(J)
        for pos in batches:
            bt = batcher.make(train[pos])
            n_row = torch.zeros(bt.ix.n_rows).index_add_(0, bt.ix.row_of, bt.observed.double())
            k = torch.zeros(bt.ix.B, dtype=torch.long).index_add_(0, bt.ix.row_trip, (n_row > 0).long())
            k_counts.index_add_(0, k.clamp(max=nmax), torch.ones_like(k, dtype=torch.float64))
            pair_sum.index_add_(0, bt.ix.row_cat, model.pair_feature(n_row))
            item_counts.index_add_(0, bt.ix.item, bt.observed.double())
        rate_k = (k_counts[1:] / N).clamp_min(2e-3)
        rate_pairs = (pair_sum / N).clamp_min(5e-3)
        rate_item = (item_counts / N).clamp_min(2e-3)

    ev = batcher.make(np.sort(val[:args.val_eval]))
    gb = batcher.make(np.sort(val[:args.val_gen]))
    wb = batcher.make(train[np.sort(np.random.default_rng(args.seed + 11).choice(N, args.whiten_trips, replace=False))])

    def second_moment():
        """Pooled posterior E[z z^T] on the fixed whitening trips, with many draws: the self-normalised
        weights of the K training draws lean toward the (wider) proposal and overstate it."""
        wb.bind(model)
        _, _, info = mixture.log_lik(model, wb, torch.Generator().manual_seed(args.seed + 12), K=args.whiten_K)
        return torch.einsum("kb,kbi,kbj->ij", info["w"], info["zs"], info["zs"]) / wb.ix.B
    best = -float("inf")

    def save(path, stage):
        out = dict(blob)
        out["model"] = {k: v.detach().clone() for k, v in model.state_dict().items()}
        out.update(form="mixture", stage=stage, init=str(args.init.resolve()), seconds=time.time() - started,
                   mixture_args={k: str(v) for k, v in vars(args).items()})
        torch.save(out, path)

    def validate(step):
        nonlocal best
        t0 = time.time()
        ev.bind(model)
        ll, ess, info = mixture.log_lik(model, ev, torch.Generator().manual_seed(args.seed + 7), K=args.val_K)
        gb.bind(model)
        X = mixture.generate(model, gb.ix, torch.Generator().manual_seed(args.seed + 8), iv=model.iv_shift(gb.ix))
        n = torch.zeros(gb.ix.B).index_add_(0, gb.ix.item_trip, X.double())
        n_obs = torch.zeros(gb.ix.B).index_add_(0, gb.ix.item_trip, gb.observed.double())
        row = {"step": step, "epoch": round(step / len(batches), 3), "val_ll": round(float(ll.mean()), 4),
               "val_ess_mean": round(float(ess.mean()), 3), "val_ess_min": round(float(ess.min()), 3),
               "gen_size_mean": round(float(n.mean()), 3), "data_size_mean": round(float(n_obs.mean()), 3),
               "gen_size_q99": float(n.quantile(0.99)), "data_size_q99": float(n_obs.quantile(0.99)),
               "gen_share_at_cap": round(float((n >= nmax).double().mean()), 4),
               "val_seconds": round(time.time() - t0, 1), "seconds": round(time.time() - started, 1)}
        if float(ll.mean()) > best:
            best = float(ll.mean())
            save(args.out.with_name(args.out.name + "_best.pt"), f"step {step} (best so far)")
            row["best"] = True
        emit(row)

    for _ in range(args.start_step // len(batches)):           # replay the batch orders already used
        rng.shuffle(batches)
    validate(args.start_step)
    for n in TRAINED:
        getattr(model, n).requires_grad_(True)
    opt = torch.optim.Adam([{"params": [model.psi_cat, model.phi], "lr": args.lr_phi},
                            {"params": [model.rho_0_free], "lr": args.lr_rho0},
                            {"params": [model.rho_c], "lr": args.lr_rho},
                            {"params": [model.f_extra], "lr": args.lr_f_extra},
                            {"params": [model.lam], "lr": args.lr_lam}])
    gen = torch.Generator().manual_seed(args.seed + 1)
    peaks = torch.zeros(N, r)                                    # each trip's last posterior peak
    seen = torch.zeros(N, dtype=torch.bool)
    cat = model.cat_of
    g_sums, g_count = None, 0
    zz, zz_n = torch.zeros(r, r), 0
    t_block = time.time()
    for step in range(args.start_step + 1, steps + 1):
        if step > 1 and (step - 1) % len(batches) == 0:          # a new epoch: new batch order
            rng.shuffle(batches)
        pos = batches[(step - 1) % len(batches)]
        bt = batcher.make(train[pos])
        bt.bind(model)
        pos_t = torch.as_tensor(pos)
        z0 = torch.where(seen[pos_t][:, None], peaks[pos_t], torch.zeros(len(pos), r))
        ll, ess, info = mixture.log_lik(model, bt, gen, K=args.K, mode_iters=args.mode_iters,
                                        spread_draws=args.spread_draws, grad=True, z0=z0)
        peaks[pos_t], seen[pos_t] = info["m"], True
        zz += torch.einsum("kb,kbi,kbj->ij", info["w"], info["zs"], info["zs"])
        zz_n += len(pos)
        psi, delta = model.psi_cat, model.phi[:, :r] - model.psi_cat[cat]
        prior = 0.5 * (args.phi_prec * psi.square().sum() + args.delta_prec * delta.square().sum()
                       + args.rho_c_prec * model.rho_c.square().sum()
                       + args.f_extra_prec * model.f_extra.square().sum()) / N
        loss = -ll.mean() + prior
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grads = [model.rho_0_free.grad.detach().clone(), model.rho_c.grad.detach().clone(),
                 model.lam.grad.detach().clone()]
        g_sums = grads if g_sums is None else [a + b for a, b in zip(g_sums, grads)]
        g_count += 1
        torch.nn.utils.clip_grad_norm_([getattr(model, n) for n in TRAINED], 10.0)
        opt.step()
        project(model, capacities, opt)
        if step % args.every == 0:
            with torch.no_grad():                                # moment matching
                r0, rc, la = (g / g_count for g in g_sums)
                model.rho_0_free.sub_(0.5 * (r0 / rate_k).clamp(-0.5, 0.5))
                model.rho_c.sub_(0.5 * (rc / rate_pairs).clamp(-0.5, 0.5))
                model.lam.sub_(0.35 * (la / rate_item).clamp(-0.35, 0.35))
            project(model, capacities, opt)
            train_trace = float(torch.trace(zz / zz_n)) / r
            eig = mixture.whiten_(model, second_moment())        # parameter-expanded EM
            for prm in (model.phi, model.psi_cat):
                opt.state.pop(prm, None)
            emit({"step": step, "lam_rel_err_mean": round(float((la / rate_item).abs().mean()), 4),
                  "whiten_eigs_top3": [round(float(v), 3) for v in eig[:3]],
                  "whiten_trace_over_rank": round(float(eig.sum() / r), 3),
                  "train_draws_trace_over_rank": round(train_trace, 3),
                  "steps_seconds": round(time.time() - t_block, 1)})
            g_sums, g_count, zz, zz_n, t_block = None, 0, torch.zeros(r, r), 0, time.time()
        if step % 10 == 0:
            emit({"step": step, "train_ll": round(float(ll.detach().mean()), 4), "ess": round(float(ess.mean()), 3),
                  "mode_grad": round(info["mode_grad"], 6)}, to_log=False)
        if step % args.val_every == 0 and step < steps:
            validate(step)
    for n in TRAINED:
        getattr(model, n).requires_grad_(False)
    validate(steps)
    save(args.out.with_suffix(".pt"), f"mixture form, {steps} steps")
    emit({"event": "trained", "steps": steps, "best_val_ll": round(best, 4),
          "seconds": round(time.time() - started, 1)})


if __name__ == "__main__":
    main()
