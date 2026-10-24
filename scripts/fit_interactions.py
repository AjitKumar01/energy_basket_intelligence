#!/usr/bin/env python3
"""Stage 4: train the full Version-4 model, starting from the exact additive base model.

Blocks added to the base model (all start at zero or small values):
    phi_j = psi_c(j) + delta_j   pairwise interactions, rank --rank, tied to categories
    psi_cat                      M2: interactions charged per category present, not per product
    f_extra                      M1: free potential for 2, 3, >= 4 products of one category
    size on categories           M3: rho_0 applies to the number of categories present
    u_cat                        M4a: household taste per category, theta_h . u_c
    psi, psi_item                --recency: sub-commodity recency and product purchase history
    hist_size                    --centre-history [mean|inclusive]: centre the history shift
                                 within each trip and learn its size effect, n * hist_size *
                                 log(1 + products ever bought) (see v4/model.py); needs --recency
    iv_slope_raw                 --m4c: discount only the price part of category inclusive value
    comp_raw                     --within-competition: products of a category compete for price
                                 and promotion changes (see v4/model.py and v4/competition_iv.py);
                                 not with --m4c

Stages:
    A  pseudo-likelihood (--pl-epochs): warms up phi with exact conditionals
    B  SML (--sml-epochs): stochastic maximum likelihood with one persistent Gibbs chain per
       training trip; price parameters are unfrozen here (centred prior at the base model);
       the running average of the parameters is scored on held-out training trips every
       --val-every epochs and the best one is kept (early stop after --patience)
    C  calibration (--cal-epochs): SML with phi and the context factors frozen, so the size
       and category terms settle to the final interactions

Moment matching: besides Adam, rho_0, rho_c and lam take preconditioned steps
(--mm-rho0, --mm-rho, --mm-lam) that move the model's category-presence, category-pair and
product rates toward the data's; these terms otherwise converge slowly.

--warm-start CK starts every parameter from a Version-4 checkpoint of the same shape (blocks
the checkpoint lacks, such as hist_size, start at zero); the price prior stays centred at the
base model.  Use it with --pl-epochs 0 to refit after a model change.

Writes OUT.pt (final), OUT_pl.pt (after stage A) and OUT.log (one JSON row per epoch).
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

from v4.bundle import data_root, load_bundle, supported_trips  # noqa: E402
from v4.checkpoints import batcher_for, load_additive  # noqa: E402
from v4.competition_iv import estimate_competition_iv  # noqa: E402
from v4.constraints import category_capacities  # noqa: E402
from v4.exact import log_partition  # noqa: E402
from v4.sampling import gibbs, importance_log_z  # noqa: E402
from v4.training import BASE_NAMES, CONTEXT_NAMES, Priors, penalties, project, pseudo_loglik  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--initialization", type=Path, required=True)
    p.add_argument("--additive", type=Path, required=True, help="the base model (fit_additive.py)")
    p.add_argument("--data-root", type=Path, help="model-data bundle (default $ENERGY_MODEL_DATA_ROOT)")
    p.add_argument("--out", type=Path, required=True, help="output path without suffix")
    p.add_argument("--recency", action="store_true", help="add recency and purchase-history features")
    p.add_argument("--m4c", action="store_true", help="add M4c (price-only inclusive value per category)")
    p.add_argument("--centre-history", nargs="?", const="mean", choices=("mean", "inclusive"),
                   help="centre the history shift within each trip (by its mean, or by its effect on the "
                        "trip's inclusive value) and learn its size effect (needs --recency)")
    p.add_argument("--warm-start", type=Path, help="start from this Version-4 checkpoint's parameters")
    p.add_argument("--iv-init", type=float, default=0.0, help="M4c: starting logit of lambda_c")
    p.add_argument("--within-competition", action="store_true",
                   help="within-category competition for price and promotion changes (learned lambda_c)")
    p.add_argument("--comp-estimator", choices=("instrument", "likelihood"), default="instrument",
                   help="estimator for within-category competition lambda_c: 'instrument' (exogenous "
                        "price-instrument Wald IV, default) or 'likelihood' (SML gradient steps)")
    p.add_argument("--comp-init", type=float, default=0.0, help="competition: starting logit of lambda_c")
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--max-per-category", type=int,
                   help="largest number of products from one category in a basket (default: nmax); "
                        "a smaller cap makes the DP cheaper and drops the baskets above it from the law")
    p.add_argument("--val-draws", type=int, default=256, help="importance draws per validation trip")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--phi-init", type=float, default=0.05, help="scale of the random starting psi")
    p.add_argument("--pl-epochs", type=int, default=2)
    p.add_argument("--sml-epochs", type=int, default=28)
    p.add_argument("--min-sml-epochs", type=int, default=16)
    p.add_argument("--cal-epochs", type=int, default=2)
    p.add_argument("--val-trips", type=int, default=2048, help="training trips held out for early stopping")
    p.add_argument("--val-every", type=int, default=4)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--val-tol", type=float, default=0.005)
    p.add_argument("--sweeps", type=int, default=4, help="Gibbs sweeps per SML step")
    p.add_argument("--lr-base", type=float, default=0.0015)
    p.add_argument("--lr-lam", type=float, default=0.008)
    p.add_argument("--lr-rho", type=float, default=0.03)
    p.add_argument("--lr-rho0", type=float, default=0.05)
    p.add_argument("--lr-phi-pl", type=float, default=0.001)
    p.add_argument("--lr-phi-sml", type=float, default=0.003)
    p.add_argument("--lr-price", type=float, default=0.003)
    p.add_argument("--lr-u-cat", type=float, default=0.005)
    p.add_argument("--lr-rec", type=float, default=0.005)
    p.add_argument("--lr-iv", type=float, default=0.05)
    p.add_argument("--lr-final-ratio", type=float, default=0.1, help="learning-rate decay over a stage")
    p.add_argument("--mm-rho", type=float, default=0.2)
    p.add_argument("--mm-rho0", type=float, default=0.5)
    p.add_argument("--mm-lam", type=float, default=0.5)
    p.add_argument("--pool-prod", type=float, default=0.5)
    p.add_argument("--phi-prec", type=float, default=20.0)
    p.add_argument("--delta-prec", type=float, default=200.0)
    p.add_argument("--rho-c-prec", type=float, default=0.2)
    p.add_argument("--f-extra-prec", type=float, default=2.0)
    p.add_argument("--price-prec", type=float, default=2.0)
    p.add_argument("--u-cat-prec", type=float, default=2.0)
    p.add_argument("--rec-prec", type=float, default=2.0)
    p.add_argument("--progress-every", type=int, default=50,
                   help="print a progress line (to stdout, not the .log) every this many updates")
    p.add_argument("--seed", type=int, default=41001)
    p.add_argument("--threads", type=int, default=4)
    return p.parse_args()


def add_version4_blocks(model, args, J, C):
    """Create the Version-4 parameters on the base model; returns (psi, delta, category)."""
    P = torch.nn.Parameter
    model.f_extra = P(torch.zeros(C, 3))
    model.u_cat = P(torch.zeros(C, model.theta.shape[1]))
    if args.m4c:
        model.iv_slope_raw = P(torch.full((C,), float(args.iv_init)))
    if args.within_competition:
        model.comp_raw = P(torch.full((C,), float(args.comp_init)))
    if args.recency:
        model.psi_item = P(torch.zeros(C, 3))
        with torch.no_grad():
            model.psi.zero_()
    if args.centre_history:
        model.hist_size = P(torch.zeros(()))
        model.centre_history = args.centre_history
    # M3: move the linear part of rho_0 (its slope over sizes 1..10) into lam, then let
    # rho_0 act on the number of categories present.
    model.size_on_categories = True
    with torch.no_grad():
        slope = (model.rho_0_free[9] - model.rho_0_free[0]) / 9.0
        model.lam.sub_(slope)
        model.rho_0_free.sub_(slope * torch.arange(1, model.rho_0_free.numel() + 1))
        model.rho_0_free.sub_(model.rho_0_free[0].clone())
    model.rank = args.rank
    category = model.cat_of.long()
    model.psi_cat = P(args.phi_init * torch.randn(C, args.rank))
    delta = torch.zeros(J, args.rank, requires_grad=True)
    with torch.no_grad():
        model.phi.zero_()
        model.phi[:, :args.rank] = model.psi_cat[category] + delta
    return model.psi_cat, delta, category


def main():
    args = parse_args()
    if args.val_trips <= 0:
        raise SystemExit("--val-trips must be positive (early stopping needs held-out trips)")
    if args.centre_history and not args.recency:
        raise SystemExit("--centre-history needs --recency")
    if args.within_competition and args.m4c:
        raise SystemExit("--within-competition and --m4c cannot be combined")
    torch.set_num_threads(args.threads)
    root = data_root(args.data_root)
    data = load_bundle(root)
    model, meta, parent = load_additive(args.initialization, args.additive, data, root)
    batcher = batcher_for(data, root, parent, recency=args.recency, purchase_history=args.recency)
    J, C, nmax = int(data["n_item"]), int(data["n_cat"]), int(meta["nmax"])

    # Held-out training trips for early stopping, then shuffled training batches.  A batch
    # is a fixed list of trips; its features are rebuilt whenever it is used (about 0.1 s),
    # because keeping every batch in memory would not fit on a large catalogue.
    rng = np.random.default_rng(args.seed)
    started = time.time()
    held = np.random.default_rng(args.seed + 99).permutation(supported_trips(data, 0, nmax))
    val_trips = np.sort(held[:args.val_trips])
    val_batches = [val_trips[s:s + args.batch] for s in range(0, args.val_trips, args.batch)]
    train_trips = np.sort(held[args.val_trips:])
    order = rng.permutation(train_trips)
    batches = [np.sort(order[s:s + args.batch]) for s in range(0, len(order), args.batch)]
    N = len(train_trips)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(args.out.with_suffix(".log"), "a")

    def progress(text):
        print(f"[progress {time.strftime('%H:%M')}] {text}", flush=True)

    def emit(row):
        print(json.dumps(row), flush=True)
        log_file.write(json.dumps(row) + "\n")
        log_file.flush()

    emit({"event": "start", "train_trips": N, "val_trips": args.val_trips, "batches": len(batches),
          "batching_seconds": round(time.time() - started, 1),
          "args": {k: str(v) for k, v in vars(args).items()}})

    torch.manual_seed(args.seed)
    psi, delta, category = add_version4_blocks(model, args, J, C)
    if args.max_per_category is not None:
        model.R = min(model.R, args.max_per_category)

    # Price (and, without --recency, the unused recency loading psi) stays at the base
    # model's values during stage A and is trained from stage B under a centred prior.
    price_names = ("gamma", "beta", "price_kappa") + tuple(
        n for n in ("disc_kappa", "disc_common") if getattr(model, n) is not None) + (
        () if args.recency else ("psi",))
    price_init = {n: getattr(model, n).detach().clone() for n in price_names}
    for n in price_names:
        getattr(model, n).requires_grad_(False)
    if args.warm_start is not None:
        state = torch.load(args.warm_start, map_location="cpu", weights_only=False)["model"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        if set(unexpected) - {"hist_size"} or set(missing) - {"hist_size", "comp_raw"}:
            raise SystemExit(f"--warm-start does not match this model: missing {missing}, unexpected {unexpected}")
        with torch.no_grad():
            delta.copy_(model.phi[:, :args.rank] - psi[category])
    use_comp_iv = args.within_competition and args.comp_estimator == "instrument"
    if use_comp_iv:
        model.comp_raw.requires_grad_(False)
        iv_stats = estimate_competition_iv(model, batches, batcher)
        emit({"event": "comp_iv_init", **iv_stats})
    rec_params = [model.psi, model.psi_item] if args.recency else []
    if args.centre_history:
        rec_params.append(model.hist_size)
    for q in rec_params:
        q.requires_grad_(True)
    base = [getattr(model, n) for n in BASE_NAMES]
    model.phi.requires_grad_(True)
    priors = Priors(args.pool_prod, args.phi_prec, args.delta_prec, args.rho_c_prec,
                    args.f_extra_prec, args.price_prec, args.u_cat_prec, args.rec_prec)
    capacities = category_capacities(data, nmax)
    gen = torch.Generator().manual_seed(args.seed + 1)

    # Data rates used to precondition the moment-matching steps.
    with torch.no_grad():
        k_counts, pair_sum, item_counts, cat_present = (
            torch.zeros(nmax + 1), torch.zeros(C), torch.zeros(J), torch.zeros(C))
        chains = []                                         # persistent SML chains, from the data
        for trips in batches:
            bt = batcher.make(trips)
            chains.append(bt.observed.clone())
            n_row = torch.zeros(bt.ix.n_rows).index_add_(0, bt.ix.row_of, bt.observed.double())
            if int(n_row.max()) >= model.R:      # keeps every observed basket (and PL step) inside the cap
                raise SystemExit(f"a training basket has {int(n_row.max())} products in one category; "
                                 f"--max-per-category must exceed it")
            k = torch.zeros(bt.ix.B, dtype=torch.long).index_add_(0, bt.ix.row_trip, (n_row > 0).long())
            k_counts.index_add_(0, k.clamp(max=nmax), torch.ones_like(k, dtype=torch.float64))
            pair_sum.index_add_(0, bt.ix.row_cat, model.pair_feature(n_row))
            cat_present.index_add_(0, bt.ix.row_cat, (n_row > 0).double())
            item_counts.index_add_(0, bt.ix.item, bt.observed.double())
        rate_k = (k_counts[1:] / N).clamp_min(2e-3)
        rate_pairs = (pair_sum / N).clamp_min(5e-3)
        rate_item = (item_counts / N).clamp_min(2e-3)
        rate_present = cat_present / N

    def sync_phi():
        with torch.no_grad():
            model.phi[:, :args.rank] = psi[category] + delta

    @torch.no_grad()
    def validation_ll():
        vgen = torch.Generator().manual_seed(args.seed + 777)
        lls, ess = [], []
        t_val = time.time()
        for b, trips in enumerate(val_batches):
            if b:
                progress(f"validation batch {b + 1}/{len(val_batches)} ({time.time() - t_val:.0f}s so far)")
            vb = batcher.make(trips)
            vb.bind(model)
            est, _, e, _ = importance_log_z(model, vb.ix, vgen, burn=80, components=32, thin=2, draws=args.val_draws)
            lls.append((model.energy(vb) - est).numpy())
            ess.append(e.numpy())
        return float(np.concatenate(lls).mean()), float(np.concatenate(ess).mean())

    def run_stage(name, epochs, step_fn, *, lr_base, lr_lam, lr_phi, lr_price=0.0, lr_u_cat=0.0,
                  lr_rec=0.0, lr_iv=0.0, average=False, monitor=False, moments=False):
        groups = [{"params": [getattr(model, n) for n in CONTEXT_NAMES], "lr": lr_base},
                  {"params": [model.lam], "lr": lr_lam},
                  {"params": [model.rho_c], "lr": args.lr_rho},
                  {"params": [model.rho_0_free], "lr": args.lr_rho0},
                  {"params": [psi, delta], "lr": lr_phi},
                  {"params": [model.f_extra], "lr": 0.2 * args.lr_rho}]
        extra = [model.f_extra]
        for params, lr in (([model.u_cat], lr_u_cat),
                           ([model.iv_slope_raw] if args.m4c else [], lr_iv),
                           ([model.comp_raw] if (args.within_competition and not use_comp_iv) else [], lr_iv),
                           (rec_params, lr_rec),
                           ([getattr(model, n) for n in price_names if getattr(model, n).requires_grad], lr_price)):
            if params and lr > 0.0:
                groups.append({"params": params, "lr": lr})
                extra.extend(params)
        opt = torch.optim.Adam(groups)
        total = epochs * len(batches)
        decay = args.lr_final_ratio ** (1.0 / max(total - 1, 1))
        mm = (args.mm_rho, args.mm_rho0, args.mm_lam) if moments else (0.0, 0.0, 0.0)
        avg, n_avg, step = None, 0, 0
        best_ll, best_state, bad = -float("inf"), None, 0
        for epoch in range(1, epochs + 1):
            t_epoch = time.time()
            values, sizes, model_present = [], [], torch.zeros(C)
            g_sums = [torch.zeros_like(model.rho_0_free), torch.zeros_like(model.rho_c),
                      torch.zeros_like(model.lam)]
            for i in rng.permutation(len(batches)):
                step += 1
                bt = batcher.make(batches[i])
                bt.bind(model)
                value, stats = step_fn(i, bt)
                loss = -value + penalties(model, priors, N, psi, delta, price_init, args.recency)
                opt.zero_grad(set_to_none=True)
                model.phi.grad = None
                loss.backward()
                with torch.no_grad():                       # chain rule phi -> (psi, delta)
                    g = model.phi.grad[:, :args.rank]
                    psi.grad = psi.grad + torch.zeros_like(psi).index_add_(0, category, g)
                    delta.grad = delta.grad + g
                    model.phi.grad = None
                g_r0, g_rc, g_lam = (model.rho_0_free.grad.detach().clone(),
                                     model.rho_c.grad.detach().clone(), model.lam.grad.detach().clone())
                for s, gr in zip(g_sums, (g_r0, g_rc, g_lam)):
                    s.add_(gr)
                torch.nn.utils.clip_grad_norm_(base + extra + [psi, delta], 10.0)
                opt.step()
                sync_phi()
                if any(mm):
                    scale = opt.param_groups[4]["lr"] / max(lr_phi, 1e-12) if lr_phi > 0 else 1.0
                    with torch.no_grad():
                        if mm[1] > 0.0:
                            model.rho_0_free.sub_(mm[1] * scale * g_r0)
                        if mm[0] > 0.0:
                            model.rho_c.sub_((mm[0] * scale / len(batches)) * (g_rc / rate_pairs))
                        if mm[2] > 0.0:
                            model.lam.sub_((mm[2] * scale / len(batches)) * (g_lam / rate_item))
                project(model, capacities, opt)
                for group in opt.param_groups:
                    group["lr"] *= decay
                values.append(float(value.detach()))
                done = len(values)
                if args.progress_every and done % args.progress_every == 0:
                    rate = (time.time() - t_epoch) / done
                    progress(f"{name} epoch {epoch}: {done}/{len(batches)} updates, objective "
                             f"{np.mean(values[-args.progress_every:]):.4f}, {rate:.1f}s/update, "
                             f"epoch done in ~{rate * (len(batches) - done) / 60:.0f} min")
                if stats is not None:
                    sizes.append(stats[:2])
                    model_present.add_(stats[2])
                if average and step > total // 3:          # running average after a third
                    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    avg = state if avg is None else {
                        k: (avg[k] + (state[k] - avg[k]) / (n_avg + 1)) if state[k].is_floating_point()
                        else state[k] for k in state}
                    n_avg += 1
            if any(mm):                                     # one damped full-data step per epoch
                with torch.no_grad():
                    r0, rc, la = (s / len(batches) for s in g_sums)
                    if mm[1] > 0.0:
                        model.rho_0_free.sub_(0.5 * (r0 / rate_k).clamp(-0.5, 0.5))
                    if mm[0] > 0.0:
                        model.rho_c.sub_(0.5 * (rc / rate_pairs).clamp(-0.5, 0.5))
                    if mm[2] > 0.0:
                        model.lam.sub_(0.35 * (la / rate_item).clamp(-0.35, 0.35))
                project(model, capacities, opt)
            if use_comp_iv:
                estimate_competition_iv(model, batches, batcher)
                if avg is not None:
                    avg["comp_raw"] = model.comp_raw.detach().clone()
            sv = torch.linalg.svdvals(model.phi_r().detach())
            row = {"stage": name, "epoch": epoch, "objective": float(np.mean(values)),
                   "phi_eig_top3": [round(float(v), 3) for v in sv[:3] ** 2],
                   "lr_phi": opt.param_groups[4]["lr"], "epoch_seconds": round(time.time() - t_epoch, 2),
                   "seconds": round(time.time() - started, 1)}
            if args.m4c:
                lam_c = model.iv_lambda().detach()
                row.update(iv_slope_mean=round(float(lam_c.mean()), 4), iv_slope_min=round(float(lam_c.min()), 4),
                           iv_slope_max=round(float(lam_c.max()), 4))
            if args.within_competition:
                lam_c = model.comp_lambda().detach()
                row.update(comp_lambda_mean=round(float(lam_c.mean()), 4), comp_lambda_min=round(float(lam_c.min()), 4),
                           comp_lambda_max=round(float(lam_c.max()), 4))
            if sizes:
                gap = (model_present / N - rate_present).abs()
                if args.centre_history:
                    row.update(hist_size=round(float(model.hist_size), 4))
                row.update(chain_mean_size=float(np.mean([s[0] for s in sizes])),
                           data_mean_size=float(np.mean([s[1] for s in sizes])),
                           cat_pres_gap_mean=round(float(gap.mean()), 4), cat_pres_gap_max=round(float(gap.max()), 4))
            if monitor and (epoch % args.val_every == 0 or epoch == epochs):
                current = {k: v.detach().clone() for k, v in model.state_dict().items()}
                scored = avg if avg is not None else current
                model.load_state_dict(scored)
                if use_comp_iv:
                    estimate_competition_iv(model, batches, batcher)
                    scored["comp_raw"] = model.comp_raw.detach().clone()
                ll, ess = validation_ll()
                row.update(val_ll=round(ll, 4), val_ess=round(ess, 4))
                if ll > best_ll + args.val_tol:
                    best_ll, best_state, bad = ll, {k: v.clone() for k, v in scored.items()}, 0
                else:
                    if ll > best_ll:
                        best_ll, best_state = ll, {k: v.clone() for k, v in scored.items()}
                    if epoch >= args.min_sml_epochs:
                        bad += 1
                model.load_state_dict(current)
                emit(row)
                if bad >= args.patience and epoch >= args.min_sml_epochs:
                    emit({"event": "early_stop", "stage": name, "epoch": epoch, "best_val_ll": round(best_ll, 4)})
                    break
            else:
                emit(row)
        if best_state is not None:
            model.load_state_dict(best_state)
        elif avg is not None:
            model.load_state_dict(avg)
        if use_comp_iv:
            estimate_competition_iv(model, batches, batcher)
        with torch.no_grad():                               # re-derive delta from the kept phi
            delta.copy_(model.phi[:, :args.rank] - psi[category])

    def save(path, stage):
        torch.save({"model": {k: v.detach().clone() for k, v in model.state_dict().items()},
                    "initialization": str(args.initialization.resolve()),
                    "additive": str(args.additive.resolve()),
                    "rank": args.rank, "max_per_category": model.R, "use_f_extra": True, "presence_phi": True, "cat_rho0": True,
                    "u_cat": True, "unfreeze_price": True, "recency": bool(args.recency),
                    "centre_history": args.centre_history, "within_competition": bool(args.within_competition),
                    "comp_estimator": args.comp_estimator if args.within_competition else None,
                    "iv_slope": bool(args.m4c), "iv_price_only": bool(args.m4c), "stage": stage,
                    "seconds": time.time() - started, "args": {k: str(v) for k, v in vars(args).items()}},
                   path)

    # Stage A: pseudo-likelihood.
    run_stage("pseudo-likelihood", args.pl_epochs, lambda i, bt: (pseudo_loglik(model, bt.ix, bt.observed), None),
              lr_base=args.lr_base, lr_lam=args.lr_lam, lr_phi=args.lr_phi_pl, lr_u_cat=args.lr_u_cat,
              lr_rec=args.lr_rec, lr_iv=args.lr_iv)
    save(args.out.with_name(args.out.name + "_pl.pt"), "pseudo-likelihood")

    # Stage B: SML.  Chains start at the observed baskets and persist across epochs.
    for n in price_names:
        getattr(model, n).requires_grad_(True)

    def sml_step(i, bt):
        ix = bt.ix
        chains[i] = X = gibbs(model, ix, chains[i], gen, sweeps=args.sweeps)
        with torch.no_grad():
            z = model.basket_field(ix, X) + torch.randn(ix.B, args.rank, generator=gen)
            n_row = torch.zeros(ix.n_rows).index_add_(0, ix.row_of, X.double())
            present = torch.zeros(C).index_add_(0, ix.row_cat, (n_row > 0).double())
        # Utilities and the M4c shift once per step, shared by the energy and the normaliser.
        b = model.slot_utilities(ix)
        iv = model.iv_shift(ix, b_slot=b)
        value = (model.energy(bt, iv=iv)
                 - log_partition(model, ix, model.slot_utilities_given_z(ix, z, b=b), z=z, iv=iv)).mean()
        chain_size = float(torch.zeros(ix.B).index_add_(0, ix.item_trip, X.double()).mean())
        return value, (chain_size, float(bt.observed.sum()) / ix.B, present)

    run_stage("sml", args.sml_epochs, sml_step, lr_base=args.lr_base, lr_lam=args.lr_lam,
              lr_phi=args.lr_phi_sml, lr_price=args.lr_price, lr_u_cat=args.lr_u_cat, lr_rec=args.lr_rec,
              lr_iv=args.lr_iv, average=True, monitor=True, moments=True)
    # Stage C: calibration, with phi and the context factors fixed.
    if args.cal_epochs > 0:
        run_stage("calibrate", args.cal_epochs, sml_step, lr_base=0.0, lr_lam=0.5 * args.lr_lam, lr_phi=0.0,
                  lr_price=0.5 * args.lr_price, lr_u_cat=0.5 * args.lr_u_cat, lr_rec=0.5 * args.lr_rec,
                  lr_iv=0.5 * args.lr_iv, moments=True)
    ll, ess = validation_ll()
    emit({"event": "final_val", "val_ll": round(ll, 4), "val_ess": round(ess, 4)})
    save(args.out.with_suffix(".pt"), "sml")
    emit({"event": "trained", "seconds": round(time.time() - started, 1)})


if __name__ == "__main__":
    main()
