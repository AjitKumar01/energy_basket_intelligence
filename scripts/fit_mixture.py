"""Fit the mixture form of the Version-4 basket law (v4/mixture.py, docs/mixture_theory.pdf).

    python scripts/fit_mixture.py --initialization runs/commodity/initialization.pt \
        --additive runs/commodity/additive/v3_pipeline_additive_best.pt --out runs/commodity/mixture \
        --recency --centre-history mean --within-competition --comp-iv-trips 40000 --promo-common \
        --max-per-category 40 --epochs 2 --threads 4

Starts from the exact additive base model (scripts/fit_additive.py), adds every Version-4 block
(v4/training.add_blocks) and maximises, per training basket, the K-sample importance bound on
log p(S) plus the Gaussian priors of v4/training.penalties.  Every parameter is trained by this
objective except the competition weights lambda_c, which the price instrument re-estimates
(v4/competition_iv.py) at the start and every --comp-every steps.  Price parameters have a prior
centred at the base model's values.  Every --every steps:

    moment matching  one damped step of lam, rho_0, rho_c along their averaged gradient divided by
                     the data rate
    whitening        parameter-expanded EM: rescale phi, psi by the square root of the pooled posterior
                     second moment of z (mixture.whiten_), estimated with --whiten-K draws on
                     --whiten-trips fixed training trips (the K training draws overstate it)

--hier-prior (docs/overfitting_theory.pdf, Section 4): every household and product-context row gets
the prior N(m, diag s2) of its block, and m, s2 are re-estimated by EM every --em-every steps.

Each trip's posterior peak is kept and starts its search on the next epoch.  Validation (every
--val-every steps and at the end): the mixture log-likelihood of the first --val-eval held-out trips
(K = --val-K draws; a lower bound in expectation) and the size of exact draws for --val-gen trips.
The held-out trips are the first --val-trips of the supported training-split trips in the seeded
order (seed + 99).

--init CK continues from a checkpoint this script wrote (with --start-step, from the step it was
saved at, the same seed and epochs: the batch order is replayed).

Writes OUT.pt (final), OUT_best.pt (best validation), OUT.log (one JSON row per validation and
per moment-matching step).
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
from v4.checkpoints import batcher_for, load_additive, load_model  # noqa: E402
from v4.competition_iv import estimate_competition_iv  # noqa: E402
from v4.constraints import category_capacities  # noqa: E402
from v4.hier_prior import HierPrior  # noqa: E402
from v4.training import CONTEXT_NAMES, Priors, add_blocks, penalties, project  # noqa: E402

# Blocks for --anchor (docs/training_theory.pdf, Section 3).
ANCHOR_BLOCKS = {"household": ("theta", "gamma"),
                 "product_context": ("alpha", "beta", "mu", "zeta", "w_dsp", "w_mlr", "delta", "xi")}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--initialization", type=Path, help="initialization artifact (scripts/initialize.py)")
    p.add_argument("--additive", type=Path, help="the base model (scripts/fit_additive.py)")
    p.add_argument("--init", type=Path, help="continue from a checkpoint written by this script")
    p.add_argument("--data-root", type=Path, help="model-data bundle (default $ENERGY_MODEL_DATA_ROOT)")
    p.add_argument("--out", type=Path, required=True, help="output path without suffix")
    # Blocks (a fresh start only; a continued fit keeps those of its checkpoint).
    p.add_argument("--rank", type=int, default=16, help="dimension of the taste z")
    p.add_argument("--phi-init", type=float, default=0.05, help="scale of the random starting psi")
    p.add_argument("--recency", action="store_true", help="add recency and purchase-history features")
    p.add_argument("--centre-history", nargs="?", const="mean", choices=("mean", "inclusive"),
                   help="centre the history shift within each trip and learn its size effect (needs --recency)")
    p.add_argument("--within-competition", action="store_true",
                   help="within-category competition for price and promotion changes (lambda_c by instrument)")
    p.add_argument("--promo-common", action="store_true",
                   help="common display and mailer slopes with per-product deviations shrunk to zero")
    p.add_argument("--max-per-category", type=int,
                   help="largest number of products from one category in a basket (default: nmax)")
    p.add_argument("--comp-iv-trips", type=int, default=0,
                   help="instrument: estimate lambda_c on this many training trips (a fixed random subset; 0: all)")
    p.add_argument("--comp-every", type=int, default=0, help="steps between instrument estimates (0: one epoch)")
    # Schedule.
    p.add_argument("--epochs", type=float, default=1.0, help="passes over the training trips (fractions allowed)")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--K", type=int, default=8, help="importance draws per training basket")
    p.add_argument("--mode-iters", type=int, default=8, help="peak-search iterations per step")
    p.add_argument("--spread-draws", type=int, default=32, help="exact draws for the posterior spread")
    p.add_argument("--every", type=int, default=30, help="steps between moment matching and whitening")
    p.add_argument("--val-trips", type=int, default=1024, help="held-out trips (never trained on)")
    p.add_argument("--val-eval", type=int, default=64, help="held-out trips scored at each validation")
    p.add_argument("--val-K", type=int, default=256)
    p.add_argument("--val-gen", type=int, default=1024, help="held-out trips for the generated-size check")
    p.add_argument("--val-every", type=int, default=120, help="steps between validations")
    p.add_argument("--whiten-trips", type=int, default=256, help="fixed training trips for the whitening moment")
    p.add_argument("--whiten-K", type=int, default=256,
                   help="draws per trip for the whitening moment (K=8 overstates it by ~6%%, so the scale creeps up)")
    # Learning rates.
    p.add_argument("--lr-base", type=float, default=0.0015, help="context factors and promotion slopes")
    p.add_argument("--lr-phi", type=float, default=0.003)
    p.add_argument("--lr-rho0", type=float, default=0.05)
    p.add_argument("--lr-rho", type=float, default=0.03)
    p.add_argument("--lr-f-extra", type=float, default=0.006)
    p.add_argument("--lr-lam", type=float, default=0.008)
    p.add_argument("--lr-price", type=float, default=0.003)
    p.add_argument("--lr-u-cat", type=float, default=0.005)
    p.add_argument("--lr-rec", type=float, default=0.005)
    # Prior precisions (v4/training.Priors).
    p.add_argument("--pool-prod", type=float, default=0.5)
    p.add_argument("--phi-prec", type=float, default=20.0)
    p.add_argument("--delta-prec", type=float, default=200.0)
    p.add_argument("--rho-c-prec", type=float, default=0.2)
    p.add_argument("--f-extra-prec", type=float, default=2.0)
    p.add_argument("--price-prec", type=float, default=2.0)
    p.add_argument("--u-cat-prec", type=float, default=2.0)
    p.add_argument("--rec-prec", type=float, default=2.0)
    p.add_argument("--promo-dev-prec", type=float, default=4.0, help="--promo-common: prior on the deviations")
    p.add_argument("--hier-prior", action="store_true",
                   help="EXPERIMENTAL, did not fix the overfitting (docs/BRIEF_mixture_overfitting.md): Gaussian priors with estimated scales on the household (theta, gamma) and product-context "
                        "(alpha, beta, mu, zeta) blocks (v4/hier_prior.py); gamma, beta leave the price prior")
    p.add_argument("--em-every", type=int, default=0, help="--hier-prior: steps between scale updates (0: one epoch)")
    # Switches for the experiments of docs/training_theory.pdf, Section 7.
    p.add_argument("--no-moment-match", action="store_true", help="skip the moment-matching steps")
    p.add_argument("--no-whiten", action="store_true", help="skip the whitening steps")
    p.add_argument("--anchor", default="", help="comma list of blocks (household, product_context) held near their "
                                                "values at the start of the fit by --anchor-prec")
    p.add_argument("--anchor-prec", type=float, default=0.0,
                   help="precision tau of the anchor penalty tau/2 |x - x0|^2 (total log-likelihood units); inf freezes")
    p.add_argument("--interaction-only", action="store_true",
                   help="train only psi, phi, rho_0, rho_c, f_extra and lam (the scope of the fits before d716f95, "
                        "such as runs/commodity/v4_mixture_w); with --init it also accepts their checkpoints")
    p.add_argument("--seed", type=int, default=41001)
    p.add_argument("--start-step", type=int, default=0,
                   help="with --init: the step its checkpoint was saved at (same seed and epochs)")
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    if (args.init is None) == (args.initialization is None or args.additive is None):
        p.error("give either --initialization and --additive (a fresh fit) or --init (a continued one)")
    if args.start_step and args.init is None:
        p.error("--start-step needs --init")
    if args.centre_history and not args.recency:
        p.error("--centre-history needs --recency")
    args.anchor = [b for b in args.anchor.split(",") if b]
    if any(b not in ANCHOR_BLOCKS for b in args.anchor):
        p.error(f"--anchor blocks must be among {sorted(ANCHOR_BLOCKS)}")
    if args.anchor and args.anchor_prec <= 0.0:
        p.error("--anchor needs --anchor-prec > 0")
    return args


def start_fresh(args, data, root):
    """The base model with every Version-4 block added, its Batcher, the checkpoint blob and nmax."""
    model, meta, parent = load_additive(args.initialization, args.additive, data, root)
    add_blocks(model, args.rank, phi_init=args.phi_init, recency=args.recency,
               centre_history=args.centre_history, within_competition=args.within_competition,
               gen=torch.Generator().manual_seed(args.seed + 2))
    if args.max_per_category is not None:
        model.R = min(model.R, args.max_per_category)
    batcher = batcher_for(data, root, parent, recency=args.recency, purchase_history=args.recency)
    blob = {"initialization": str(args.initialization.resolve()), "additive": str(args.additive.resolve()),
            "rank": args.rank, "max_per_category": model.R, "cat_rho0": True, "recency": bool(args.recency),
            "centre_history": args.centre_history, "within_competition": bool(args.within_competition),
            "comp_estimator": "instrument" if args.within_competition else None,
            "promo_common": bool(args.promo_common)}
    return model, batcher, blob, int(meta["nmax"])


def continue_from(args, data, root):
    """The model of a checkpoint this script wrote, its Batcher, its blob and nmax."""
    model, meta, parent, flags, blob = load_model(args.init, data, root)
    if blob.get("form") != "mixture":
        raise SystemExit(f"--init {args.init} was not written by this script")
    if "price_init" not in blob:
        if not args.interaction_only:
            raise SystemExit(f"--init {args.init} has no price_init (an older fit): continue it with --interaction-only")
        blob["price_init"] = {}                                  # price terms are not trained
    batcher = batcher_for(data, root, parent, recency=flags["recency"], purchase_history=flags["recency"])
    return model, batcher, blob, int(meta["nmax"])


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    started = time.time()
    root = data_root(args.data_root)
    data = load_bundle(root)
    fresh = args.init is None
    model, batcher, blob, nmax = (start_fresh if fresh else continue_from)(args, data, root)
    for p in model.parameters():
        p.requires_grad_(False)
    r = model.rank
    J, C = int(model.lam.shape[0]), int(model.rho_c.shape[0])
    recency = bool(blob["recency"])
    promo_common = bool(blob["promo_common"])
    competition = model.comp_raw is not None
    capacities = category_capacities(data, nmax)

    held = np.random.default_rng(args.seed + 99).permutation(supported_trips(data, 0, nmax))
    val = held[:args.val_trips]
    train = held[args.val_trips:]
    N = len(train)
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(N)                                   # positions in `train`
    batches = [np.sort(order[s:s + args.batch]) for s in range(0, N, args.batch)]
    steps = int(round(args.epochs * len(batches)))
    # The batches hold shuffled trips, so the first ones are a fixed random subset.
    n_iv = len(batches) if args.comp_iv_trips <= 0 else max(1, -(-args.comp_iv_trips // args.batch))
    iv_batches = [train[pos] for pos in batches[:n_iv]]
    comp_every = args.comp_every or len(batches)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(args.out.with_suffix(".log"), "a")

    def emit(row, to_log=True):
        print(json.dumps(row), flush=True)
        if to_log:
            log_file.write(json.dumps(row) + "\n")
            log_file.flush()

    emit({"event": "start", "fresh": fresh, "train_trips": N, "val_trips": len(val), "batches": len(batches),
          "steps": steps, "args": {k: str(v) for k, v in vars(args).items()}})

    # Data rates for moment matching (and promotion exposure for the common slopes), from every
    # training trip.
    with torch.no_grad():
        k_counts, pair_sum, item_counts = torch.zeros(nmax + 1), torch.zeros(C), torch.zeros(J)
        promo_slots = (torch.zeros(J), torch.zeros(J))           # displayed / mailed training slots
        for pos in batches:
            bt = batcher.make(train[pos])
            for n_slots, key in zip(promo_slots, ("disp", "mail")):
                n_slots.index_add_(0, bt.ix.item, bt.ctx[key].double())
            n_row = torch.zeros(bt.ix.n_rows).index_add_(0, bt.ix.row_of, bt.observed.double())
            if int(n_row.max()) >= model.R:      # every observed basket must lie inside the cap
                raise SystemExit(f"a training basket has {int(n_row.max())} products in one category; "
                                 f"--max-per-category must exceed it")
            k = torch.zeros(bt.ix.B, dtype=torch.long).index_add_(0, bt.ix.row_trip, (n_row > 0).long())
            k_counts.index_add_(0, k.clamp(max=nmax), torch.ones_like(k, dtype=torch.float64))
            pair_sum.index_add_(0, bt.ix.row_cat, model.pair_feature(n_row))
            item_counts.index_add_(0, bt.ix.item, bt.observed.double())
        rate_k = (k_counts[1:] / N).clamp_min(2e-3)
        rate_pairs = (pair_sum / N).clamp_min(5e-3)
        rate_item = (item_counts / N).clamp_min(2e-3)

    # Price (and, without recency, the unused recency loading psi) has a prior centred at the
    # base model's values.
    price_names = ("gamma", "beta", "price_kappa") + tuple(
        n for n in ("disc_kappa", "disc_common") if getattr(model, n) is not None) + (() if recency else ("psi",))
    if fresh:
        blob["price_init"] = {n: getattr(model, n).detach().clone() for n in price_names}
        if promo_common:                 # common slopes start at the exposure-weighted base weights
            start = {}
            for name, weight, n_slots in (("dsp_common", model.w_dsp, promo_slots[0]),
                                          ("mlr_common", model.w_mlr, promo_slots[1])):
                common = float((n_slots * weight).sum() / n_slots.sum().clamp_min(1.0))
                setattr(model, name, torch.nn.Parameter(torch.tensor(common)))
                start[name] = round(common, 4)
                with torch.no_grad():
                    weight.zero_()
            emit({"event": "promo_common_init", **start})
        if competition:
            t_iv = time.time()
            emit({"event": "comp_iv", "step": 0, **estimate_competition_iv(model, iv_batches, batcher),
                  "iv_trips": int(sum(len(b) for b in iv_batches)), "iv_seconds": round(time.time() - t_iv, 1)})
    price_init = blob["price_init"]
    hier = None
    if args.hier_prior or "hier_prior" in blob:
        if fresh or "hier_prior" in blob:
            hier = HierPrior(model)
            if "hier_prior" in blob:
                hier.load(blob["hier_prior"])
        else:
            raise SystemExit("--hier-prior cannot be switched on when continuing a fit without it")
        price_init = {k: v for k, v in price_init.items() if k not in hier.blocks}
        emit({"event": "hier_prior_init", **{n: {"s2_mean": round(float(b["s2"].mean()), 5)}
                                            for n, b in hier.blocks.items()}})
    em_every = args.em_every or len(batches)
    # Anchored blocks: x0 = their values at the start of the fit (kept in the checkpoint, so a continued
    # fit anchors at the same values); an infinite precision freezes them.
    anchor_names = [n for b in args.anchor for n in ANCHOR_BLOCKS[b] if getattr(model, n, None) is not None]
    frozen = set(anchor_names) if args.anchor_prec == float("inf") else set()
    if anchor_names:
        blob.setdefault("anchor_init", {})
        for n in anchor_names:
            blob["anchor_init"].setdefault(n, getattr(model, n).detach().clone())
        price_init = {k: v for k, v in price_init.items() if k not in anchor_names}
        emit({"event": "anchor", "names": anchor_names, "prec": args.anchor_prec})
    anchor_init = {n: blob["anchor_init"][n] for n in anchor_names if n not in frozen}

    def anchor_penalty():
        total = 0.0
        for n, x0 in anchor_init.items():
            total = total + 0.5 * args.anchor_prec * (getattr(model, n) - x0).square().sum()
        return total / N
    priors = Priors(args.pool_prod, args.phi_prec, args.delta_prec, args.rho_c_prec, args.f_extra_prec,
                    args.price_prec, args.u_cat_prec, args.rec_prec, args.promo_dev_prec if promo_common else 0.0)
    rec_params = ([model.psi, model.psi_item] if recency else []) + (
        [model.hist_size] if model.hist_size is not None else [])
    promo_params = [model.dsp_common, model.mlr_common] if promo_common else []

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
        out.update(form="mixture", stage=stage, seconds=time.time() - started,
                   mixture_args={k: str(v) for k, v in vars(args).items()})
        if hier is not None:
            out["hier_prior"] = hier.state()
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
    groups = [([model.psi_cat, model.phi], args.lr_phi), ([model.rho_0_free], args.lr_rho0),
              ([model.rho_c], args.lr_rho), ([model.f_extra], args.lr_f_extra), ([model.lam], args.lr_lam),
              ([getattr(model, n) for n in CONTEXT_NAMES] + promo_params, args.lr_base),
              ([getattr(model, n) for n in price_names], args.lr_price), ([model.u_cat], args.lr_u_cat),
              (rec_params, args.lr_rec)]
    if args.interaction_only:
        groups = groups[:5]
    frozen_ids = {id(getattr(model, n)) for n in frozen}
    groups = [([q for q in params if id(q) not in frozen_ids], lr) for params, lr in groups]
    trained = [q for params, _ in groups for q in params]
    for q in trained:
        q.requires_grad_(True)
    opt = torch.optim.Adam([{"params": params, "lr": lr} for params, lr in groups if params])
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
        delta = model.phi[:, :r] - model.psi_cat[cat]
        opt.zero_grad(set_to_none=True)
        (-ll.mean()).backward()                                  # the likelihood alone first, so that
        if hier is not None:                                     # its gradients give the curvature
            hier.accumulate(len(pos))
        prior = penalties(model, priors, N, model.psi_cat, delta, price_init, recency)
        if hier is not None:
            prior = prior + hier.penalty(N)
        if anchor_init:
            prior = prior + anchor_penalty()
        prior.backward()
        grads = [model.rho_0_free.grad.detach().clone(), model.rho_c.grad.detach().clone(),
                 model.lam.grad.detach().clone()]
        g_sums = grads if g_sums is None else [a + b for a, b in zip(g_sums, grads)]
        g_count += 1
        torch.nn.utils.clip_grad_norm_(trained, 10.0)
        opt.step()
        project(model, capacities, opt)
        if step % args.every == 0:
            r0, rc, la = (g / g_count for g in g_sums)
            if not args.no_moment_match:
                with torch.no_grad():                            # moment matching
                    model.rho_0_free.sub_(0.5 * (r0 / rate_k).clamp(-0.5, 0.5))
                    model.rho_c.sub_(0.5 * (rc / rate_pairs).clamp(-0.5, 0.5))
                    model.lam.sub_(0.35 * (la / rate_item).clamp(-0.35, 0.35))
                project(model, capacities, opt)
            train_trace = float(torch.trace(zz / zz_n)) / r
            if args.no_whiten:                                   # still measured, not applied
                M = second_moment()
                eig = torch.linalg.eigvalsh(0.5 * (M + M.T)).flip(0)
            else:
                eig = mixture.whiten_(model, second_moment())    # parameter-expanded EM
                for prm in (model.phi, model.psi_cat):
                    opt.state.pop(prm, None)
            emit({"step": step, "lam_rel_err_mean": round(float((la / rate_item).abs().mean()), 4),
                  "whiten_eigs_top3": [round(float(v), 3) for v in eig[:3]],
                  "whiten_trace_over_rank": round(float(eig.sum() / r), 3),
                  "train_draws_trace_over_rank": round(train_trace, 3),
                  "steps_seconds": round(time.time() - t_block, 1)})
            g_sums, g_count, zz, zz_n, t_block = None, 0, torch.zeros(r, r), 0, time.time()
        if hier is not None and step % em_every == 0:
            emit({"event": "hier_prior", "step": step, **hier.update()})
        if competition and step % comp_every == 0:
            t_iv = time.time()
            emit({"event": "comp_iv", "step": step, **estimate_competition_iv(model, iv_batches, batcher),
                  "iv_seconds": round(time.time() - t_iv, 1)})
        if step % 10 == 0:
            emit({"step": step, "train_ll": round(float(ll.detach().mean()), 4), "ess": round(float(ess.mean()), 3),
                  "mode_grad": round(info["mode_grad"], 6)}, to_log=False)
        if step % args.val_every == 0 and step < steps:
            validate(step)
    for q in trained:
        q.requires_grad_(False)
    validate(steps)
    save(args.out.with_suffix(".pt"), f"mixture form, {steps} steps")
    emit({"event": "trained", "steps": steps, "best_val_ll": round(best, 4),
          "seconds": round(time.time() - started, 1)})


if __name__ == "__main__":
    main()
