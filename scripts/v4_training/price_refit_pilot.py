#!/usr/bin/env python3
"""Pilot: refit only the price-related weights of an additive parent on a changed price input.

Loads the exact additive parent, rebuilds the price panel under the chosen contract
(default chain_product_week_only: the chain's weekly price, no store deviations), freezes
every parameter except the product intercepts (lam), the price factors (gamma, beta,
price_kappa) and the display/mailer weights, and maximises the same exact-normaliser
objective as fit_exact_additive.py (size penalties and, optionally, the uniform-price
elasticity penalty).  Validation early stopping on the parent's validation panel.

The saved checkpoint has the parent's layout ({"model": state_dict, ...}) plus
supported_price_component.price_feature_contract, so loaders pick up the same price panel.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/version4"))
torch.set_default_dtype(torch.float64)

from data import build                                     # noqa: E402
from features import Features                              # noqa: E402
from fit import Batcher                                    # noqa: E402
from fit_exact_additive import exact_loglik_and_size, load_model, validate  # noqa: E402
from pipeline_support import supported_trips               # noqa: E402
from provenance import file_sha256                         # noqa: E402
from price_response import additive_uniform_price_response  # noqa: E402

TRAINED = ("lam", "gamma", "beta", "price_kappa", "w_dsp", "w_mlr")
GLOBAL = ("price_kappa", "disc_kappa", "disc_common")   # one scalar each: the larger learning rate


def price_summary(model):
    with torch.no_grad():
        g = (torch.nn.functional.softplus(model.gamma) @ torch.nn.functional.softplus(model.beta).T)
        kappa = float(torch.nn.functional.softplus(model.price_kappa))
        extra = {}
        if getattr(model, "disc_kappa", None) is not None or getattr(model, "disc_common", None) is not None:
            kd = (float(torch.nn.functional.softplus(model.disc_kappa))
                  if getattr(model, "disc_kappa", None) is not None else kappa)
            c0 = float(model.disc_common) if getattr(model, "disc_common", None) is not None else 0.0
            extra = {"disc_kappa": kd, "disc_common": c0,
                     "mean_own_discount_slope": float(g.mean()) * kd + c0}
        return {"kappa": kappa, **extra, "mean_g": float(g.mean()), "mean_own_slope": float(g.mean()) * kappa,
                "median_product_own_slope": float(g.mean(0).median()) * kappa,
                "w_dsp_mean": float(model.w_dsp.mean()), "w_mlr_mean": float(model.w_mlr.mean()),
                "w_dsp_share_negative": float((model.w_dsp < 0).double().mean())}


@torch.no_grad()
def uniform_elasticity(model, batcher, trips, chunk):
    out = []
    for s in range(0, len(trips), chunk):
        ix, ctx, line_ctx, house, li, lt, lc, _ = batcher.make(trips[s:s + chunk])
        model.house, model.ctx = house, ctx
        _, size_p = exact_loglik_and_size(model, ix, li, lt, lc, line_ctx)
        out.append(float(additive_uniform_price_response(model, ix, size_p).elasticity))
    return float(np.mean(out))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--initialization", type=Path, required=True)
    p.add_argument("--additive", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True, help="checkpoint path; .json log alongside")
    p.add_argument("--contract", default="chain_product_week_only",
                   choices=("chain_product_week_only", "chain_and_store"))
    p.add_argument("--discount-panel", type=Path,
                   help="split price into regular and discount parts with this panel "
                        "(build_discount_panel.py); the discount multiplier starts equal to kappa")
    p.add_argument("--no-disc-kappa", action="store_true",
                   help="with --discount-panel: no separate discount multiplier")
    p.add_argument("--disc-common", action="store_true",
                   help="with --discount-panel: add a common discount slope, starting at 0")
    p.add_argument("--elast-w", type=float, default=0.0)
    p.add_argument("--elast-target", type=float, default=-0.121)
    p.add_argument("--iters", type=int, default=4000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.01, help="learning rate for the global price_kappa")
    p.add_argument("--lr-unit", type=float, default=5e-4,
                   help="learning rate for per-product/per-household weights (the parent ended at 6.25e-5)")
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--patience", type=int, default=10, help="evaluations without improvement before stopping")
    p.add_argument("--lr-patience", type=int, default=3)
    p.add_argument("--min-delta", type=float, default=1e-3)
    p.add_argument("--validation-trips", type=int, default=1024)
    p.add_argument("--chunk", type=int, default=128)
    p.add_argument("--seed", type=int, default=29001)
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)

    data = build()
    model, meta, _ = load_model(a.initialization, data)
    parent = torch.load(a.additive, map_location="cpu", weights_only=False)
    model.load_state_dict(parent["model"])
    cfg = parent["config"]
    with torch.no_grad():
        model.phi.zero_()
    trained = TRAINED
    if a.discount_panel is not None:
        # Starting at disc_kappa = price_kappa and disc_common = 0 reproduces the parent exactly.
        if not a.no_disc_kappa:
            model.disc_kappa = torch.nn.Parameter(model.price_kappa.detach().clone())
            trained = trained + ("disc_kappa",)
        if a.disc_common:
            model.disc_common = torch.nn.Parameter(torch.zeros(()))
            trained = trained + ("disc_common",)
    for name, q in model.named_parameters():
        q.requires_grad_(name in trained)
    params = [getattr(model, n) for n in trained]
    nmax = int(meta["nmax"])
    no_rec = bool(meta.get("no_rec", False))

    def make_batcher(store):
        f = Features(int(data["n_item"]), int(data["n_store"]), include_recency=not no_rec,
                     include_store_price=store, discount_panel=a.discount_panel or False)
        return Batcher(data, f, nmax, include_recency=not no_rec)

    batcher = make_batcher(a.contract == "chain_and_store")
    train = supported_trips(data, 0, nmax)
    counts = np.bincount(np.clip(data["trip_nlines"][train], 1, nmax), minlength=nmax + 1)[1:].astype(float)
    empirical_size = torch.as_tensor(counts / counts.sum())
    vpop = supported_trips(data, 1, nmax)
    valid = vpop[np.random.default_rng(int(cfg["seed"]) + 1).permutation(len(vpop))[:a.validation_trips]]
    rng = np.random.default_rng(a.seed + 7)

    log = {"args": {k: str(v) for k, v in vars(a).items()}, "trained": trained, "evals": []}
    out_json = a.output.with_suffix(".json")
    a.output.parent.mkdir(parents=True, exist_ok=True)

    def record(it, extra=None):
        v = validate(model, batcher, valid, a.chunk)
        row = {"iter": it, "val_ll": v["basket_loglik"], "val_se": v["standard_error"],
               "exp_size": v["expected_size_mean"], "obs_size": v["observed_size_mean"],
               **price_summary(model), "sec": round(time.perf_counter() - t0, 1), **(extra or {})}
        log["evals"].append(row)
        print(json.dumps(row), flush=True)
        return row

    t0 = time.perf_counter()
    # The parent scored with its own (store-deviation) panel, for reference.
    if a.contract != "chain_and_store":
        ref = validate(model, make_batcher(True), valid, a.chunk)
        log["parent_val_ll_own_panel"] = ref["basket_loglik"]
        print(json.dumps({"parent_val_ll_own_panel": ref["basket_loglik"]}), flush=True)
    log["start_uniform_elasticity"] = uniform_elasticity(model, batcher, valid[:512], a.chunk)
    best = record(0, {"uniform_elasticity": log["start_uniform_elasticity"]})
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    glob = [q for n, q in zip(trained, params) if n in GLOBAL]
    unit = [q for n, q in zip(trained, params) if n not in GLOBAL]
    opt = torch.optim.AdamW([{"params": glob, "lr": a.lr}, {"params": unit, "lr": a.lr_unit}],
                            weight_decay=float(cfg["weight_decay"]))
    since, since_lr = 0, 0
    for it in range(1, a.iters + 1):
        trips = train[rng.choice(len(train), size=a.batch, replace=False)]
        ix, ctx, line_ctx, house, li, lt, lc, _ = batcher.make(trips)
        model.house, model.ctx = house, ctx
        opt.zero_grad(set_to_none=True)
        score, size_p = exact_loglik_and_size(model, ix, li, lt, lc, line_ctx)
        loss = -score.mean()
        pbar = size_p.mean(0).clamp_min(1e-12)
        pbar = pbar / pbar.sum()
        target = empirical_size[:pbar.numel()]
        size_pen = -(target * pbar.log()).sum()
        smooth = (target + float(cfg["rkl_eps"])) / (target + float(cfg["rkl_eps"])).sum()
        size_pen = size_pen + float(cfg["rkl_w"]) * (pbar * (pbar.log() - smooth.log())).sum()
        loss = loss + float(cfg["size_kl"]) * size_pen
        if a.elast_w > 0:
            el = additive_uniform_price_response(model, ix, size_p, step=float(cfg["elast_step"])).elasticity
            loss = loss + a.elast_w * (el - a.elast_target).square()
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite loss at {it}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, float(cfg["clip"]))
        opt.step()
        if int(cfg["lam_centre"]):
            with torch.no_grad():   # exact gauge move, as in fit_exact_additive.py
                m = model.lam.mean().clone()
                model.lam.sub_(m)
                model.rho_0_free.sub_(m * torch.arange(1, model.rho_0_free.numel() + 1, dtype=model.lam.dtype))
        if it % a.eval_every == 0:
            row = record(it)
            if row["val_ll"] > best["val_ll"] + a.min_delta:
                best, since, since_lr = row, 0, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                since += 1
                since_lr += 1
                if since_lr >= a.lr_patience:
                    for g in opt.param_groups:
                        g["lr"] *= 0.5
                    since_lr = 0
                    print(json.dumps({"iter": it, "lr": [g["lr"] for g in opt.param_groups]}), flush=True)
                if since >= a.patience:
                    break
            out_json.write_text(json.dumps(log, indent=1))
    model.load_state_dict(best_state)
    log["best"] = best
    log["best_uniform_elasticity"] = uniform_elasticity(model, batcher, valid[:512], a.chunk)
    log["stopped_iter"] = it
    blob = {k: v for k, v in parent.items() if k not in ("model", "optimizer")}
    blob.update(model=model.state_dict(), best_validation=best["val_ll"], best_iteration=best["iter"],
                supported_price_component={"price_feature_contract": a.contract},
                discount_panel=str(a.discount_panel) if a.discount_panel else None,
                discount_panel_sha256=file_sha256(a.discount_panel) if a.discount_panel else None,
                price_refit_pilot={"parent": str(a.additive), "trained": trained,
                                   "elast_w": a.elast_w, "elast_target": a.elast_target})
    torch.save(blob, a.output)
    out_json.write_text(json.dumps(log, indent=1))
    print(json.dumps({"done": True, "best": best, "best_uniform_elasticity": log["best_uniform_elasticity"]}), flush=True)


if __name__ == "__main__":
    main()
