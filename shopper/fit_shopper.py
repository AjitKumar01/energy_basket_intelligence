#!/usr/bin/env python3
"""Fit SHOPPER (shopper/shopper.py) by MAP with Adam on random item orderings.

Each step draws a minibatch of training trips, a fresh random ordering of each basket (the
paper's -shuffle 1), and maximises the ordering's log-probability plus the log prior divided by
the number of training trips.  Every --eval-every steps the validation trips are scored by the
basket log-probability log P(S) (importance sampling over orderings); the best validation
state is kept and training stops after --patience evaluations without improvement.

Data: --tsv DIR (the released code's format) or --bundle (the Version-4 model-data bundle at
$ENERGY_MODEL_DATA_ROOT, through the energy model's own Batcher).
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from shopper import Shopper  # noqa: E402


def load_source(a):
    if a.tsv is not None:
        from shopper_data import TsvSource
        return TsvSource(a.tsv, norm_price=a.norm_price, seed=a.seed)
    from shopper_data import BundleSource
    return BundleSource(price_contract=a.price_contract, discount_panel=a.discount_panel)


@torch.no_grad()
def score(model, src, trips, chunk, perms, seed):
    gen = torch.Generator().manual_seed(seed)
    out = []
    for s in range(0, len(trips), chunk):
        out.append(model.basket_logprob_is(src.make(trips[s:s + chunk]), perms=perms, gen=gen))
    return torch.cat(out).double().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tsv", type=Path)
    p.add_argument("--bundle", action="store_true")
    p.add_argument("--price-contract", choices=("chain_and_store", "chain_product_week_only"))
    p.add_argument("--discount-panel", type=Path)
    p.add_argument("--norm-price", type=int, default=1)
    p.add_argument("--K", type=int, default=50)
    p.add_argument("--P", type=int, default=10)
    p.add_argument("--D", type=int, default=10)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--shp", type=float, default=100.0, help="gamma prior shape on the price factors (paper default)")
    p.add_argument("--rte", type=float, default=1000.0, help="gamma prior rate on the price factors (paper default)")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--iters", type=int, default=20000)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--val-trips", type=int, default=2048)
    p.add_argument("--val-perms", type=int, default=32)
    p.add_argument("--chunk", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--double", action="store_true")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    torch.set_default_dtype(torch.float64 if a.double else torch.float32)
    torch.manual_seed(a.seed)
    src = load_source(a)
    train, val = src.split(0), src.split(1)
    rng = np.random.default_rng(a.seed)
    val = np.sort(rng.choice(val, min(a.val_trips, len(val)), replace=False))
    model = Shopper(src.n_item, src.n_user, src.n_week, src.item_group, K=a.K, P=a.P, D=a.D, seed=a.seed)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    gen = torch.Generator().manual_seed(a.seed + 1)
    log = {"args": {k: str(v) for k, v in vars(a).items()}, "n_train": int(len(train)), "evals": []}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    best, best_state, since, t0 = -math.inf, None, 0, time.time()
    run_loss = []
    for it in range(1, a.iters + 1):
        batch = src.make(np.sort(rng.choice(train, a.batch, replace=False)))
        lp = model.sequence_logprob(batch, model.random_order(batch, gen))
        loss = -lp.mean() + model.neg_log_prior(shp=a.shp, rte=a.rte) / len(train)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        run_loss.append(float(lp.mean()))
        if it % a.eval_every == 0 or it == a.iters:
            v = score(model, src, val, a.chunk, a.val_perms, a.seed + 2)
            row = {"iter": it, "train_order_ll": float(np.mean(run_loss[-a.eval_every:])),
                   "val_basket_ll": float(v.mean()), "val_se": float(v.std(ddof=1) / math.sqrt(len(v))),
                   "sec": round(time.time() - t0, 1)}
            log["evals"].append(row)
            print(json.dumps(row), flush=True)
            if row["val_basket_ll"] > best + 1e-3:
                best, since = row["val_basket_ll"], 0
                best_state = {k: x.detach().clone() for k, x in model.state_dict().items()}
            else:
                since += 1
                if since >= a.patience:
                    break
            a.out.with_suffix(".json").write_text(json.dumps(log, indent=1))
    model.load_state_dict(best_state)
    log["best_val_basket_ll"] = best
    torch.save({"model": model.state_dict(), "config": vars(a), "n_item": src.n_item, "n_user": src.n_user,
                "n_week": src.n_week, "item_group": src.item_group, "best_val_basket_ll": best}, a.out)
    a.out.with_suffix(".json").write_text(json.dumps(log, indent=1))
    print(json.dumps({"done": True, "best_val_basket_ll": best}), flush=True)


if __name__ == "__main__":
    main()
