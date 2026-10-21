#!/usr/bin/env python3
"""Stage 2b: create the untrained starting model (no optimisation, training trips only).

    sizes      nmax = largest training basket (held-out baskets beyond it stop the run
               unless --nmax is given); R = nmax products per category
    lam        log of each product's purchase rate per trip in which it was offered,
               centred (a common shift is the same as a linear change of rho_0)
    taste      theta, alpha from a rank-(K-1) SVD of smoothed household-by-product log
               shares; the last coordinate is each household's size shift (its alpha is
               fixed at 1), so the remaining loadings are centred over products
    rho_0      set so that, at the starting utilities, the model's average size law over 64
               training trips equals the data's
    other      price slopes, season, store: small random values from --seed; promotions and
               phi start at zero

Writes the initialization file used by fit_additive.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_default_dtype(torch.float64)

from v4.batches import Batcher  # noqa: E402
from v4.bundle import data_fingerprint, data_root, item_categories, load_bundle  # noqa: E402
from v4.checkpoints import save_initialization  # noqa: E402
from v4.exact import additive_log_size  # noqa: E402
from v4.features import Features  # noqa: E402
from v4.model import BasketModel  # noqa: E402


def popularity_logits(data, trips):
    """log((purchases + 0.5) / (trips offering the product + 1)), centred."""
    J, C, S = (int(data[k]) for k in ("n_item", "n_cat", "n_store"))
    count = np.zeros(J)
    for t in trips:
        count[data["line_item"][data["line_ptr"][t]:data["line_ptr"][t + 1]]] += 1.0
    exposure = np.zeros(J)
    per_store = np.bincount(data["trip_store"][trips], minlength=S)
    ptr, items = data["store_cat_ptr"], data["store_items"]
    for s in range(S):
        exposure[items[int(ptr[s * C]):int(ptr[(s + 1) * C])]] += per_store[s]
    seen = exposure > 0
    value = np.empty(J)
    value[seen] = np.log((count[seen] + 0.5) / (exposure[seen] + 1.0))
    value[~seen] = np.median(value[seen])
    return torch.as_tensor(value - value.mean())


@torch.no_grad()
def initialize_taste(model, data, trips, prior=100.0, clip=3.0, seed=0):
    """theta_h . alpha_j from household log shares (moment equation at weak interaction).

    residual[h, j] = log((n_hj + prior * share_j) / (n_h + prior)) - log share_j, with the
    product's mean (over trips) and the household's share-weighted mean removed and clipped;
    its trip-weighted rank-(K-1) SVD gives the composition taste.  The household size shift
    (last coordinate) is set so the taste does not change each household's expected basket
    size over the products it was offered (weighted by exp(lam))."""
    from scipy import sparse
    from sklearn.utils.extmath import randomized_svd

    N, J, K = model.theta.shape[0], model.alpha.shape[0], model.K
    keep = np.zeros(len(data["trip_split"]), dtype=bool)
    keep[np.asarray(trips)] = True
    line_trip = np.repeat(np.arange(len(data["line_ptr"]) - 1), np.diff(data["line_ptr"]))
    keep = keep[line_trip]
    h = data["trip_user"][line_trip[keep]].astype(np.int64)
    j = data["line_item"][keep].astype(np.int64)
    count = np.zeros((N, J), dtype=np.float32)
    np.add.at(count, (h, j), 1.0)
    share = count.sum(0) + np.float32(0.5)
    share /= share.sum()
    residual = np.log((count + np.float32(prior) * share[None, :])
                      / (count.sum(1)[:, None] + np.float32(prior))) - np.log(share[None, :])
    weight = np.bincount(data["trip_user"][trips], minlength=N).astype(np.float32)
    weight /= max(float(weight.mean()), 1e-12)
    residual -= (residual @ share)[:, None]
    residual -= ((weight[:, None] * residual).sum(0) / np.maximum(weight.sum(), 1e-12))[None, :]
    np.clip(residual, -clip, clip, out=residual)
    U, sv, Vt = randomized_svd(residual * np.sqrt(weight[:, None]), n_components=K - 1, n_iter=4,
                               random_state=seed)
    root = np.sqrt(np.maximum(sv, 0.0))
    theta_c = U * root[None, :] / np.sqrt(weight[:, None]).clip(min=1e-12)
    alpha_c = Vt.T * root[None, :]
    fitted = theta_c @ alpha_c.T

    # Each household's log-mean of exp(taste) over the products its stores offered.
    S, C = int(data["n_store"]), int(data["n_cat"])
    visits = sparse.coo_matrix((np.ones(len(trips), dtype=np.float32),
                                (data["trip_user"][trips], data["trip_store"][trips])), shape=(N, S)).tocsr()
    rows, cols = [], []
    ptr, items = data["store_cat_ptr"], data["store_items"]
    for s in range(S):
        lo, hi = int(ptr[s * C]), int(ptr[(s + 1) * C])
        rows.extend([s] * (hi - lo))
        cols.extend(items[lo:hi].tolist())
    offered = sparse.coo_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(S, J)).tocsr()
    exposure = (visits @ offered).tocoo()
    w = exposure.data * np.exp(model.lam.numpy().clip(-30.0, 30.0))[exposure.col]
    den = np.bincount(exposure.row, weights=w, minlength=N).clip(min=1e-30)
    num = np.bincount(exposure.row, weights=w * np.exp(fitted[exposure.row, exposure.col].clip(-30.0, 30.0)),
                      minlength=N).clip(min=1e-30)
    common = np.log(num / den)

    alpha_bar = alpha_c.mean(0)
    theta = np.zeros((N, K), dtype=theta_c.dtype)
    alpha = np.zeros((J, K), dtype=alpha_c.dtype)
    theta[:, :K - 1], alpha[:, :K - 1] = theta_c, alpha_c - alpha_bar[None, :]
    theta[:, -1] = theta_c @ alpha_bar - common
    alpha[:, -1] = 1.0
    mean = theta.mean(0)                 # the model centres theta; move that offset into lam
    theta -= mean[None, :]
    model.theta.copy_(torch.as_tensor(theta))
    model.alpha.copy_(torch.as_tensor(alpha))
    model.lam.add_(torch.as_tensor(alpha @ mean))


@torch.no_grad()
def initialize_size_potential(model, data, trips, batcher, nmax):
    """rho_0(n) = mean log A_n - log P_data(n), with A_n the total weight of size-n baskets
    at the starting utilities (rho_0 = 0) averaged over 64 training trips."""
    count = np.bincount(np.clip(data["trip_nlines"][trips], 0, nmax), minlength=nmax + 1) + 0.5
    target = torch.log(torch.as_tensor(count / count.sum()))
    sample = trips[np.random.default_rng(0).choice(len(trips), size=64, replace=False)]
    batch = batcher.make(sample)
    batch.bind(model)
    log_a = torch.cat([torch.zeros(1), additive_log_size(model, batch.ix).mean(0)])
    if not bool(torch.isfinite(log_a).all()):
        raise RuntimeError("some basket size is impossible at the sampled trips")
    rho = log_a - target
    model.rho_0_free.copy_((rho - rho[0])[1:])


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data-root", type=Path, help="model-data bundle (default $ENERGY_MODEL_DATA_ROOT)")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--K", type=int, default=32, help="taste rank")
    p.add_argument("--Kz", type=int, default=32, help="columns of phi")
    p.add_argument("--Kp", type=int, default=8, help="rank of the price slopes")
    p.add_argument("--nmax", type=int, help="largest basket size (default: largest training basket)")
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    root = data_root(args.data_root)
    data = load_bundle(root)
    manifest = json.loads((root / "basket_input/affinity_manifest.json").read_text())
    partition = root / "basket_input/items_affinity.parquet"
    if (not manifest.get("training_only") or int(manifest["n_items"]) != int(data["n_item"])
            or manifest["partition_sha256"] != hashlib.sha256(partition.read_bytes()).hexdigest()):
        raise SystemExit("the category partition does not match its training-only manifest")

    training = np.flatnonzero(data["trip_split"] == 0)
    sizes = data["trip_nlines"]
    nmax = int(sizes[training].max()) if args.nmax is None else args.nmax
    outside = {name: int(((data["trip_split"] == code) & (sizes > nmax)).sum())
               for code, name in ((1, "validation"), (2, "test"))}
    if any(outside.values()) and args.nmax is None:
        raise SystemExit(f"held-out baskets exceed 1..{nmax}: {outside}; pass --nmax")
    model = BasketModel(int(data["n_item"]), int(data["n_user"]), int(data["n_cat"]), K=args.K, Kz=args.Kz,
                        Kp=args.Kp, nmax=nmax, R=nmax, S=int(data["n_store"]), seed=args.seed, taste_init=0.03)
    with torch.no_grad():
        model.cat_of.copy_(torch.as_tensor(item_categories(data)))
        model.lam.copy_(popularity_logits(data, training))
    initialize_taste(model, data, training, seed=args.seed)
    features = Features(root, int(data["n_item"]), int(data["n_store"]), recency=False,
                        price_contract="chain_product_week_only")
    initialize_size_potential(model, data, training, Batcher(data, features), nmax)

    metadata = {"J": int(data["n_item"]), "N": int(data["n_user"]), "C": int(data["n_cat"]),
                "S": int(data["n_store"]), "K": args.K, "Kz": args.Kz, "Kp": args.Kp, "nmax": nmax, "R": nmax,
                "seed": args.seed, "heldout_trips_outside_support": outside, "household_size_rank1": True,
                "no_rec": True, "data_fingerprint_sha256": data_fingerprint(root)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_initialization(args.output, model, metadata)
    print(f"[initialize] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
