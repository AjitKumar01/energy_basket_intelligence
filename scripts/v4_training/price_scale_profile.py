#!/usr/bin/env python3
"""Does the model's own likelihood want a stronger price response?

Multiplies the price coefficient g_hj of a Phi = 0 (additive) checkpoint by s and reports the
exact mean log-likelihood per trip (exact DP normaliser) on a sample of training and validation
trips, for each s.  The own-price deviation slope g*kappa and the common shift g both scale.
--what promo scales the display and mailer weights instead; --what discount leaves the model
alone and scales only the temporary-discount part log(price / base_price) of the price feature
(regular-price movement unchanged), which asks whether discounts deserve a steeper slope than
regular price changes; --what kappa scales only the relative-price multiplier softplus(kappa).  If the optimum is far above
s = 1, the price weights are under-fitted; if it is near 1, the full-basket likelihood itself
does not ask for a stronger response.
"""
import argparse, json, os
from pathlib import Path
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(k, "4")
import numpy as np, torch
import stress


def discount_panel(data, shape):
    """log(price / base_price) per (product, day) from the chain-week price file, carried across
    missing weeks exactly as the price panel is (scripts/data/22_basket_data.py)."""
    import pandas as pd
    root = Path(os.environ["ENERGY_MODEL_DATA_ROOT"])
    items = pd.read_parquet(root / "basket_input/items.parquet", columns=["PRODUCT_ID", "item_id"])
    iid = dict(zip(items.PRODUCT_ID, items.item_id))
    pw = pd.read_parquet(root / "data/price_week.parquet", columns=["PRODUCT_ID", "WEEK_NO", "price", "base_price"])
    pw = pw[pw.PRODUCT_ID.isin(iid)]
    J, D = shape
    grid = np.full((J, 128), np.nan)
    grid[pw.PRODUCT_ID.map(iid).to_numpy(), pw.WEEK_NO.to_numpy()] = np.log(pw.price / pw.base_price).to_numpy()
    grid = pd.DataFrame(grid).ffill(axis=1).bfill(axis=1).fillna(0.0).to_numpy()
    b = pd.read_parquet(root / "basket_input/baskets.parquet", columns=["DAY", "WEEK_NO"]).drop_duplicates("DAY")
    day_week = np.full(D, np.nan); day_week[b.DAY.to_numpy()] = b.WEEK_NO.to_numpy()
    day_week = pd.Series(day_week).ffill().bfill().to_numpy().astype(int)
    out = torch.from_numpy(grid[:, day_week].astype(np.float32))
    print(json.dumps({"discount_panel_mean": float(out.mean()), "share_discounted": float((out < -1e-6).double().mean())}), flush=True)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--initialization", type=Path, required=True)
    p.add_argument("--additive", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, default=None, help="optional stress.py checkpoint (its Phi is ignored)")
    p.add_argument("--scales", type=float, nargs="+", default=[0.5, 1, 2, 3, 4, 6, 8])
    p.add_argument("--what", choices=("price", "promo", "discount", "kappa"), default="price")
    p.add_argument("--trips", type=int, default=4096)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    data, model, meta, batcher = stress.setup(a.initialization, a.additive, 4)
    if a.checkpoint is not None:
        blob = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
        stress.configure_model_from_blob(model, blob)
        with torch.no_grad():
            model.phi.zero_()
    model.eval()
    for q in model.parameters():
        q.requires_grad_(False)
    cache = np.load(Path(os.environ["ENERGY_MODEL_DATA_ROOT"]) / "basket_input/v3_index_affinity.npz")
    nonempty = np.diff(cache["line_ptr"]) > 0
    rng = np.random.default_rng(0)
    J = int(data["n_item"])
    orig_pc = model.price_coefficients
    kappa0 = model.price_kappa.detach().clone()
    dev0 = batcher.F.dev.clone()
    discount = discount_panel(data, dev0.shape) if a.what == "discount" else None
    w_dsp, w_mlr = model.w_dsp.detach().clone(), model.w_mlr.detach().clone()
    out = {"what": a.what, "additive": str(a.additive), "checkpoint": str(a.checkpoint), "by_split": {}}
    for split_name, code in (("train", 0), ("validation", 1)):
        pool = np.flatnonzero((cache["trip_split"] == code) & nonempty)
        trips = np.sort(rng.choice(pool, min(a.trips, len(pool)), replace=False)).astype(np.int64)
        batches = [stress.Batch(batcher, trips[s:s + 256], J) for s in range(0, len(trips), 256)]
        rows = []
        for s in a.scales:
            if a.what == "discount":
                batcher.F.dev = dev0 + (s - 1.0) * discount
                batches = [stress.Batch(batcher, trips[i:i + 256], J) for i in range(0, len(trips), 256)]
            elif a.what == "kappa":
                with torch.no_grad():
                    k = s * torch.nn.functional.softplus(kappa0)
                    model.price_kappa.copy_(k + torch.log(-torch.expm1(-k)))      # inverse softplus
            elif a.what == "price":
                model.price_coefficients = (lambda it, trip, s=s: s * orig_pc(it, trip))
            else:
                with torch.no_grad():
                    model.w_dsp.copy_(s * w_dsp); model.w_mlr.copy_(s * w_mlr)
            ll = []
            with torch.no_grad():
                for bt in batches:
                    bt.bind(model)
                    u = model.b_flat(bt.ix)
                    ll.append((stress.obs_energy(model, bt, 0) - stress.log_zc(model, bt.ix, u)).numpy())
            ll = np.concatenate(ll)
            rows.append({"scale": s, "mean_ll": float(ll.mean()), "se": float(ll.std() / np.sqrt(len(ll)))})
            print(json.dumps({"split": split_name, **rows[-1]}), flush=True)
        model.price_coefficients = orig_pc
        batcher.F.dev = dev0
        with torch.no_grad():
            model.w_dsp.copy_(w_dsp); model.w_mlr.copy_(w_mlr); model.price_kappa.copy_(kappa0)
        best = max(rows, key=lambda r: r["mean_ll"])
        out["by_split"][split_name] = {"trips": len(trips), "rows": rows, "best_scale": best["scale"],
                                       "gain_at_best_over_1": best["mean_ll"] - next(r for r in rows if r["scale"] == 1)["mean_ll"]}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
