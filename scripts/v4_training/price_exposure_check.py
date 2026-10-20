#!/usr/bin/env python3
"""Dunnhumby price checks, training weeks only.

1. Exposure: on purchased lines, compare the price the model is given (the chain's modal weekly
   price for the product) with the unit price actually paid, and the chain's discount flag
   (price below base_price by >= 5%) with the paid price's.
2. Slope: a within-category conditional logit on all training trips, with a fixed effect per
   product, the model's own price feature (log price deviation), display and mailer.  Each
   purchased product in a (trip, category) is one choice among that store's offered products.
   The paid price is not used here: it exists only for bought products, so it is outcome-selected.
"""
import argparse, json, os
from pathlib import Path
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(k, "4")
import numpy as np, pandas as pd, torch

ROOT = Path(os.environ["ENERGY_MODEL_DATA_ROOT"])


def exposure():
    items = pd.read_parquet(ROOT / "basket_input/items.parquet", columns=["PRODUCT_ID", "item_id"])
    b = pd.read_parquet(ROOT / "basket_input/baskets.parquet", columns=["item_id", "WEEK_NO", "store_id", "price", "split"])
    b = b[(b.split == "train") & (b.price > 0)].merge(items, on="item_id")
    pw = pd.read_parquet(ROOT / "data/price_week.parquet", columns=["PRODUCT_ID", "WEEK_NO", "price", "base_price"])
    m = b.merge(pw.rename(columns={"price": "chain_price"}), on=["PRODUCT_ID", "WEEK_NO"], how="inner")
    d = np.log(m.price) - np.log(m.chain_price)
    chain_disc = m.chain_price < 0.95 * m.base_price
    paid_disc = m.price < 0.95 * m.base_price
    lp = np.log(m.price); lc = np.log(m.chain_price)
    wi = lambda x: x - x.groupby(m.PRODUCT_ID).transform("mean")
    return {"purchased_lines": int(len(m)), "share_lines_matched": float(len(m) / len(b)),
            "paid_equals_chain_within_1pct": float((d.abs() < 0.01).mean()),
            "mean_abs_log_gap_paid_vs_chain": float(d.abs().mean()),
            "chain_discount_share": float(chain_disc.mean()), "paid_discount_share": float(paid_disc.mean()),
            "agreement_discount_flag": float((chain_disc == paid_disc).mean()),
            "P(paid discounted | chain discounted)": float(paid_disc[chain_disc].mean()),
            "P(paid discounted | chain not discounted)": float(paid_disc[~chain_disc].mean()),
            "corr_within_product_log_price_paid_vs_chain": float(np.corrcoef(wi(lp), wi(lc))[0, 1]),
            "regression_slope_paid_on_chain_within_product": float(np.polyfit(wi(lc), wi(lp), 1)[0])}


def week_panel(column, shape, transform):
    """A chain-week column of data/price_week.parquet as a (product, day) panel, carried across
    missing weeks like the model's price panel (scripts/data/22_basket_data.py)."""
    items = pd.read_parquet(ROOT / "basket_input/items.parquet", columns=["PRODUCT_ID", "item_id"])
    iid = dict(zip(items.PRODUCT_ID, items.item_id))
    pw = pd.read_parquet(ROOT / "data/price_week.parquet", columns=["PRODUCT_ID", "WEEK_NO", "price", "base_price", "promo_depth"])
    pw = pw[pw.PRODUCT_ID.isin(iid)]
    J, D = shape
    grid = np.full((J, 128), np.nan)
    grid[pw.PRODUCT_ID.map(iid).to_numpy(), pw.WEEK_NO.to_numpy()] = transform(pw).to_numpy()
    grid = pd.DataFrame(grid).ffill(axis=1).bfill(axis=1).fillna(0.0).to_numpy()
    b = pd.read_parquet(ROOT / "basket_input/baskets.parquet", columns=["DAY", "WEEK_NO"]).drop_duplicates("DAY")
    dw = np.full(D, np.nan); dw[b.DAY.to_numpy()] = b.WEEK_NO.to_numpy()
    dw = pd.Series(dw).ffill().bfill().to_numpy().astype(int)
    return torch.from_numpy(grid[:, dw].astype(np.float32))


PANELS = {   # extra slot features gathered through the price-panel lookup
    "promo_depth": lambda pw: pw.promo_depth,
    "log_price_over_base": lambda pw: np.log(pw.price / pw.base_price),
    "log_base_price": lambda pw: np.log(pw.base_price) - np.log(pw.base_price).groupby(pw.PRODUCT_ID).transform("mean"),
}


def slope(a):
    import stress
    data, model, meta, batcher = stress.setup(a.initialization, a.additive, 4)
    cache = np.load(ROOT / "basket_input/v3_index_affinity.npz")
    tr = np.flatnonzero((cache["trip_split"] == 0) & (np.diff(cache["line_ptr"]) > 0))
    trips = np.sort(np.random.default_rng(0).choice(tr, a.trips, replace=False)).astype(np.int64)
    J = int(data["n_item"])
    F = batcher.F
    dev0, spk, spv = F.dev, F.sp_key, F.sp_val
    panels = {n: week_panel(n, tuple(dev0.shape), f) for n, f in PANELS.items()}
    panels["model_price_feature_chain_only"] = dev0          # the model's panel without its store part
    extra = {n: [] for n in panels}
    X_item, X_dlp, X_disp, X_mail, X_row, Y = [], [], [], [], [], []
    row_off = 0
    for s in range(0, len(trips), 256):
        bt = stress.Batch(batcher, trips[s:s + 256], J); ix = bt.ix
        n_row = torch.zeros(ix.n_rows).index_add_(0, ix.row_of, bt.obs.double())
        keep = (n_row > 0)[ix.row_of]                      # categories bought on this trip
        X_item.append(ix.item[keep].numpy()); X_dlp.append(bt.ctx["dlp"][keep].numpy())
        X_disp.append(bt.ctx["disp"][keep].numpy()); X_mail.append(bt.ctx["mail"][keep].numpy())
        rows = ix.row_of[keep].numpy()
        _, rr = np.unique(rows, return_inverse=True)
        X_row.append(rr + row_off); row_off += rr.max() + 1
        Y.append(bt.obs[keep].numpy())
        F.sp_key, F.sp_val = torch.empty(0, dtype=spk.dtype), torch.empty(0, dtype=spv.dtype)
        for n, panel in panels.items():
            F.dev = panel
            extra[n].append(stress.Batch(batcher, trips[s:s + 256], J).ctx["dlp"][keep].numpy())
        F.dev, F.sp_key, F.sp_val = dev0, spk, spv
    extra = {n: np.concatenate(v) for n, v in extra.items()}
    item, dlp, disp, mail, row, y = (np.concatenate(v) for v in (X_item, X_dlp, X_disp, X_mail, X_row, Y))
    out = {"trips": len(trips), "choice_sets": int(row_off), "slots": int(len(item)), "chosen": int(y.sum())}

    def fit(names):
        cols = {"model_price_feature": dlp, **extra}
        it = torch.as_tensor(item); r = torch.as_tensor(row); yy = torch.as_tensor(y)
        X = torch.as_tensor(np.c_[tuple(cols[n] for n in names) + (disp, mail)], dtype=torch.float64)
        fe = torch.zeros(J, dtype=torch.float64, requires_grad=True)
        w = torch.zeros(X.shape[1], dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([fe, w], max_iter=500, line_search_fn="strong_wolfe", tolerance_grad=1e-9)
        from ragged import seg_max, seg_sum
        R = int(r.max()) + 1

        def nll():
            opt.zero_grad()
            u = fe[it] + X @ w
            m = seg_max(u.detach().unsqueeze(0), r, R)[0]
            lse = m + torch.log(seg_sum(torch.exp(u - m[r]).unsqueeze(0), r, R)[0])
            ll = (u * yy).sum() - (lse[r] * yy).sum()      # each chosen product: one choice from its set
            loss = -ll / yy.sum() + 1e-3 * fe.square().mean()
            loss.backward()
            return loss
        opt.step(nll)
        w_ = w.detach()
        res = {n: float(w_[i]) for i, n in enumerate(names)}
        res.update(display=float(w_[-2]), mailer=float(w_[-1]), loss=float(nll().detach()))
        print(json.dumps({"fit": names, **res}), flush=True)
        return res
    out["feature_sd_within_product"] = {n: float(pd.Series(v).groupby(item).transform(lambda x: x - x.mean()).std())
                                        for n, v in {"model_price_feature": dlp, **extra}.items()}
    store_part = dlp - extra["model_price_feature_chain_only"]
    out["store_price_part"] = {"share_nonzero_chosen_slots": float((np.abs(store_part[y]) > 1e-6).mean()),
                               "share_nonzero_other_slots": float((np.abs(store_part[~y]) > 1e-6).mean()),
                               "mean_on_chosen_slots": float(store_part[y].mean()), "mean_on_other_slots": float(store_part[~y].mean())}
    print(json.dumps(out["store_price_part"]), flush=True)
    out["fits"] = [fit(n) for n in (["model_price_feature"], ["model_price_feature_chain_only"], ["promo_depth"], ["log_price_over_base"],
                                     ["log_price_over_base", "log_base_price"], ["model_price_feature", "promo_depth"])]
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--initialization", type=Path, required=True)
    p.add_argument("--additive", type=Path, required=True)
    p.add_argument("--trips", type=int, default=16384)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    out = {"exposure": exposure()}
    print(json.dumps(out, indent=1), flush=True)
    out["within_category_logit"] = slope(a)
    print(json.dumps(out["within_category_logit"], indent=1), flush=True)
    a.output.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
