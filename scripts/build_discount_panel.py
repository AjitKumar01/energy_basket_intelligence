#!/usr/bin/env python3
"""Stage 2: build the (product x day) discount panel that splits the price input in two.

    depth[j, d] = min(log(price_jw / base_price_jw), 0)    w = the week of day d
    dsc[j, d]   = depth[j, d] - mean of depth[j, .] over training days

price is the chain's modal weekly price (the column the bundle's log_price_dev.npy is built from) and base_price the regular price of the same product-week in
data/price_week.parquet (or observed.npz regular_price on synthetic stress worlds).  Both
columns are carried across missing weeks exactly as the price panel is (forward, then
backward), so dlp = (dlp - dsc) + dsc splits each price deviation into a regular-price part
and a temporary-discount part.  dsc is centred on each product's training days exactly like
log_price_dev.npy: an uncentred depth averages about -0.15, so a larger discount multiplier
would also move every product's level and the basket size, which the product intercepts
would then have to undo.  Counterfactuals use only changes in dsc.
A dataset without a regular price has no panel and keeps one price slope.

The script checks that its carried price reproduces log_price_dev.npy up to the per-product
training mean before writing.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from v4.bundle import data_root  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, help="model-data bundle (default $ENERGY_MODEL_DATA_ROOT)")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    ROOT = data_root(a.data_root)
    dev = np.load(ROOT / "basket_input/log_price_dev.npy")
    J, D = dev.shape
    items = pd.read_parquet(ROOT / "basket_input/items.parquet", columns=["PRODUCT_ID", "item_id"])
    iid = dict(zip(items.PRODUCT_ID, items.item_id))
    pw = pd.read_parquet(ROOT / "data/price_week.parquet", columns=["PRODUCT_ID", "WEEK_NO", "price", "base_price"])
    pw = pw[pw.PRODUCT_ID.isin(iid)]
    rows, weeks = pw.PRODUCT_ID.map(iid).to_numpy(), pw.WEEK_NO.to_numpy()
    grids = {}
    for col in ("price", "base_price"):
        g = np.full((J, 128), np.nan)
        g[rows, weeks] = pw[col].to_numpy(np.float64)
        grids[col] = pd.DataFrame(g).ffill(axis=1).bfill(axis=1).to_numpy()
    obs_npz = ROOT.parent / "observed.npz"
    if obs_npz.is_file():
        with np.load(obs_npz) as z:
            if "regular_price" in z:
                reg = z["regular_price"].astype(np.float64)
                T = reg.shape[1]
                g_base = np.full((J, 128), np.nan)
                g_base[:, 1:T + 1] = reg
                grids["base_price"] = pd.DataFrame(g_base).ffill(axis=1).bfill(axis=1).to_numpy()
    if not all(np.isfinite(g).all() for g in grids.values()):
        raise SystemExit("weekly price grid has holes")
    # The day -> week map comes from all transactions, as for the price panel (the basket
    # file starts at the analysis window, so it cannot place the earlier days).
    tx_path = ROOT / "data/tx.parquet"
    if tx_path.is_file():
        b = pd.read_parquet(tx_path, columns=["DAY", "WEEK_NO"]).drop_duplicates()
        if b.groupby("DAY").WEEK_NO.nunique().max() != 1:
            raise SystemExit("DAY maps to more than one WEEK_NO")
        day_week = np.full(D, np.nan)
        day_week[b.DAY.to_numpy()] = b.WEEK_NO.to_numpy()
        day_week = pd.Series(day_week).ffill().bfill().to_numpy().astype(int)
    else:
        day_week = (np.arange(D) // 7) + 1

    log_price = np.log(np.clip(grids["price"], 1e-3, None))[:, day_week]
    offset = log_price - dev
    mismatch = float(np.abs(offset - offset.mean(1, keepdims=True)).max())
    if mismatch > 1e-4:
        raise SystemExit(f"carried price does not reproduce log_price_dev.npy (max {mismatch:.2e})")
    meta = json.loads((ROOT / "basket_input/meta.json").read_text())
    train_days = (day_week >= int(meta["analysis_first_week"])) & (day_week < int(meta["val_from_week"]))
    depth = np.minimum(np.log(grids["price"] / grids["base_price"]), 0.0)[:, day_week]
    dsc = (depth - depth[:, train_days].mean(1, keepdims=True)).astype(np.float32)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(a.output, dsc)
    summary = {"shape": [J, D], "price_panel_mismatch": mismatch,
               "train_days": int(train_days.sum()),
               "share_cells_discounted_5pct": float((depth < np.log(0.95)).mean()),
               "mean_depth": float(depth.mean()), "mean_depth_when_discounted": float(depth[depth < 0].mean()),
               "max_abs_train_mean_dsc": float(np.abs(dsc[:, train_days].mean(1)).max()),
               "sd_dlp": float(dev.std()), "sd_regular_part": float((dev - dsc).std()), "sd_dsc": float(dsc.std()),
               "corr_regular_dsc": float(np.corrcoef((dev - dsc).ravel(), dsc.ravel())[0, 1])}
    a.output.with_suffix(".json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
