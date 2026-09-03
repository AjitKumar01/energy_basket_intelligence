"""
Stage 23 -- Promotional placement, from dunnhumby's causal_data.csv.

HANDOFF called this the highest-value open step, and the reason is that nothing else in
this dataset moves price for a reason that is plausibly outside the household's own
demand.  The file records, per PRODUCT_ID x STORE_ID x WEEK_NO:

    display   in-store placement   0 none, 1 store front, 2 store rear, 3 front end cap,
                                   4 mid-aisle end cap, 5 rear end cap, 6 side-aisle end
                                   cap, 7 in-aisle, 9 secondary location, A in-shelf
    mailer    weekly circular      0 not on ad, A interior page feature, C interior line
                                   item, D front page feature, F front page line item,
                                   H wrap front, J wrap interior coupon, L wrap back,
                                   P back page, X/Z free-standing

ONE STRUCTURAL FACT decides how it is encoded.  Every row of causal_data has display != 0
OR mailer != 0 -- measured, not assumed: 100.0% of the 6,522,942 rows on our products.
Within the file's observed week range, absence means not promoted.  The raw file has no
coverage in weeks 1--8 or 102, so absence there is *missing*, not zero.  Stage 22 excludes
those weeks and this stage records and verifies the coverage boundary.

This stage does NOT rebuild basket_input.  Re-running 22 would renumber items, stores and
splits and invalidate every fitted model, so the promo panel is written alongside as
promo.npz and the store map is recovered exactly by joining BASKET_ID against data/tx.

Writes basket_input/promo.npz with the same sparse-key layout store_price.npz uses:
key = (item * n_stores + store) * 128 + week, sorted, with int8 disp/mail beside it.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "data")
IN = os.path.join(HERE, "..", "..", "basket_input")
RAW = os.path.join(os.environ.get(
    "NF_RAW_DIR",
    os.path.join(HERE, "..", "..", "..", "dunnhumby_The-Complete-Journey",
                 "dunnhumby_The-Complete-Journey CSV")), "")


def log(m):
    print(f"[23] {m}", flush=True)


def main(a):
    global IN
    if a.indir:
        IN = os.path.abspath(a.indir)
    meta = json.load(open(os.path.join(IN, "meta.json")))
    n_stores = int(meta["n_stores"])
    items = pd.read_parquet(os.path.join(IN, "items.parquet"))
    iid = items.set_index("PRODUCT_ID").item_id
    bk = pd.read_parquet(os.path.join(IN, "baskets.parquet"),
                         columns=["BASKET_ID", "store_id"]).drop_duplicates("BASKET_ID")

    # Recover raw STORE_ID -> store_id exactly, by joining the basket ids that stage 22
    # already numbered.  Reconstructing 22's mode-of-basket rule independently would risk
    # a silent off-by-one that no downstream check would catch.
    tx = pd.read_parquet(os.path.join(DATA, "tx.parquet"),
                         columns=["BASKET_ID", "STORE_ID"]).drop_duplicates("BASKET_ID")
    mp = bk.merge(tx, on="BASKET_ID", how="inner")
    smap = mp.groupby("STORE_ID").store_id.agg(lambda s: s.mode().iat[0])
    log(f"recovered {len(smap)} of {n_stores} store ids by BASKET_ID join")
    if len(smap) != n_stores:
        log("  WARNING: store map is incomplete; unmapped stores contribute no promo")

    keep = set(items.PRODUCT_ID.astype(np.int64))
    rows, n_raw, coverage_min, coverage_max = [], 0, None, None
    for ch in pd.read_csv(RAW + "causal_data.csv",
                          usecols=["PRODUCT_ID", "STORE_ID", "WEEK_NO", "display", "mailer"],
                          dtype={"PRODUCT_ID": np.int64, "STORE_ID": np.int64,
                                 "WEEK_NO": np.int16, "display": str, "mailer": str},
                          chunksize=a.chunk):
        n_raw += len(ch)
        lo, hi = int(ch.WEEK_NO.min()), int(ch.WEEK_NO.max())
        coverage_min = lo if coverage_min is None else min(coverage_min, lo)
        coverage_max = hi if coverage_max is None else max(coverage_max, hi)
        rows.append(ch[ch.PRODUCT_ID.isin(keep)])
    c = pd.concat(rows, ignore_index=True)
    del rows
    log(f"causal_data: {n_raw:,} rows, {len(c):,} on our {len(keep):,} products")
    required = meta.get("promotion_coverage_required")
    if required != [coverage_min, coverage_max]:
        raise SystemExit(f"basket window requires promotion weeks {required}, but raw "
                         f"causal coverage is {[coverage_min, coverage_max]}")

    c["item"] = c.PRODUCT_ID.map(iid)
    c["store"] = c.STORE_ID.map(smap)
    c = c.dropna(subset=["item", "store"])
    c["item"] = c.item.astype(np.int64)
    c["store"] = c.store.astype(np.int64)
    c = c[(c.WEEK_NO >= 0) & (c.WEEK_NO < 128)]

    # binary, because the file only ever records promoted cells
    c["disp"] = (c.display.fillna("0") != "0").astype(np.int8)
    c["mail"] = (c.mailer.fillna("0") != "0").astype(np.int8)
    both = int(((c.disp == 0) & (c.mail == 0)).sum())
    log(f"mapped to the model's index: {len(c):,} rows; "
        f"{both} of them record no promotion at all "
        f"({'consistent with the documented layout' if both == 0 else 'UNEXPECTED'})")

    key = (c.item.to_numpy() * n_stores + c.store.to_numpy()) * 128 + c.WEEK_NO.to_numpy()
    # one product can appear twice for a store-week; take the max so any promotion counts
    g = pd.DataFrame({"key": key, "disp": c.disp.to_numpy(), "mail": c.mail.to_numpy()}) \
        .groupby("key", as_index=False).max()
    o = np.argsort(g.key.to_numpy())
    np.savez_compressed(os.path.join(IN, "promo.npz"),
                        keys=g.key.to_numpy()[o].astype(np.int64),
                        disp=g.disp.to_numpy()[o].astype(np.int8),
                        mail=g.mail.to_numpy()[o].astype(np.int8),
                        coverage_min_week=np.int16(coverage_min),
                        coverage_max_week=np.int16(coverage_max))
    log(f"wrote basket_input/promo.npz: {len(g):,} (item, store, week) cells")
    log(f"  on display {100 * g.disp.mean():.1f}%, on mailer {100 * g.mail.mean():.1f}% "
        f"of recorded cells")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--chunk", type=int, default=4_000_000)
    p.add_argument("--indir", default=None,
                   help="basket_input directory (default: repository basket_input)")
    main(p.parse_args())
