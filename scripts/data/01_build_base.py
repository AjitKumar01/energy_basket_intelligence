"""
Stage 1 -- Build the cleaned transaction base and the price panel.

Mirrors the data construction in Donnelly, Ruiz, Blei & Athey (2023),
"Counterfactual Inference for Consumer Choice Across Many Product Categories",
adapted to dunnhumby "The Complete Journey".

Paper setting                          dunnhumby analogue
-------------------------------------  --------------------------------------
1 store, 23 months, loyalty card       561 stores, 102 weeks, loyalty panel
UPC                                    PRODUCT_ID
"category" (unit of substitution)      COMMODITY_DESC
"class"/"subclass" (held out)          SUB_COMMODITY_DESC / BRAND / MANUFACTURER
trip = household x calendar day        checkout basket = BASKET_ID; a household-day can
                                       contain several checkouts
session = day (prices constant)        session = day (chain) or (store, day)
price = daily median transacted price  weekly *modal* transacted unit price

transaction_data carries no price column, only sales_value and three discount
columns, so every price here is a reconstruction.  See the block comment in main()
and 10_price_definition_audit.py for the evidence behind each one; briefly,
base_price is the regular posted shelf price, loyalty_price is what a card holder
faces, paid_price is what the shopper actually handed over, and unit_price is
whichever of the first two the model consumes (NF_PRICE_BASIS, default "loyalty").

Outputs (parquet, in ../data):
    tx.parquet          cleaned lines, with base / loyalty / paid prices
    trips.parquet       one row per (household, day) trip; spend and spend_paid
    price_week.parquet  product x week chain panel: price, base_price,
                        promo_depth, n_tx, n_store
    price_store_week.parquet  product x store x week panel: price, base_price, n_tx
"""
import os
import numpy as np
import pandas as pd

# Raw dunnhumby CSVs.  Defaults to a sibling of the repository; override with
# NF_RAW_DIR if the download lives somewhere else.
RAW = os.path.join(os.environ.get(
    "NF_RAW_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
                 "dunnhumby_The-Complete-Journey",
                 "dunnhumby_The-Complete-Journey CSV")), "")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")
os.makedirs(OUT, exist_ok=True)


def log(msg):
    print(f"[01] {msg}", flush=True)


def modal_price(tx, keys, col, order):
    """Most frequent cent value of `col` within each `keys` cell, aligned to `order`.

    Vectorised: a lambda over 2.5M rows takes minutes.  Ties are broken toward the
    lower price -- on a tie the shelf offered both, and the lower one is the one a
    shopper could actually obtain -- which also makes the result deterministic.
    """
    c = tx[keys].copy()
    c["cents"] = np.round(tx[col].to_numpy() * 100).astype(np.int64)
    g = c.groupby(keys + ["cents"], sort=False).size().rename("n").reset_index()
    g = g.sort_values(["n", "cents"], ascending=[True, False], kind="mergesort")
    g = g.drop_duplicates(keys, keep="last")
    return (g.set_index(keys).cents / 100.0).reindex(
        pd.MultiIndex.from_frame(order[keys])).to_numpy()


def main():
    log("reading transaction_data.csv ...")
    tx = pd.read_csv(
        RAW + "transaction_data.csv",
        dtype={
            "household_key": np.int32, "BASKET_ID": np.int64, "DAY": np.int16,
            "PRODUCT_ID": np.int32, "QUANTITY": np.int32, "STORE_ID": np.int32,
            "TRANS_TIME": np.int16, "WEEK_NO": np.int16,
        },
    )
    n0 = len(tx)

    # ------------------------------------------------------------------ prices
    # transaction_data has no price column, so every price here is a construction
    # and each one answers a different question.  The user guide (p.3) states that
    # sales_value is the money the retailer receives, "taking the coupon match and
    # loyalty card discount into account", and that it is explicitly *not* what the
    # customer paid: a manufacturer coupon is reimbursed to the retailer, so the
    # shopper handed over sales_value + coupon_disc.  All three discount columns are
    # stored as negative numbers, so subtracting one adds it back.
    #
    # The guide's two named formulas have their labels swapped relative to its own
    # worked examples; the examples are authoritative.  Line 2 (q=2, sales_value
    # $2.00, retail_disc -$1.34) is called out as a regular shelf price of $1.67 and
    # a with-card price of $1.00, and line 3 as an outlay of $2.34 on sales_value of
    # $2.89 with a -$0.55 coupon.  The three definitions below reproduce all of it.
    #
    #   base_price     regular posted shelf price, before any discount -- the cost
    #                  borne by the retailer.  Exactly uniform across shoppers within
    #                  a product x store x week (99.1% of cells with >=2 buyers), so
    #                  this, not loyalty_price, is *the* posted price.
    #   loyalty_price  what a card holder faces at the shelf: base net of the loyalty
    #                  promotion, but before that shopper's own coupons.  Uniform in
    #                  only 92.9% of the same cells -- retail_disc carries a
    #                  household-idiosyncratic component (4.4% of cells have mixed
    #                  discount status among shoppers of the same item, store, week).
    #   paid_price     what the shopper actually handed over.  Household specific by
    #                  construction, and <= 0 on 21% of coupon lines, so it is a
    #                  line-level quantity only and never becomes a session price.
    tx = tx[(tx.QUANTITY > 0) & (tx.SALES_VALUE > 0)]
    tx["base_price"] = (tx.SALES_VALUE - tx.RETAIL_DISC - tx.COUPON_MATCH_DISC) / tx.QUANTITY
    tx["loyalty_price"] = (tx.SALES_VALUE - tx.COUPON_MATCH_DISC) / tx.QUANTITY
    tx["paid_price"] = (tx.SALES_VALUE + tx.COUPON_DISC) / tx.QUANTITY
    tx["paid_value"] = tx.SALES_VALUE + tx.COUPON_DISC
    tx["coupon_used"] = (tx.COUPON_DISC < 0).astype(np.int8)

    # The price the model sees.  Every household in this panel holds a loyalty card,
    # so the price faced at the shelf is the loyalty price; base_price is what a
    # non-cardholder would pay and is faced by nobody in the sample.  Selecting it
    # instead drops two thirds of the identifying variation (chain week-to-week price
    # change rate 23.9% -> 7.5%, within-item CV 0.125 -> 0.050), so the default is
    # "loyalty" and NF_PRICE_BASIS=base is the sensitivity check.
    basis = os.environ.get("NF_PRICE_BASIS", "loyalty")
    if basis not in ("loyalty", "base"):
        raise SystemExit(f"NF_PRICE_BASIS must be 'loyalty' or 'base', got {basis!r}")
    tx["unit_price"] = tx.loyalty_price if basis == "loyalty" else tx.base_price
    log(f"price basis: {basis}  (unit_price = {basis}_price)")

    # Drop implausible lines: random-weight / bulk lines and price outliers.  The
    # sign-anomalous lines matter out of proportion to their count: a positive
    # retail_disc inverts the discount.  Both real instances (sales_value $7.98 with
    # retail_disc +$3.99, and $0.51 with +$0.26) yield a base price *half* the amount
    # paid, which then enters the panel as a 50% price cut that never happened.  Four
    # further lines are float noise at ~1e-16, hence the threshold rather than > 0.
    anom = (tx.RETAIL_DISC > 1e-9) | (tx.COUPON_DISC > 1e-9) | (tx.COUPON_MATCH_DISC > 1e-9)
    bad = ((tx.QUANTITY > 30) | (tx.unit_price > 100) | (tx.unit_price < 0.05) | anom)
    log(f"dropping {bad.sum():,} of {n0:,} lines (bulk quantity, extreme unit price, "
        f"or sign-anomalous discount [{int(anom.sum())} lines])")
    tx = tx[~bad].copy()

    # ------------------------------------------------------- calendar handling
    # WEEK_NO is given.  Week 1 is short (days 1-5), week 102 is short (706-711).
    # Establish the weekday index from the basket-count profile: the two busiest
    # residues of DAY % 7 are the weekend.
    bask_by_res = tx.groupby(tx.DAY % 7).BASKET_ID.nunique()
    weekend = set(bask_by_res.sort_values(ascending=False).index[:2])
    log(f"busiest DAY%7 residues (assumed Sat/Sun): {sorted(weekend)}")
    # Anchor: call the later of the two weekend residues 'Sunday' (=0) so that
    # weekday ids run 0..6 = Sun..Sat.  Only the partition matters for the model.
    sunday_res = max(weekend)
    tx["weekday"] = ((tx.DAY % 7) - sunday_res) % 7
    tx["weekday"] = tx.weekday.astype(np.int8)
    tx["hour"] = (tx.TRANS_TIME // 100 + (tx.TRANS_TIME % 100) / 60.0).astype(np.float32)

    # ------------------------------------------------------------------- trips
    # This legacy summary is a household-day (the paper's definition), not the basket
    # model's observational unit.  Stage 22 correctly uses BASKET_ID: 18,469 retained
    # household-days contain multiple checkout baskets, so the two must not be conflated.
    trips = (
        tx.groupby(["household_key", "DAY"])
        .agg(
            n_lines=("PRODUCT_ID", "size"),
            n_items=("PRODUCT_ID", "nunique"),
            # spend is retailer receipts (the guide's sales_value).  spend_paid is
            # what the household actually handed over: the two differ by the
            # manufacturer coupons the retailer is reimbursed for -- 0.43% of
            # turnover overall, but up to 8.6% for an individual household, so any
            # household-level spend measure should use spend_paid.
            spend=("SALES_VALUE", "sum"),
            spend_paid=("paid_value", "sum"),
            store_id=("STORE_ID", lambda s: s.mode().iat[0]),
            n_stores=("STORE_ID", "nunique"),
            week=("WEEK_NO", "first"),
            weekday=("weekday", "first"),
            hour=("hour", "median"),
        )
        .reset_index()
    )
    log(f"trips: {len(trips):,}   households: {trips.household_key.nunique():,}   "
        f"multi-store trips: {(trips.n_stores > 1).mean():.4f}")

    # ------------------------------------------------------------ price panels
    # A posted price is a single number, not a central tendency: within a
    # product x store x week the base price takes one value in 99.1% of cells.  The
    # right estimator is therefore the mode, which recovers that value exactly, and
    # not the median, which is only near it once the household-idiosyncratic part of
    # retail_disc is mixed in (the two differ by more than a cent on 2.8% of
    # chain-week cells, by up to $4.49).
    pw = (
        tx.groupby(["PRODUCT_ID", "WEEK_NO"])
        .agg(n_tx=("unit_price", "size"), n_store=("STORE_ID", "nunique"))
        .reset_index()
    )
    pw["price"] = modal_price(tx, ["PRODUCT_ID", "WEEK_NO"], "unit_price", pw)
    pw["base_price"] = modal_price(tx, ["PRODUCT_ID", "WEEK_NO"], "base_price", pw)
    # Depth of the loyalty promotion, 0 when the item is at its regular price.
    pw["promo_depth"] = (1.0 - pw.price / pw.base_price).clip(lower=0.0)

    # Store-level product x store x week price.
    psw = (
        tx.groupby(["PRODUCT_ID", "STORE_ID", "WEEK_NO"])
        .agg(n_tx=("unit_price", "size"))
        .reset_index()
    )
    psw["price"] = modal_price(tx, ["PRODUCT_ID", "STORE_ID", "WEEK_NO"], "unit_price", psw)
    psw["base_price"] = modal_price(tx, ["PRODUCT_ID", "STORE_ID", "WEEK_NO"], "base_price", psw)

    log(f"chain price cells: {len(pw):,}   store price cells: {len(psw):,}")

    tx.to_parquet(os.path.join(OUT, "tx.parquet"), index=False)
    trips.to_parquet(os.path.join(OUT, "trips.parquet"), index=False)
    pw.to_parquet(os.path.join(OUT, "price_week.parquet"), index=False)
    psw.to_parquet(os.path.join(OUT, "price_store_week.parquet"), index=False)
    log("wrote tx / trips / price_week / price_store_week parquet")


if __name__ == "__main__":
    main()
