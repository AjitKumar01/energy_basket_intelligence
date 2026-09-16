"""Strict readers for the external price-choice verification datasets."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from external_choice import ChoicePanel, validate_panel


ERIM_PURCHASE_WIDTHS = [2, 2, 6, 1, 1, 8, 13, 4, 1, 5, 7, 7, 3, 3, 5,
                        3, 5, 11, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3]
ERIM_PURCHASE_COLUMNS = [
    "cell", "market", "week", "dow", "trip", "household", "upc", "store",
    "store_type", "units_raw", "expenditure_raw", "weight_raw", "special_coupon_code",
    "store_coupon_units", "store_coupon_value_raw", "manufacturer_coupon_units",
    "manufacturer_coupon_value_raw", "equivalence_raw", "line_number", "end_display",
    "front_display", "aisle_display", "other_display", "advertised", "ad_type",
    "in_ad_coupon", "point_of_purchase", "special_price", "coupon_factor", "filler",
]
ERIM_SHOPPING_WIDTHS = [2, 4, 1, 8, 6, 7]
ERIM_SHOPPING_COLUMNS = ["market", "store", "store_type", "household", "week", "spend_raw"]
ERIM_RETAIL_WIDTHS = [2, 6, 4, 13, 5, 9, 7, 7, 5, 11, 4, 1, 1, 1, 1,
                      1, 1, 1, 1, 1, 2]
ERIM_RETAIL_COLUMNS = [
    "market", "week", "store", "upc", "units", "revenue_raw", "weight_raw",
    "store_coupon_value_raw", "store_coupon_units", "equivalence_raw", "line_number",
    "end_display", "front_display", "aisle_display", "other_display", "advertised",
    "ad_type", "in_ad_coupon", "point_of_purchase", "special_price", "filler",
]
ERIM_PRODUCT_WIDTHS = [13, 30, 2, 6, 7, 3, 11, 4]
ERIM_PRODUCT_COLUMNS = [
    "upc", "description", "weight_code", "weight_description", "weight_amount",
    "multipack", "equivalence_factor", "line_number",
]


@dataclass(frozen=True)
class ERIMChoiceData:
    panel: ChoicePanel
    week: np.ndarray
    product_labels: list[str]
    audit: dict


def _read_fixed(path: Path, widths: list[int], columns: list[str],
                numeric_columns: list[str]) -> pd.DataFrame:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    expected = sum(widths)
    with path.open("rb") as stream:
        first = stream.readline().rstrip(b"\r\n")
    if len(first) != expected:
        raise ValueError(f"{path.name} has record length {len(first)}, expected {expected}")
    frame = pd.read_fwf(path, widths=widths, names=columns, dtype=str,
                        keep_default_na=False)
    for column in columns:
        frame[column] = frame[column].str.strip()
    for column in numeric_columns:
        converted = pd.to_numeric(frame[column], errors="coerce")
        if converted.isna().any():
            example = frame.loc[converted.isna(), column].iloc[0]
            raise ValueError(f"invalid numeric field {column}={example!r} in {path.name}")
        frame[column] = converted.astype(np.int64)
    return frame


def read_erim_purchase(path: Path) -> pd.DataFrame:
    numeric = ERIM_PURCHASE_COLUMNS[:19]
    frame = _read_fixed(path, ERIM_PURCHASE_WIDTHS, ERIM_PURCHASE_COLUMNS, numeric)
    frame["units"] = frame.units_raw / 100.0
    frame["expenditure"] = frame.expenditure_raw / 100.0
    frame["unit_price"] = frame.expenditure / frame.units
    return frame


def read_erim_shopping(path: Path) -> pd.DataFrame:
    frame = _read_fixed(path, ERIM_SHOPPING_WIDTHS, ERIM_SHOPPING_COLUMNS,
                        ERIM_SHOPPING_COLUMNS)
    frame["spend"] = frame.spend_raw / 100.0
    return frame


def read_erim_retail(path: Path, *, allowed_weeks: set[int] | None = None) -> pd.DataFrame:
    numeric = ERIM_RETAIL_COLUMNS[:11]
    frame = _read_fixed(path, ERIM_RETAIL_WIDTHS, ERIM_RETAIL_COLUMNS, numeric)
    if allowed_weeks is not None:
        frame = frame[frame.week.isin(allowed_weeks)].copy()
        if frame.empty:
            raise ValueError("ERIM retail data have no rows in the requested weeks")
    if bool((frame.units <= 0).any()):
        raise ValueError("ERIM retail prices require positive aggregate units")
    frame["revenue"] = frame.revenue_raw / 100.0
    frame["average_price"] = frame.revenue / frame.units
    if bool((frame.average_price <= 0).any()):
        raise ValueError("ERIM retail average prices must be positive")
    return frame


def read_erim_products(path: Path) -> dict[int, str]:
    """Read the official ERIM UPC stub and return human-readable labels."""
    frame = _read_fixed(path, ERIM_PRODUCT_WIDTHS, ERIM_PRODUCT_COLUMNS,
                        ["upc", "weight_amount", "multipack", "equivalence_factor",
                         "line_number"])
    if frame.upc.duplicated().any():
        raise ValueError("ERIM product UPCs are not unique")
    labels = {}
    for row in frame.itertuples(index=False):
        description = " ".join(str(row.description).split())
        size = " ".join(str(row.weight_description).split())
        label = description if not size else f"{description} ({size})"
        labels[int(row.upc)] = label
    return labels


def read_bayesm_csv(choice_path: Path, demographics_path: Path) -> tuple[ChoicePanel, list[str]]:
    choice = pd.read_csv(choice_path)
    demographics = pd.read_csv(demographics_path)
    if choice.columns[:2].tolist() != ["hhid", "choice"]:
        raise ValueError("bayesm choice data must begin with hhid and choice")
    price_columns = choice.columns[2:].tolist()
    if len(price_columns) < 2 or choice[price_columns].isna().any().any():
        raise ValueError("bayesm choice prices are incomplete")
    if bool((choice[price_columns] <= 0).any().any()):
        raise ValueError("bayesm choice prices must be positive")
    if demographics.hhid.duplicated().any():
        raise ValueError("bayesm household demographics are not unique")
    merged = choice.merge(demographics, on="hhid", how="left", validate="many_to_one")
    covariate_columns = [column for column in demographics.columns if column != "hhid"]
    if merged[covariate_columns].isna().any().any():
        raise ValueError("a bayesm choice has no household demographics")
    n, alternatives = len(merged), len(price_columns)
    product = np.broadcast_to(np.arange(alternatives, dtype=np.int64), (n, alternatives)).copy()
    selected = merged.choice.to_numpy(np.int64) - 1
    panel = ChoicePanel(
        product=product,
        log_price=np.log(merged[price_columns].to_numpy(np.float64)),
        mask=np.ones((n, alternatives), dtype=bool),
        chosen_slot=selected,
        group=merged.hhid.to_numpy(),
        covariates=merged[covariate_columns].to_numpy(np.float64),
    )
    validate_panel(panel)
    return panel, price_columns


def build_erim_choice_data(purchase_path: Path, shopping_path: Path,
                           retail_path: Path,
                           product_path: Path | None = None) -> ERIMChoiceData:
    """Construct price choice sets, removing the focal household from store prices."""
    purchase = read_erim_purchase(purchase_path)
    shopping = read_erim_shopping(shopping_path)
    retail = read_erim_retail(retail_path)
    keys = ["market", "household", "store", "week"]
    opportunities = (shopping.groupby(keys, as_index=False)
                     .agg(shopping_occasions=("spend", "size"), shopping_spend=("spend", "sum")))
    purchase_in_window = purchase.merge(opportunities[keys], on=keys, how="inner")
    contribution = (purchase_in_window.groupby(keys + ["upc"], as_index=False)
                    .agg(household_units=("units", "sum"),
                         household_expenditure=("expenditure", "sum")))
    outcomes = purchase_in_window[keys + ["dow", "trip", "upc"]].drop_duplicates()

    retail_key = ["market", "store", "week"]
    retail_groups = {
        key: frame.copy() for key, frame in retail.groupby(retail_key, sort=False)
    }
    contribution_groups = {
        key: frame.set_index("upc") for key, frame in contribution.groupby(keys, sort=False)
    }
    product_values = np.sort(retail.upc.unique())
    product_index = {int(upc): i for i, upc in enumerate(product_values)}
    rows: list[tuple[list[int], list[float], int, int, int]] = []
    exclusions = {
        "no_retail_choice_set": 0,
        "chosen_product_absent_from_retail": 0,
        "chosen_price_unsupported_after_household_removal": 0,
        "fewer_than_two_leaveout_alternatives": 0,
    }
    for record in outcomes.itertuples(index=False):
        retail_frame = retail_groups.get((record.market, record.store, record.week))
        if retail_frame is None:
            exclusions["no_retail_choice_set"] += 1
            continue
        upcs = retail_frame.upc.to_numpy(np.int64)
        if int(record.upc) not in set(upcs):
            exclusions["chosen_product_absent_from_retail"] += 1
            continue
        units = retail_frame.units.to_numpy(np.float64).copy()
        revenue = retail_frame.revenue.to_numpy(np.float64).copy()
        household_frame = contribution_groups.get(
            (record.market, record.household, record.store, record.week))
        if household_frame is not None:
            for slot, upc in enumerate(upcs):
                if upc in household_frame.index:
                    value = household_frame.loc[upc]
                    if isinstance(value, pd.DataFrame):
                        raise RuntimeError("household contributions were not uniquely aggregated")
                    units[slot] -= float(value.household_units)
                    revenue[slot] -= float(value.household_expenditure)
        supported = (units > 0) & (revenue > 0)
        chosen_position = np.flatnonzero(upcs == int(record.upc))
        if len(chosen_position) != 1:
            raise RuntimeError("retail choice set has duplicate product identifiers")
        if not supported[int(chosen_position[0])]:
            exclusions["chosen_price_unsupported_after_household_removal"] += 1
            continue
        if int(supported.sum()) < 2:
            exclusions["fewer_than_two_leaveout_alternatives"] += 1
            continue
        supported_upcs = upcs[supported]
        prices = revenue[supported] / units[supported]
        chosen_slot = int(np.flatnonzero(supported_upcs == int(record.upc))[0])
        rows.append((
            [product_index[int(upc)] for upc in supported_upcs],
            np.log(prices).tolist(), chosen_slot, int(record.household), int(record.week)))
    if not rows:
        raise RuntimeError("ERIM joins produced no price-supported choices")
    maximum = max(len(row[0]) for row in rows)
    n = len(rows)
    product = np.zeros((n, maximum), dtype=np.int64)
    log_price = np.zeros((n, maximum), dtype=np.float64)
    mask = np.zeros((n, maximum), dtype=bool)
    chosen = np.empty(n, dtype=np.int64)
    household = np.empty(n, dtype=np.int64)
    week = np.empty(n, dtype=np.int64)
    for i, (products, prices, selected, group, period) in enumerate(rows):
        length = len(products)
        product[i, :length] = products
        log_price[i, :length] = prices
        mask[i, :length] = True
        chosen[i], household[i], week[i] = selected, group, period
    panel = ChoicePanel(product, log_price, mask, chosen, household,
                        np.empty((n, 0), dtype=np.float64))
    validate_panel(panel, n_products=len(product_values))
    chosen_weeks = set(outcomes.week.unique())
    available_opportunities = opportunities.merge(
        retail[retail_key].drop_duplicates(), on=retail_key, how="left", indicator=True)
    audit = {
        "raw_purchase_rows": int(len(purchase)),
        "raw_shopping_rows": int(len(shopping)),
        "raw_retail_rows": int(len(retail)),
        "households": int(purchase.household.nunique()),
        "products_in_retail": int(len(product_values)),
        "shopping_household_store_weeks": int(len(opportunities)),
        "shopping_opportunities_with_retail_choice_set": int(
            available_opportunities._merge.eq("both").sum()),
        "category_purchase_outcomes_in_shopping_window": int(len(outcomes)),
        "price_supported_choice_outcomes": int(n),
        "choice_set_size": {
            "minimum": int(mask.sum(axis=1).min()),
            "median": float(np.median(mask.sum(axis=1))),
            "maximum": int(mask.sum(axis=1).max()),
        },
        "weeks_in_outcomes": int(len(chosen_weeks)),
        "exclusions": exclusions,
        "price_definition": "leave-one-household-out store-UPC-week average pre-coupon price",
        "choice_set_limitation": (
            "alternatives are UPCs with positive recorded store-week sales; "
            "the source does not prove shelf availability of unsold UPCs"),
    }
    label_map = read_erim_products(product_path) if product_path is not None else {}
    labels = [label_map.get(int(upc), str(int(upc))) for upc in product_values]
    audit["product_descriptions_matched"] = int(
        sum(int(upc) in label_map for upc in product_values))
    return ERIMChoiceData(panel, week, labels, audit)
