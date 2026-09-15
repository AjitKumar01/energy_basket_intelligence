#!/usr/bin/env python3
"""Audit whether Type B/C coupon campaigns support model-ready offer exposures.

This is deliberately an EDA, not an effect estimator.  It distinguishes information
known before purchase (recipient, campaign window, coupon/product applicability) from
facts revealed only after purchase (redemption and paid price).  No basket-model
parameters are loaded or fitted.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from checkpoint_io import ROOT
from provenance import file_sha256, strict_json_dumps


SUPPORTED_TYPES = ("TypeB", "TypeC")


def safe_ratio(numerator: float | int, denominator: float | int) -> float | None:
    return None if denominator == 0 else float(numerator / denominator)


def finite_quantiles(values, probabilities=(0.0, 0.1, 0.5, 0.9, 1.0)) -> dict:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {}
    return {str(q): float(np.quantile(array, q)) for q in probabilities}


def prepare_coupon_map(coupon: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    columns = ["CAMPAIGN", "COUPON_UPC", "PRODUCT_ID"]
    exact_duplicates = int(coupon.duplicated(columns).sum())
    unique = coupon[columns].drop_duplicates().copy()
    reused = unique.groupby("COUPON_UPC").CAMPAIGN.nunique()
    return unique, {
        "raw_rows": int(len(coupon)),
        "unique_campaign_coupon_product_rows": int(len(unique)),
        "exact_duplicate_rows_removed": exact_duplicates,
        "coupon_codes_reused_across_campaigns": int((reused > 1).sum()),
        "coupon_identity_rule": "CAMPAIGN plus COUPON_UPC; COUPON_UPC alone is not unique",
    }


def cohort_window_metrics(baskets: pd.DataFrame, transaction: pd.DataFrame,
                          households: np.ndarray, products: np.ndarray,
                          first_day: int, last_day: int) -> dict:
    household_set = set(int(value) for value in households)
    window_baskets = baskets[
        (baskets.DAY >= first_day) & (baskets.DAY <= last_day)
        & baskets.household_key.isin(household_set)]
    total = window_baskets.groupby("household_key").BASKET_ID.nunique()
    window_lines = transaction[
        (transaction.DAY >= first_day) & (transaction.DAY <= last_day)
        & transaction.household_key.isin(household_set)
        & transaction.PRODUCT_ID.isin(products)]
    eligible = window_lines.groupby("household_key").BASKET_ID.nunique()
    total = total.reindex(households, fill_value=0).astype(np.int64)
    eligible = eligible.reindex(households, fill_value=0).astype(np.int64)
    active = total > 0
    rates = (eligible[active] / total[active]).to_numpy(np.float64)
    return {
        "households": int(len(households)),
        "active_households": int(active.sum()),
        "baskets": int(total.sum()),
        "baskets_containing_eligible_product": int(eligible.sum()),
        "basket_incidence": safe_ratio(int(eligible.sum()), int(total.sum())),
        "active_household_mean_basket_incidence":
            float(rates.mean()) if len(rates) else None,
        "active_household_incidence_sd":
            float(rates.std(ddof=1)) if len(rates) > 1 else None,
    }


def standardized_difference(first: dict, second: dict) -> float | None:
    mean_first = first["active_household_mean_basket_incidence"]
    mean_second = second["active_household_mean_basket_incidence"]
    sd_first = first["active_household_incidence_sd"]
    sd_second = second["active_household_incidence_sd"]
    if any(value is None for value in (mean_first, mean_second, sd_first, sd_second)):
        return None
    pooled = math.sqrt((sd_first ** 2 + sd_second ** 2) / 2.0)
    return None if pooled == 0 else float((mean_first - mean_second) / pooled)


def campaign_transaction_audit(assignments: pd.DataFrame, descriptions: pd.DataFrame,
                               coupon_map: pd.DataFrame, transaction: pd.DataFrame,
                               modeled_products: set[int]) -> list[dict]:
    baskets = transaction[["household_key", "BASKET_ID", "DAY"]].drop_duplicates()
    all_households = np.sort(transaction.household_key.unique())
    minimum_day, maximum_day = int(transaction.DAY.min()), int(transaction.DAY.max())
    rows = []
    for description in descriptions.sort_values("START_DAY").itertuples(index=False):
        campaign = int(description.CAMPAIGN)
        recipients = np.sort(assignments.loc[
            assignments.CAMPAIGN == campaign, "household_key"].unique())
        controls = np.setdiff1d(all_households, recipients, assume_unique=True)
        mapping = coupon_map[coupon_map.CAMPAIGN == campaign]
        products = np.sort(mapping.PRODUCT_ID.unique())
        duration = int(description.END_DAY - description.START_DAY + 1)
        periods = {
            "pre": (int(description.START_DAY - duration),
                    int(description.START_DAY - 1)),
            "during": (int(description.START_DAY), int(description.END_DAY)),
            "post": (int(description.END_DAY + 1), int(description.END_DAY + duration)),
        }
        metrics = {}
        for period, (first, last) in periods.items():
            observed_first, observed_last = max(first, minimum_day), min(last, maximum_day)
            if observed_first > observed_last:
                recipient_metric = cohort_window_metrics(
                    baskets, transaction, recipients, products, 1, 0)
                control_metric = cohort_window_metrics(
                    baskets, transaction, controls, products, 1, 0)
            else:
                recipient_metric = cohort_window_metrics(
                    baskets, transaction, recipients, products,
                    observed_first, observed_last)
                control_metric = cohort_window_metrics(
                    baskets, transaction, controls, products,
                    observed_first, observed_last)
            metrics[period] = {
                "requested_day_range": [first, last],
                "observed_day_range": (
                    [observed_first, observed_last]
                    if observed_first <= observed_last else None),
                "complete_window": first >= minimum_day and last <= maximum_day,
                "recipient": recipient_metric,
                "nonrecipient": control_metric,
            }
        pre_recipient = metrics["pre"]["recipient"]
        pre_control = metrics["pre"]["nonrecipient"]
        during_recipient = metrics["during"]["recipient"]
        during_control = metrics["during"]["nonrecipient"]
        pre_gap = standardized_difference(pre_recipient, pre_control)
        rates = [pre_recipient["basket_incidence"], pre_control["basket_incidence"],
                 during_recipient["basket_incidence"], during_control["basket_incidence"]]
        difference_in_differences = None
        if all(value is not None for value in rates):
            difference_in_differences = float(
                (rates[2] - rates[0]) - (rates[3] - rates[1]))
        rows.append({
            "campaign": campaign,
            "type": str(description.DESCRIPTION),
            "start_day": int(description.START_DAY),
            "end_day": int(description.END_DAY),
            "duration_days": duration,
            "recipient_households": int(len(recipients)),
            "coupon_codes": int(mapping.COUPON_UPC.nunique()),
            "eligible_products": int(len(products)),
            "modeled_eligible_products": int(sum(int(p) in modeled_products for p in products)),
            "recipient_coupon_instances": int(len(recipients) * mapping.COUPON_UPC.nunique()),
            "recipient_product_instances": int(len(recipients) * len(products)),
            "preperiod_recipient_vs_nonrecipient_standardized_difference": pre_gap,
            "descriptive_basket_incidence_difference_in_differences":
                difference_in_differences,
            "periods": metrics,
        })
    return rows


def redemption_audit(redemption: pd.DataFrame, assignments: pd.DataFrame,
                     descriptions: pd.DataFrame, coupon_map: pd.DataFrame,
                     transaction: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    chosen = (redemption.reset_index(names="redemption_id")
              .merge(descriptions, on="CAMPAIGN", how="inner", validate="many_to_one"))
    chosen = chosen[chosen.DESCRIPTION.isin(SUPPORTED_TYPES)].copy()
    assigned = assignments[["household_key", "CAMPAIGN"]].drop_duplicates()
    chosen = chosen.merge(
        assigned.assign(recipient_recorded=True),
        on=["household_key", "CAMPAIGN"], how="left", validate="many_to_one")
    chosen["recipient_recorded"] = chosen.recipient_recorded.fillna(False).astype(bool)
    chosen["within_campaign_window"] = (
        (chosen.DAY >= chosen.START_DAY) & (chosen.DAY <= chosen.END_DAY))
    coupon_keys = coupon_map[["CAMPAIGN", "COUPON_UPC"]].drop_duplicates()
    chosen = chosen.merge(
        coupon_keys.assign(coupon_in_campaign=True),
        on=["CAMPAIGN", "COUPON_UPC"], how="left", validate="many_to_one")
    chosen["coupon_in_campaign"] = chosen.coupon_in_campaign.fillna(False).astype(bool)
    valid = chosen[
        chosen.recipient_recorded & chosen.within_campaign_window
        & chosen.coupon_in_campaign].copy()

    # Count every recorded redemption on the household-day, including Type A.  Restricting
    # this count to Type B/C would incorrectly label a line as unambiguous when another
    # campaign coupon was redeemed on the same checkout day.
    day_redemptions = (redemption.groupby(["household_key", "DAY"]).size()
                       .rename("redemptions_on_household_day"))
    valid = valid.join(day_redemptions, on=["household_key", "DAY"])
    applicable = valid[["redemption_id", "household_key", "DAY", "CAMPAIGN",
                        "COUPON_UPC", "redemptions_on_household_day"]].merge(
        coupon_map, on=["CAMPAIGN", "COUPON_UPC"], how="left",
        validate="many_to_many")
    line_columns = ["household_key", "DAY", "PRODUCT_ID", "BASKET_ID", "QUANTITY",
                    "COUPON_DISC", "COUPON_MATCH_DISC", "loyalty_price", "paid_price"]
    candidates = applicable.merge(
        transaction[line_columns], on=["household_key", "DAY", "PRODUCT_ID"],
        how="inner", validate="many_to_many")
    candidates["discounted_line"] = candidates.COUPON_DISC < 0

    any_count = candidates.groupby("redemption_id").size().rename("eligible_lines")
    any_products = (candidates.groupby("redemption_id").PRODUCT_ID.nunique()
                    .rename("eligible_products_purchased"))
    discounted = candidates[candidates.discounted_line]
    discount_count = (discounted.groupby("redemption_id").size()
                      .rename("discounted_eligible_lines"))
    discount_products = (discounted.groupby("redemption_id").PRODUCT_ID.nunique()
                         .rename("discounted_eligible_products"))
    linkage = valid.set_index("redemption_id").join(
        [any_count, any_products, discount_count, discount_products])
    count_columns = ["eligible_lines", "eligible_products_purchased",
                     "discounted_eligible_lines", "discounted_eligible_products"]
    linkage[count_columns] = linkage[count_columns].fillna(0).astype(np.int64)
    linkage["high_confidence_single_line"] = (
        (linkage.redemptions_on_household_day == 1)
        & (linkage.discounted_eligible_lines == 1))

    high_ids = linkage.index[linkage.high_confidence_single_line]
    high = discounted[discounted.redemption_id.isin(high_ids)].copy()
    high["observed_manufacturer_discount"] = (-high.COUPON_DISC).round(2)
    high["observed_match_discount"] = (-high.COUPON_MATCH_DISC).round(2)
    high["observed_total_coupon_reduction"] = (
        high.observed_manufacturer_discount + high.observed_match_discount)

    value_rows = []
    high_grouped = high.groupby(["CAMPAIGN", "COUPON_UPC"])
    for key, frame in coupon_keys.groupby(["CAMPAIGN", "COUPON_UPC"]):
        if key in high_grouped.groups:
            sample = high_grouped.get_group(key).observed_manufacturer_discount
            frequencies = sample.value_counts()
            maximum_frequency = int(frequencies.max())
            mode_amount = float(frequencies[frequencies == maximum_frequency].index.min())
            sample_count = int(len(sample))
            modal_share = maximum_frequency / sample_count
            minimum, maximum = float(sample.min()), float(sample.max())
        else:
            sample_count, modal_share = 0, None
            mode_amount = minimum = maximum = None
        value_rows.append({
            "CAMPAIGN": int(key[0]),
            "COUPON_UPC": int(key[1]),
            "high_confidence_redemptions": sample_count,
            "modal_observed_manufacturer_discount": mode_amount,
            "modal_share": modal_share,
            "minimum_observed_manufacturer_discount": minimum,
            "maximum_observed_manufacturer_discount": maximum,
            "stable_observed_amount": bool(
                sample_count >= 3 and modal_share is not None and modal_share >= .9),
        })
    values = pd.DataFrame(value_rows)

    integrity = {
        "type_bc_redemption_rows": int(len(chosen)),
        "exact_duplicate_redemption_rows": int(chosen.duplicated(
            ["household_key", "DAY", "COUPON_UPC", "CAMPAIGN"]).sum()),
        "recipient_assignment_recorded": int(chosen.recipient_recorded.sum()),
        "inside_campaign_window": int(chosen.within_campaign_window.sum()),
        "coupon_code_present_in_campaign_map": int(chosen.coupon_in_campaign.sum()),
        "passes_all_three_integrity_checks": int(len(valid)),
        "valid_with_any_applicable_product_purchased": int(
            (linkage.eligible_lines > 0).sum()),
        "valid_with_any_discounted_applicable_line": int(
            (linkage.discounted_eligible_lines > 0).sum()),
        "valid_with_exactly_one_discounted_applicable_line": int(
            (linkage.discounted_eligible_lines == 1).sum()),
        "high_confidence_single_redemption_single_discounted_line": int(
            linkage.high_confidence_single_line.sum()),
        "valid_with_no_applicable_product_purchase": int(
            (linkage.eligible_lines == 0).sum()),
        "valid_with_applicable_purchase_but_no_discounted_applicable_line": int(
            ((linkage.eligible_lines > 0)
             & (linkage.discounted_eligible_lines == 0)).sum()),
    }
    return integrity, linkage.reset_index(), values


def daily_assignment_overlap(assignments: pd.DataFrame,
                             descriptions: pd.DataFrame) -> dict:
    expanded = assignments.merge(
        descriptions[["CAMPAIGN", "START_DAY", "END_DAY"]],
        on="CAMPAIGN", validate="many_to_one")
    pieces = []
    for row in expanded.itertuples(index=False):
        pieces.append(pd.DataFrame({
            "household_key": int(row.household_key),
            "DAY": np.arange(int(row.START_DAY), int(row.END_DAY) + 1),
        }))
    daily = pd.concat(pieces, ignore_index=True)
    counts = daily.groupby(["household_key", "DAY"]).size()
    return {
        "recipient_campaign_days_counting_overlaps": int(len(daily)),
        "unique_recipient_days": int(len(counts)),
        "recipient_days_with_multiple_active_type_bc_campaigns": int((counts > 1).sum()),
        "share_unique_recipient_days_with_multiple_campaigns":
            float((counts > 1).mean()),
        "maximum_simultaneous_campaigns": int(counts.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path,
                        default=ROOT / "dunnhumby_The-Complete-Journey" /
                        "dunnhumby_The-Complete-Journey CSV")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.raw_dir.resolve()
    paths = {
        "campaign_table": raw / "campaign_table.csv",
        "campaign_desc": raw / "campaign_desc.csv",
        "coupon": raw / "coupon.csv",
        "coupon_redempt": raw / "coupon_redempt.csv",
        "transaction": ROOT / "data/tx.parquet",
        "model_items": ROOT / "basket_input/items.parquet",
        "model_baskets": ROOT / "basket_input/baskets.parquet",
        "model_meta": ROOT / "basket_input/meta.json",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing EDA inputs: {missing}")

    campaign_table = pd.read_csv(paths["campaign_table"])
    descriptions = pd.read_csv(paths["campaign_desc"])
    coupon_raw = pd.read_csv(paths["coupon"])
    redemption = pd.read_csv(paths["coupon_redempt"])
    transaction = pd.read_parquet(paths["transaction"], columns=[
        "household_key", "BASKET_ID", "DAY", "WEEK_NO", "PRODUCT_ID", "QUANTITY",
        "COUPON_DISC", "COUPON_MATCH_DISC", "loyalty_price", "paid_price"])
    model_items = pd.read_parquet(paths["model_items"], columns=["PRODUCT_ID"])
    model_baskets = pd.read_parquet(
        paths["model_baskets"], columns=["BASKET_ID", "DAY", "split"])

    merged_assignment = campaign_table.merge(
        descriptions, on="CAMPAIGN", suffixes=("_assignment", "_description"),
        how="left", validate="many_to_one")
    description_mismatch = (
        merged_assignment.DESCRIPTION_assignment
        != merged_assignment.DESCRIPTION_description)
    if merged_assignment.START_DAY.isna().any():
        raise RuntimeError("campaign assignment lacks description or date range")
    assignments = campaign_table[
        campaign_table.DESCRIPTION.isin(SUPPORTED_TYPES)].copy()
    selected_descriptions = descriptions[
        descriptions.DESCRIPTION.isin(SUPPORTED_TYPES)].copy()
    _, all_mapping_audit = prepare_coupon_map(coupon_raw)
    typebc_coupon_raw = coupon_raw[coupon_raw.CAMPAIGN.isin(
        selected_descriptions.CAMPAIGN)].copy()
    coupon_map, mapping_audit = prepare_coupon_map(typebc_coupon_raw)
    modeled_products = set(model_items.PRODUCT_ID.astype(int))
    model_basket_ids = set(model_baskets.BASKET_ID.astype(int))
    model_transaction = transaction[transaction.BASKET_ID.isin(model_basket_ids)].copy()
    modeled_households = set(model_transaction.household_key.astype(int))
    campaign_rows = campaign_transaction_audit(
        assignments, selected_descriptions, coupon_map, transaction, modeled_products)
    model_assignments = assignments[
        assignments.household_key.isin(modeled_households)].copy()
    modeled_coupon_map = coupon_map[
        coupon_map.PRODUCT_ID.isin(modeled_products)].copy()
    model_campaign_rows = campaign_transaction_audit(
        model_assignments, selected_descriptions, modeled_coupon_map,
        model_transaction, modeled_products)
    model_campaign_by_id = {row["campaign"]: row for row in model_campaign_rows}
    for row in campaign_rows:
        model_row = model_campaign_by_id[row["campaign"]]
        row["model_aligned"] = {
            "recipient_households": model_row["recipient_households"],
            "coupon_codes_with_a_modeled_product": model_row["coupon_codes"],
            "eligible_products": model_row["eligible_products"],
            "recipient_coupon_instances": model_row["recipient_coupon_instances"],
            "preperiod_recipient_vs_nonrecipient_standardized_difference":
                model_row[
                    "preperiod_recipient_vs_nonrecipient_standardized_difference"],
            "descriptive_basket_incidence_difference_in_differences":
                model_row["descriptive_basket_incidence_difference_in_differences"],
            "periods": model_row["periods"],
        }
    redemption_integrity, linkage, value_reliability = redemption_audit(
        redemption, assignments, selected_descriptions, coupon_map, transaction)

    recipient_households = set(assignments.household_key.astype(int))
    typebc_coupon_keys = coupon_map[["CAMPAIGN", "COUPON_UPC"]].drop_duplicates()
    recipients_per_campaign = assignments.groupby("CAMPAIGN").household_key.nunique()
    typebc_coupon_keys = typebc_coupon_keys.join(
        recipients_per_campaign.rename("recipients"), on="CAMPAIGN")
    offered_coupon_instances = int(typebc_coupon_keys.recipients.sum())
    stable_keys = value_reliability[value_reliability.stable_observed_amount]
    stable_weight = int(stable_keys.merge(
        recipients_per_campaign.rename("recipients"),
        left_on="CAMPAIGN", right_index=True).recipients.sum())

    modeled_coupon_keys = modeled_coupon_map[
        ["CAMPAIGN", "COUPON_UPC"]].drop_duplicates()
    model_recipients_per_campaign = (model_assignments.groupby("CAMPAIGN")
                                     .household_key.nunique())
    modeled_coupon_keys = modeled_coupon_keys.join(
        model_recipients_per_campaign.rename("recipients"), on="CAMPAIGN")
    modeled_offered_coupon_instances = int(modeled_coupon_keys.recipients.sum())
    modeled_stable_keys = stable_keys.merge(
        modeled_coupon_keys[["CAMPAIGN", "COUPON_UPC", "recipients"]],
        on=["CAMPAIGN", "COUPON_UPC"], how="inner")
    modeled_stable_weight = int(modeled_stable_keys.recipients.sum())
    model_relevant_redemptions = linkage[
        linkage.household_key.isin(modeled_households)].merge(
            modeled_coupon_keys[["CAMPAIGN", "COUPON_UPC"]],
            on=["CAMPAIGN", "COUPON_UPC"], how="inner")

    split_ranges = (model_baskets.groupby("split").DAY
                    .agg(first_day="min", last_day="max").to_dict(orient="index"))
    campaign_split_membership = []
    for row in selected_descriptions.itertuples(index=False):
        active_splits = [
            split for split, bounds in split_ranges.items()
            if int(row.START_DAY) <= int(bounds["last_day"])
            and int(row.END_DAY) >= int(bounds["first_day"])
        ]
        campaign_split_membership.append({
            "campaign": int(row.CAMPAIGN),
            "active_model_splits": active_splits,
        })
    crossing_campaigns = [
        row for row in campaign_split_membership
        if len(row["active_model_splits"]) > 1]

    pre_smd = [row["preperiod_recipient_vs_nonrecipient_standardized_difference"]
               for row in campaign_rows]
    did = [row["descriptive_basket_incidence_difference_in_differences"]
           for row in campaign_rows
           if row["periods"]["during"]["complete_window"]]
    model_pre_smd = [
        row["model_aligned"][
            "preperiod_recipient_vs_nonrecipient_standardized_difference"]
        for row in campaign_rows]
    model_did = [
        row["model_aligned"][
            "descriptive_basket_incidence_difference_in_differences"]
        for row in campaign_rows
        if row["model_aligned"]["periods"]["during"]["complete_window"]]
    campaign_redemptions = linkage.groupby("CAMPAIGN").size()
    campaign_redeemers = linkage.groupby("CAMPAIGN").household_key.nunique()
    for row in campaign_rows:
        campaign = row["campaign"]
        redemption_count = int(campaign_redemptions.get(campaign, 0))
        row["valid_redemptions"] = redemption_count
        row["redeemer_households"] = int(campaign_redeemers.get(campaign, 0))
        row["redemptions_per_offered_coupon_instance"] = safe_ratio(
            redemption_count, row["recipient_coupon_instances"])

    by_type = []
    for campaign_type in SUPPORTED_TYPES:
        type_rows = [row for row in campaign_rows if row["type"] == campaign_type]
        type_campaign_ids = [row["campaign"] for row in type_rows]
        type_mapping = coupon_map[coupon_map.CAMPAIGN.isin(type_campaign_ids)]
        type_linkage = linkage[linkage.DESCRIPTION == campaign_type]
        type_model_linkage = model_relevant_redemptions[
            model_relevant_redemptions.DESCRIPTION == campaign_type]
        during_baskets = sum(
            row["periods"]["during"]["recipient"]["baskets"] for row in type_rows)
        during_eligible = sum(
            row["periods"]["during"]["recipient"]
            ["baskets_containing_eligible_product"] for row in type_rows)
        model_during_baskets = sum(
            row["model_aligned"]["periods"]["during"]["recipient"]["baskets"]
            for row in type_rows)
        model_during_eligible = sum(
            row["model_aligned"]["periods"]["during"]["recipient"]
            ["baskets_containing_eligible_product"] for row in type_rows)
        by_type.append({
            "type": campaign_type,
            "campaigns": int(len(type_rows)),
            "recipient_campaign_assignments": int(sum(
                row["recipient_households"] for row in type_rows)),
            "distinct_recipient_households": int(assignments.loc[
                assignments.DESCRIPTION == campaign_type, "household_key"].nunique()),
            "campaign_coupon_keys": int(type_mapping[
                ["CAMPAIGN", "COUPON_UPC"]].drop_duplicates().shape[0]),
            "eligible_products": int(type_mapping.PRODUCT_ID.nunique()),
            "modeled_eligible_products": int(type_mapping[
                type_mapping.PRODUCT_ID.isin(modeled_products)].PRODUCT_ID.nunique()),
            "offered_coupon_instances": int(sum(
                row["recipient_coupon_instances"] for row in type_rows)),
            "valid_redemptions": int(len(type_linkage)),
            "high_confidence_single_line_links": int(
                type_linkage.high_confidence_single_line.sum()),
            "model_relevant_valid_redemptions": int(len(type_model_linkage)),
            "during_recipient_baskets": int(during_baskets),
            "during_recipient_baskets_with_eligible_product": int(during_eligible),
            "during_recipient_eligible_product_incidence": safe_ratio(
                during_eligible, during_baskets),
            "model_aligned_during_recipient_baskets": int(model_during_baskets),
            "model_aligned_during_recipient_baskets_with_eligible_product":
                int(model_during_eligible),
            "model_aligned_during_recipient_eligible_product_incidence": safe_ratio(
                model_during_eligible, model_during_baskets),
        })

    duplicate_assignments = int(assignments.duplicated(
        ["household_key", "CAMPAIGN"]).sum())
    result = {
        "status": "completed",
        "analysis_scope": "Type B and Type C coupon campaign EDA; no model fit",
        "implementation_sha256": file_sha256(Path(__file__).resolve()),
        "input_sha256": {name: file_sha256(path) for name, path in paths.items()},
        "identity_and_integrity": {
            "type_bc_campaigns": int(len(selected_descriptions)),
            "type_b_campaigns": int((selected_descriptions.DESCRIPTION == "TypeB").sum()),
            "type_c_campaigns": int((selected_descriptions.DESCRIPTION == "TypeC").sum()),
            "recipient_campaign_assignments": int(len(assignments)),
            "recipient_households": int(len(recipient_households)),
            "duplicate_recipient_campaign_assignments": duplicate_assignments,
            "assignment_description_mismatches": int(description_mismatch.sum()),
            "all_campaign_coupon_map": all_mapping_audit,
            "type_bc_coupon_map": mapping_audit,
        },
        "availability": {
            "interpretation": (
                "The data guide states that every Type B/C campaign participant received "
                "all coupons in that campaign. Recipient, validity window, and eligible "
                "products are therefore reconstructable after exact-map deduplication."),
            "coupon_codes": int(typebc_coupon_keys.COUPON_UPC.nunique()),
            "campaign_coupon_keys": int(len(typebc_coupon_keys)),
            "eligible_products": int(coupon_map.PRODUCT_ID.nunique()),
            "modeled_eligible_products": int(modeled_coupon_map.PRODUCT_ID.nunique()),
            "modeled_catalogue_products": int(len(modeled_products)),
            "modeled_catalogue_coupon_coverage": safe_ratio(
                coupon_map[coupon_map.PRODUCT_ID.isin(modeled_products)].PRODUCT_ID.nunique(),
                len(modeled_products)),
            "offered_household_campaign_coupon_instances": offered_coupon_instances,
            "model_relevant_offered_household_campaign_coupon_instances":
                modeled_offered_coupon_instances,
            "recipient_households_in_basket_model": int(
                len(recipient_households & modeled_households)),
            "recipient_households_outside_basket_model": int(
                len(recipient_households - modeled_households)),
            **daily_assignment_overlap(assignments, selected_descriptions),
        },
        "redemption_linkage": {
            **redemption_integrity,
            "valid_redemptions_per_offered_coupon_instance": safe_ratio(
                redemption_integrity["passes_all_three_integrity_checks"],
                offered_coupon_instances),
            "valid_redemptions_for_model_relevant_coupon_keys_and_households":
                int(len(model_relevant_redemptions)),
            "model_relevant_high_confidence_single_line_links": int(
                model_relevant_redemptions.high_confidence_single_line.sum()),
        },
        "discount_value_reliability": {
            "face_value_present_in_coupon_table": False,
            "campaign_coupon_keys": int(len(value_reliability)),
            "keys_with_any_high_confidence_observed_discount": int(
                (value_reliability.high_confidence_redemptions > 0).sum()),
            "keys_with_at_least_three_high_confidence_observations": int(
                (value_reliability.high_confidence_redemptions >= 3).sum()),
            "keys_meeting_stable_observed_amount_rule": int(
                value_reliability.stable_observed_amount.sum()),
            "stable_rule": (
                "at least three single-redemption/single-discounted-line observations "
                "and at least 90% share at the modal cent amount"),
            "offered_coupon_instances_covered_by_stable_observed_amount": stable_weight,
            "share_offered_coupon_instances_covered_by_stable_observed_amount":
                safe_ratio(stable_weight, offered_coupon_instances),
            "model_relevant_coupon_keys": int(len(modeled_coupon_keys)),
            "model_relevant_coupon_keys_meeting_stable_observed_amount_rule": int(
                len(modeled_stable_keys)),
            "model_relevant_offered_coupon_instances_covered_by_stable_observed_amount":
                modeled_stable_weight,
            "share_model_relevant_offered_coupon_instances_covered_by_stable_amount":
                safe_ratio(modeled_stable_weight, modeled_offered_coupon_instances),
            "high_confidence_observed_manufacturer_discount_quantiles": finite_quantiles(
                value_reliability.loc[
                    value_reliability.high_confidence_redemptions > 0,
                    "modal_observed_manufacturer_discount"]),
            "warning": (
                "An observed transaction discount is not verified coupon face value. "
                "Quantity rules, other coupons, and ambiguous line assignment remain."),
        },
        "selection_and_temporal_evidence": {
            "recipient_vs_nonrecipient_preperiod_standardized_difference_quantiles":
                finite_quantiles(pre_smd),
            "campaigns_with_absolute_preperiod_standardized_difference_at_least_0_1": int(
                sum(value is not None and abs(value) >= .1 for value in pre_smd)),
            "campaigns_with_absolute_preperiod_standardized_difference_at_least_0_25": int(
                sum(value is not None and abs(value) >= .25 for value in pre_smd)),
            "complete_during_campaign_windows": int(sum(
                row["periods"]["during"]["complete_window"] for row in campaign_rows)),
            "descriptive_difference_in_differences_quantiles_complete_windows":
                finite_quantiles(did),
            "model_aligned_recipient_vs_nonrecipient_preperiod_smd_quantiles":
                finite_quantiles(model_pre_smd),
            "model_aligned_campaigns_with_absolute_preperiod_smd_at_least_0_1": int(
                sum(value is not None and abs(value) >= .1 for value in model_pre_smd)),
            "model_aligned_campaigns_with_absolute_preperiod_smd_at_least_0_25": int(
                sum(value is not None and abs(value) >= .25 for value in model_pre_smd)),
            "model_aligned_complete_during_campaign_windows": int(sum(
                row["model_aligned"]["periods"]["during"]["complete_window"]
                for row in campaign_rows)),
            "model_aligned_descriptive_did_quantiles_complete_windows":
                finite_quantiles(model_did),
            "model_split_day_ranges": {
                split: {key: int(value) for key, value in bounds.items()}
                for split, bounds in split_ranges.items()
            },
            "campaigns_crossing_model_split_boundaries": int(len(crossing_campaigns)),
            "crossing_campaign_details": crossing_campaigns,
            "split_rule": (
                "Coupon evaluation must split by whole campaign; a naive basket-time "
                "split places the same offer campaign on both sides of a boundary."),
            "causal_assignment_documented": False,
            "interpretation": (
                "Preperiod imbalance and absence of a randomized assignment mechanism "
                "prevent causal coupon-effect claims. Difference-in-differences values "
                "are descriptive diagnostics only."),
        },
        "reliability_decision": {
            "binary_offer_availability_for_type_bc": "usable_with_documented_limits",
            "coupon_product_applicability": "usable_after_exact_duplicate_removal",
            "coupon_monetary_value": "not_reliable_catalogue_wide",
            "realized_paid_price_as_choice_regressor": "invalid_outcome_selected",
            "causal_coupon_counterfactual": "not_identified",
            "heldout_binary_offer_prediction_research": "feasible",
            "model_usage_authorized_by_this_eda": False,
        },
        "by_campaign_type": by_type,
        "per_campaign": campaign_rows,
        "discount_value_key_audit": value_reliability.to_dict(orient="records"),
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(strict_json_dumps(result))
    print("[coupon-eda] Type B/C campaigns="
          f"{result['identity_and_integrity']['type_bc_campaigns']} "
          f"assignments={len(assignments)} recipients={len(recipient_households)}",
          flush=True)
    print("[coupon-eda] valid redemptions="
          f"{redemption_integrity['passes_all_three_integrity_checks']} "
          "high-confidence line links="
          f"{redemption_integrity['high_confidence_single_redemption_single_discounted_line']}",
          flush=True)
    print("[coupon-eda] stable observed-amount keys="
          f"{result['discount_value_reliability']['keys_meeting_stable_observed_amount_rule']}"
          f"/{len(value_reliability)}; model usage authorized=False", flush=True)
    print(f"[coupon-eda] report={output}", flush=True)


if __name__ == "__main__":
    main()
