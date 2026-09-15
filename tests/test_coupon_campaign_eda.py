import pandas as pd

from eda_typebc_coupon_campaigns import prepare_coupon_map, safe_ratio


def test_coupon_map_deduplicates_only_within_campaign_identity():
    raw = pd.DataFrame({
        "CAMPAIGN": [1, 1, 1, 2],
        "COUPON_UPC": [10, 10, 10, 10],
        "PRODUCT_ID": [100, 100, 101, 100],
    })
    mapping, audit = prepare_coupon_map(raw)
    assert len(mapping) == 3
    assert audit["exact_duplicate_rows_removed"] == 1
    assert audit["coupon_codes_reused_across_campaigns"] == 1
    assert set(map(tuple, mapping.to_numpy())) == {
        (1, 10, 100), (1, 10, 101), (2, 10, 100)}


def test_safe_ratio_fails_closed_on_zero_denominator():
    assert safe_ratio(1, 0) is None
    assert safe_ratio(1, 4) == 0.25
