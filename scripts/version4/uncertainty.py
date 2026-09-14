"""Uncertainty of fixed-model paired scores with repeated household observations."""
from __future__ import annotations

import numpy as np


def household_cluster_se(values: np.ndarray, household: np.ndarray) -> float:
    """Cluster-robust SE for a trip-weighted mean, allowing household dependence."""
    values = np.asarray(values, dtype=np.float64)
    household = np.asarray(household)
    if values.ndim != 1 or values.shape != household.shape or not np.isfinite(values).all():
        raise ValueError("finite scores and household IDs must be matching vectors")
    unique, inverse = np.unique(household, return_inverse=True)
    if len(values) < 2 or len(unique) < 2:
        return float("inf")
    sums = np.bincount(inverse, weights=values - values.mean(), minlength=len(unique))
    return float(np.sqrt(len(unique) / (len(unique) - 1) * (sums @ sums) / len(values)**2))


def paired_score_summary(values, household=None, cluster_label="household") -> dict:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("at least two finite paired scores are required")
    naive = float(values.std(ddof=1) / np.sqrt(len(values)))
    se = naive if household is None else household_cluster_se(values, household)
    mean = float(values.mean())
    return {"trips": len(values), "mean": mean, "standard_error": se,
            "trip_naive_standard_error": naive,
            "standard_error_method": ("trip_iid" if household is None else
                                      f"{cluster_label}_cluster_robust"),
            "clusters": None if household is None else len(np.unique(household)),
            "95_interval": [mean - 1.96 * se, mean + 1.96 * se],
            "median": float(np.median(values))}
