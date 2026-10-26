"""Paired comparison of two .npz files written by law_checks.py (--check logz or ll) from two
code versions on the same trips:

    python bench/compare.py OLD.npz NEW.npz
Prints, per array, the per-trip difference NEW - OLD: min, max, mean and its standard error.
"""
import json
import sys

import numpy as np

old, new = np.load(sys.argv[1]), np.load(sys.argv[2])
assert np.array_equal(old["trips"], new["trips"]), "the two files are on different trips"
for k in old.files:
    if k == "trips" or k not in new.files:
        continue
    d = new[k] - old[k]
    print(json.dumps({"array": k, "trips": int(d.size), "min": float(d.min()), "max": float(d.max()),
                      "mean": float(d.mean()), "se": float(d.std(ddof=1) / np.sqrt(d.size))}))
