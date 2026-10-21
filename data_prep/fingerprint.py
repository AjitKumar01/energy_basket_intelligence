#!/usr/bin/env python3
"""Build a bundle's ragged assortment index (if missing) and write its data fingerprint.

    fingerprint.py --data-root BUNDLE_ROOT
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from v4.bundle import data_root, load_bundle, write_fingerprint  # noqa: E402

p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
p.add_argument("--data-root", type=Path, help="bundle root (default $ENERGY_MODEL_DATA_ROOT)")
p.add_argument("--core-only", action="store_true",
               help="leave optional panels out of the fingerprint (Dunnhumby, whose fingerprint predates them)")
a = p.parse_args()
root = data_root(a.data_root)
load_bundle(root)
print(f"[fingerprint] {write_fingerprint(root, core_only=a.core_only)}", flush=True)
