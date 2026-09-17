"""Process-level defaults for running the retail API (no model imports here)."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = ROOT / "data/erim_basket/model_input_availability_category"


def apply_default_data_root() -> Path:
    """Select the served model-data bundle before model modules are imported.

    ``data`` and ``features`` read ``ENERGY_MODEL_DATA_ROOT`` at import time, so entry
    points must call this first.  ``RETAIL_API_DATA_ROOT`` or an explicit
    ``ENERGY_MODEL_DATA_ROOT`` selects another bundle; checkpoint loading rejects a
    checkpoint whose data fingerprint does not match the selected bundle.
    """
    configured = os.environ.get("RETAIL_API_DATA_ROOT", str(DEFAULT_DATA_ROOT))
    os.environ.setdefault(
        "ENERGY_MODEL_DATA_ROOT", str(Path(configured).expanduser().resolve()))
    return Path(os.environ["ENERGY_MODEL_DATA_ROOT"])
