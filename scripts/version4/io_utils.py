"""Small shared helpers: where the model-data bundle lives and how reports are written."""
from __future__ import annotations

import json
import math
import numbers
import os
from pathlib import Path
from typing import Any, Mapping


def model_data_root(default: str | Path) -> Path:
    """The model-data bundle: ``ENERGY_MODEL_DATA_ROOT`` when set, else ``default``."""
    configured = os.environ.get("ENERGY_MODEL_DATA_ROOT")
    return Path(configured).expanduser().resolve() if configured else Path(default).resolve()


def json_safe(value: Any) -> Any:
    """Recursively map non-finite diagnostics to JSON null."""
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        converted = float(value)
        return converted if math.isfinite(converted) else None
    return value


def strict_json_dumps(value: Any, *, indent: int = 2) -> str:
    return json.dumps(json_safe(value), indent=indent, allow_nan=False) + "\n"
