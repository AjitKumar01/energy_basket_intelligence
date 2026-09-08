"""Resolve the licensed raw-data directory consistently for standalone stages."""
from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RELEASE_SUBDIRECTORY = Path(
    "dunnhumby_The-Complete-Journey") / "dunnhumby_The-Complete-Journey CSV"


def resolve_raw_directory() -> Path:
    """Honor NF_RAW_DIR, then check sibling and repository-local layouts."""
    configured = os.environ.get("NF_RAW_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    candidates = (ROOT.parent / RELEASE_SUBDIRECTORY,
                  ROOT / RELEASE_SUBDIRECTORY)
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    return candidates[0].resolve()
