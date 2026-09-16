"""Run logging, atomic status manifests and frozen-source identity for driver scripts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class Tee:
    """Mirror a text stream to a new log file that must not already exist."""

    def __init__(self, original, path):
        self.original = original
        self.stream = path.open("x", buffering=1)

    def write(self, text):
        self.original.write(text)
        self.stream.write(text)
        return len(text)

    def flush(self):
        self.original.flush()
        self.stream.flush()


def write_manifest(path, value):
    temporary = path.with_suffix(".pending.json")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def source_identity():
    """Digest every source and test file so a long run can refuse mid-run code edits."""
    paths = sorted((ROOT / "scripts").rglob("*.py"))
    paths += sorted((ROOT / "scripts").rglob("*.cpp"))
    paths += sorted((ROOT / "tests").glob("*.py"))
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths}
