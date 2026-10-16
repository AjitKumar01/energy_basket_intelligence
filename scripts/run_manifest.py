"""Run logging and the per-invocation status manifest for the pipeline driver."""
from __future__ import annotations

import json


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
