#!/usr/bin/env python3
"""Print the hardware/backend decision without reading data or starting training."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from runtime_capabilities import detect_runtime, resolve_backend


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"),
                        default="auto")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    capabilities = detect_runtime(Path.cwd())
    try:
        selected = resolve_backend(args.device, capabilities)
    except RuntimeError as exc:
        selected = None
        error = str(exc)
    else:
        error = None
    payload = asdict(capabilities) | {
        "requested_device": args.device,
        "selected_device": selected,
        "error": error,
    }
    rendered = json.dumps(payload, indent=2) + "\n"
    print(rendered, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    if error:
        raise SystemExit(error)


if __name__ == "__main__":
    main()
