#!/usr/bin/env python3
"""Run the evidence-gated retail API locally."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import uvicorn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--workers", type=int, default=1,
                        help="use one worker unless memory capacity is verified")
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()
    uvicorn.run(
        "retail_api.app:app", host=args.host, port=args.port,
        workers=args.workers, log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
