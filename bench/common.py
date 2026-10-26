"""Shared setup for the bench scripts: load a fitted Version-4 model and held-out trips.

Every script takes the same first options:
    --world syn|real   which model (paths from bench/env.sh, read from the environment)
    --threads N        torch threads (default 4)
and prints one JSON object per line (flushed), so logs can be followed with tail -f.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_default_dtype(torch.float64)

from v4.bundle import load_bundle, supported_trips  # noqa: E402
from v4.checkpoints import freeze, load_version4  # noqa: E402

WORLDS = {"syn": ("SYN_RUN_DIR", "SYN_DATA", "SYN_CK"), "real": ("REAL_RUN_DIR", "REAL_DATA", "REAL_CK")}


def parser(doc):
    p = argparse.ArgumentParser(description=doc.splitlines()[0])
    p.add_argument("--world", choices=WORLDS, required=True)
    p.add_argument("--threads", type=int, default=4)
    return p


def emit(**row):
    print(json.dumps(row), flush=True)


def load(args):
    """(model, batcher, held-out trips).  Held-out trips are the fit's own validation split
    (seed 41001 + 99, as in scripts/fit_interactions.py), in a fixed order."""
    names = WORLDS[args.world]
    missing = [n for n in names if not os.environ.get(n)]
    if missing:
        raise SystemExit(f"{missing} not set: run  source bench/env.sh  first")
    run_dir, data_dir, ck = (os.environ[n] for n in names)
    torch.set_num_threads(args.threads)
    os.chdir(run_dir)                                   # checkpoints record paths relative to it
    data = load_bundle(Path(data_dir))
    model, batcher, _, _ = load_version4(ck, data, Path(data_dir))
    freeze(model)
    held = np.random.default_rng(41001 + 99).permutation(supported_trips(data, 0, int(model.nmax)))
    emit(world=args.world, checkpoint=ck, nmax=int(model.nmax), R=int(model.R), rank=int(model.rank),
         size_on_categories=bool(model.size_on_categories), threads=args.threads)
    return model, batcher, held


def batch_of(model, batcher, trips):
    b = batcher.make(np.sort(np.asarray(trips)))
    b.bind(model)
    return b
