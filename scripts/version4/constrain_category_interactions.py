#!/usr/bin/env python3
"""Project a Version-4 checkpoint onto the complete-support category-safe set.

This is a deterministic parameter projection, not a new probability model.  It is useful
for auditing an already learned checkpoint before the same constraint is used throughout
a fresh fit.  The optional matched parent differs from the child only by ``Phi = 0``.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("V3_AFFINITY", "1")

import numpy as np
import torch

from audit_particle_counterfactual_generation import ROOT, load_checkpoint
from category_safety import (attractive_category_rewards, category_capacities,
                             project_category_reward_)
from data import build


def atomic_save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--max-category-reward", type=float, default=1.5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--matched-parent-output", type=Path)
    args = parser.parse_args()

    data = build()
    checkpoint = args.checkpoint if args.checkpoint.is_absolute() \
        else ROOT / args.checkpoint
    output = args.output if args.output.is_absolute() else ROOT / args.output
    parent_output = (None if args.matched_parent_output is None else
                     (args.matched_parent_output if args.matched_parent_output.is_absolute()
                      else ROOT / args.matched_parent_output))
    model, blob, meta = load_checkpoint(checkpoint, data)
    capacities = category_capacities(data, int(data["n_cat"]), int(meta["nmax"]))
    before = attractive_category_rewards(
        model.rho_c, capacities).detach().cpu().numpy()
    projection = project_category_reward_(
        model, capacities, args.max_category_reward)
    after = attractive_category_rewards(
        model.rho_c, capacities).detach().cpu().numpy()

    payload = dict(blob)
    payload["model"] = model.state_dict()
    payload["estimator"] = str(blob.get("estimator", "")) + "+category_support_projection"
    payload["category_support_parent"] = str(checkpoint)
    payload["category_support_max_reward"] = float(args.max_category_reward)
    payload["category_support_projection"] = projection
    payload["optimizer"] = None
    payload["best_validation"] = None
    payload["evaluations"] = []
    payload["records"] = []
    atomic_save(output, payload)

    changed = np.flatnonzero(np.abs(after - before) > 1e-12)
    report = {
        "checkpoint": str(checkpoint),
        "output": str(output),
        "probability_law": "unchanged Version-4 energy",
        "constraint": "(-rho_c)_+ * choose(m_c,2) <= max_category_reward",
        "max_category_reward": float(args.max_category_reward),
        "projection": projection,
        "changed_categories": int(len(changed)),
        "largest_rewards_before": sorted(map(float, before), reverse=True)[:10],
        "largest_rewards_after": sorted(map(float, after), reverse=True)[:10],
    }
    if parent_output is not None:
        parent = dict(payload)
        parent_state = {key: value.clone() for key, value in payload["model"].items()}
        parent_state["phi"].zero_()
        parent["model"] = parent_state
        parent["estimator"] = str(payload["estimator"]) + "+matched_phi_zero_parent"
        parent["matched_child"] = str(output)
        atomic_save(parent_output, parent)
        report["matched_parent_output"] = str(parent_output)
    report_path = output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
