#!/usr/bin/env python3
"""Run simple promotion policies in PromotionEnv and compare their revenue uplift.

Policies:
    none          no promotion
    flat-10       10% on every promotable commodity, every week
    flat-30       30% everywhere
    random        a random level per commodity per week (seeded)
    greedy        each commodity alone: the level with the best one-week uplift when only it
                  is discounted, then all those levels together (an upper bound for simple rules
                  if commodities did not interact)

    run_policies.py --checkpoint CK [--simulator base|v4] [--initialization INIT] --out FILE
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from gymnasium.utils.env_checker import check_env

sys.path.insert(0, str(Path(__file__).resolve().parent))
from promotion_env import PromotionEnv  # noqa: E402


def episode(env, policy):
    obs, info = env.reset(seed=0)
    total, uplift, markdown, steps = 0.0, 0.0, 0.0, []
    done = False
    while not done:
        obs, reward, done, _, info = env.step(policy(obs))
        total += reward
        uplift += info["uplift"]
        markdown += info["markdown"]
        steps.append({"week": int(info["week"]), "uplift": round(info["uplift"], 2), "markdown": round(info["markdown"], 2)})
    return {"mean_relative_revenue": total / len(steps), "total_uplift": uplift, "total_markdown": markdown,
            "weeks": steps}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--initialization", type=Path)
    p.add_argument("--simulator", choices=("base", "v4"), default="base")
    p.add_argument("--trips-per-week", type=int, default=500)
    p.add_argument("--weeks", type=int, nargs="*", help="weeks to simulate (default: the validation weeks)")
    p.add_argument("--draws", type=int, default=32)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    env = PromotionEnv(a.checkpoint, simulator=a.simulator, initialization=a.initialization,
                       trips_per_week=a.trips_per_week, weeks=a.weeks or None, draws=a.draws, threads=a.threads)
    K, L = len(env.categories), len(env.levels)
    check_env(env, skip_render_check=True)
    rng = np.random.default_rng(1)

    # greedy: the best level for each commodity on its own, from the first week
    t = time.time()
    env.reset(seed=0)
    w0 = env.weeks[0]
    base = env._week(w0)["baseline"]["revenue"]
    best = np.zeros(K, dtype=int)
    single = {}
    for k in range(K):
        gains = []
        for level in range(L):
            action = np.zeros(K, dtype=int)
            action[k] = level
            gains.append(env._simulate(w0, action)["revenue"] - base)
        best[k] = int(np.argmax(gains))
        single[env.categories[k]] = [round(g, 2) for g in gains]
    policies = {"none": lambda obs: np.zeros(K, dtype=int),
                "flat-10": lambda obs: np.full(K, 1),
                "flat-30": lambda obs: np.full(K, L - 1),
                "random": lambda obs: rng.integers(0, L, K),
                "greedy": lambda obs: best}
    report = {"simulator": a.simulator, "checkpoint": str(a.checkpoint), "weeks": [int(w) for w in env.weeks],
              "trips_per_week": a.trips_per_week, "categories": env.categories,
              "levels": env.levels.tolist(),
              "single_category_uplift_first_week": single,
              "greedy_levels": dict(zip(env.categories, env.levels[best].tolist())), "policies": {}}
    for name, policy in policies.items():
        report["policies"][name] = episode(env, policy)
        r = report["policies"][name]
        print(f"[policies] {name:8s} uplift {r['total_uplift']:+10.2f}  markdown {r['total_markdown']:9.2f}  "
              f"relative revenue {r['mean_relative_revenue']:.4f}  ({time.time() - t:.0f}s)", flush=True)
    a.out.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
