#!/usr/bin/env python3
"""Check the promotion environment against the truth of a synthetic stress world.

The stress-world generator (synthetic/generate_stress_world.py) gives every trip its own
random stream, so re-simulating the test weeks with changed prices from the same household
state gives the true effect of any promotion (common random numbers).  This script rebuilds
the world from its seed (and checks the rebuild reproduces the observed baskets), then, for
the environment's promotable categories and discount levels, compares

    truth   expected revenue of the re-simulated test trips at paid prices
    model   PromotionEnv's expected revenue on the same trips (base model or Version 4)

for: each category alone at each level, and every category at each level.  Revenue is
summed over the test trips the model sees (nonempty factual baskets within nmax); trips that
were empty in the factual world and buy something under the promotion are reported apart
(the model is conditional on a trip, so it cannot see them).  Units per product are the
training averages in both.  Also: the model's chosen level per category (best one-week
revenue when only that category is promoted) and what the combined choice earns in truth.

    stress_check.py --world WORLD --out FILE [--checkpoint CK --simulator base|v4 ...]
    (without --checkpoint: only the truth, cached in --truth-cache)

Results are only comparable when their base models were fitted with the same settings (a
large world fitted with --elast-target -1.86 against small ones with the default -0.121 once
passed for a scale effect).  Every report therefore records ``base_fit``: the base model's
elasticity anchor, price contract and discount split (for Version 4, those of the base model
it was fitted from).  Before comparing reports, check them:

    stress_check.py --same-fit A.json B.json ...

which exits with status 1 and lists the differences if the settings differ or cannot be
read.  Reports written before ``base_fit`` existed are read through the checkpoint they name.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sim"))


def load_generator():
    spec = importlib.util.spec_from_file_location("gen", ROOT / "synthetic/generate_stress_world.py")
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    return gen


class Truth:
    """The rebuilt world: test trips, the household state at the start of the test weeks, and
    the map from generator trips to the model's trip rows."""

    def __init__(self, world_dir: Path):
        gen = load_generator()
        meta = json.loads((world_dir / "truth.json").read_text())
        if "args" in meta:                                   # worlds that record every setting
            args = gen.argparse.Namespace(**meta["args"])
        else:                                                # older worlds: the defaults they were made with
            args = gen.argparse.Namespace(seed=meta["seed"], households=3000, stores=meta["stores"], departments=6,
                                          categories_per_department=8, mission_variants=1, weeks=meta["weeks"],
                                          train_weeks=meta["train_weeks"], validation_weeks=meta["validation_weeks"],
                                          scenario_products=16, category_growth=meta.get("category_growth", 0.0))
        rng = np.random.default_rng(args.seed)
        world = gen.World(args, rng)
        H, S, T = args.households, args.stores, args.weeks
        visits = rng.random((H, T)) < world.visit_rate[:, None]
        hh, wk = np.nonzero(visits)
        store = np.where(rng.random(len(hh)) < 0.85, world.home[hh], rng.integers(0, S, len(hh)))
        day = wk * 7 + rng.integers(0, 7, len(hh))
        order = np.lexsort((hh, day)); hh, wk, store = hh[order], wk[order], store[order]
        trip_id = np.arange(len(hh))
        state0 = {"loyalty": np.full((H, world.C), -1), "inventory": np.zeros((H, world.C), dtype=int),
                  "taste": world.taste0.copy(), "week": 0}
        test_start = args.train_weeks + args.validation_weeks
        before = wk < test_start
        baskets_a, self.state = gen.simulate(world, (hh[before], store[before], wk[before], trip_id[before]),
                                             state0, seed_base=args.seed)
        self.trips = (hh[~before], store[~before], wk[~before], trip_id[~before])
        baskets_b, _ = gen.simulate(world, self.trips, self.state, seed_base=args.seed)
        baskets = baskets_a + baskets_b

        # the generator's cohort rules, to map its trips to the model's rows
        keep = np.array([len(b) > 0 for b in baskets])
        period = wk + 1
        trained = np.unique(hh[keep & (period <= args.train_weeks)])
        keep &= (period <= args.train_weeks) | np.isin(hh, trained)
        train_items = np.zeros(world.J, dtype=bool)
        for i in np.flatnonzero(keep & (period <= args.train_weeks)):
            train_items[baskets[i]] = True
        item_map = np.full(world.J, -1); item_map[train_items] = np.arange(int(train_items.sum()))
        keep &= np.array([any(item_map[j] >= 0 for j in b) for b in baskets])
        row = np.full(len(baskets), -1); row[keep] = np.arange(int(keep.sum()))

        observed = np.load(world_dir / "observed.npz")
        rebuilt = np.zeros_like(observed["memberships"])
        for i in np.flatnonzero(keep):
            rebuilt[row[i], [item_map[j] for j in baskets[i] if item_map[j] >= 0]] = True
        if not np.array_equal(rebuilt, observed["memberships"]):
            raise RuntimeError("the rebuilt world does not reproduce observed.npz")
        self.gen, self.world, self.test_start, self.item_map = gen, world, test_start, item_map
        self.test_row = row[~before]                         # model row of each test trip, or -1
        self.hh_trained = np.isin(hh[~before], trained)
        self.seed = args.seed

    def revenue(self, categories, depth, units, rows):
        """True revenue at paid prices when category c (generator id) is discounted by depth[c].
        Returns (revenue on the given model rows, revenue on trips with no model row whose
        households are in the panel, per-category revenue on the rows)."""
        price = self.world.price.copy()
        d_item = np.zeros(self.world.J)
        for c, d in zip(categories, depth):
            d_item[self.world.category == c] = d
        price[:, self.test_start:] *= (1.0 - d_item)[:, None]
        baskets, _ = self.gen.simulate(self.world, self.trips, self.state, price=price, seed_base=self.seed)
        wanted = np.zeros(int(self.test_row.max()) + 1, dtype=bool); wanted[rows] = True
        cat_of = {c: k for k, c in enumerate(categories)}
        on_rows, off_rows, by_cat = 0.0, 0.0, np.zeros(len(categories))
        wk = self.trips[2]
        for i, b in enumerate(baskets):
            items = [j for j in b if self.item_map[j] >= 0]
            if not items:
                continue
            value = np.array([units[self.item_map[j]] * price[j, wk[i]] for j in items])
            r = self.test_row[i]
            if r >= 0 and wanted[r]:
                on_rows += value.sum()
                for j, v in zip(items, value):
                    k = cat_of.get(int(self.world.category[j]))
                    if k is not None:
                        by_cat[k] += v
            elif r < 0 and self.hh_trained[i]:
                off_rows += value.sum()
        return on_rows, off_rows, by_cat


def actions(K, L):
    """(name, level index per category): each category alone at each level, then flat levels."""
    out = [("none", np.zeros(K, dtype=int))]
    for k in range(K):
        for l in range(1, L):
            a = np.zeros(K, dtype=int); a[k] = l
            out.append((f"cat{k}@{l}", a))
    for l in range(1, L):
        out.append((f"flat@{l}", np.full(K, l)))
    return out


def summarise(rows, K, L, levels):
    """Model vs truth: relative uplift of each action, chosen levels, and the combined choice."""
    def uplift(r, side):
        return r[side]["revenue"] / rows[0][side]["revenue"] - 1.0
    single = [r for r in rows if r["name"].startswith("cat")]
    t = np.array([uplift(r, "truth") for r in single]); m = np.array([uplift(r, "model") for r in single])
    best = {}
    for side in ("truth", "model"):
        choice = []
        for k in range(K):
            opts = [(0.0, 0)] + [(uplift(r, side), l) for l in range(1, L)
                                 for r in single if r["name"] == f"cat{k}@{l}"]
            choice.append(max(opts)[1])
        best[side] = choice
    flat = {r["name"]: {"truth": uplift(r, "truth"), "model": uplift(r, "model")} for r in rows if r["name"].startswith("flat")}
    return {"single_category_uplift_corr": float(np.corrcoef(t, m)[0, 1]) if t.std() > 0 and m.std() > 0 else None,
            "single_category_uplift_mean": {"truth": float(t.mean()), "model": float(m.mean())},
            "single_category_uplift_mae": float(np.abs(t - m).mean()),
            "flat": flat,
            "chosen_levels": {s: [float(levels[l]) for l in best[s]] for s in best},
            "same_level_share": float(np.mean(np.array(best["truth"]) == np.array(best["model"])))}


BASE_FIT_KEYS = ("elast_w", "elast_target", "price_feature_contract", "discount_panel", "disc_kappa", "disc_common")


def _existing(path, *anchors):
    """``path`` if it exists, else the first existing ``anchor / path`` (relative paths)."""
    path = Path(path)
    for cand in [path] + ([Path(a) / path for a in anchors] if not path.is_absolute() else []):
        if cand.exists():
            return cand.resolve()
    raise FileNotFoundError(path)


def base_fit(checkpoint, simulator, *anchors) -> dict:
    """The fit settings of the base model behind ``checkpoint`` (BASE_FIT_KEYS)."""
    import torch
    path = _existing(checkpoint, *anchors)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if simulator == "v4":
        path = _existing(ck["additive"], *anchors, *path.parents)
        ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck.get("config") or ck.get("args") or {}
    cfg = cfg if isinstance(cfg, dict) else vars(cfg)
    out = {"base_checkpoint": str(path)}
    for k in BASE_FIT_KEYS:
        v = cfg.get(k)
        out[k] = str(Path(v).resolve()) if k == "discount_panel" and v else (v if isinstance(v, (int, float, bool, type(None))) else str(v))
    return out


def same_fit(reports) -> int:
    """Check that the reports' base models share their fit settings; 0 if so, else 1."""
    fits, bad = {}, []
    for r in reports:
        rep = json.loads(Path(r).read_text())
        try:
            fits[r] = rep.get("base_fit") or base_fit(rep["checkpoint"], rep["simulator"], Path.cwd(),
                                                      *Path(r).resolve().parents)
        except (FileNotFoundError, KeyError) as e:
            bad.append(f"{r}: base fit settings unreadable ({e})")
    keys = [k for k in BASE_FIT_KEYS if len({json.dumps((f or {}).get(k)) for f in fits.values()}) > 1]
    for k in keys:
        bad.append(f"{k} differs: " + ", ".join(f"{r}={fits[r].get(k)!r}" for r in fits))
    for r, f in fits.items():
        print(f"[same-fit] {r}: " + " ".join(f"{k}={f.get(k)}" for k in BASE_FIT_KEYS), flush=True)
    for b in bad:
        print(f"[same-fit] REFUSED: {b}", flush=True)
    if not bad:
        print(f"[same-fit] OK: {len(fits)} reports share their base fit settings", flush=True)
    return 1 if bad else 0


def main():
    if "--same-fit" in sys.argv:
        q = argparse.ArgumentParser(description="check that reports share their base fit settings")
        q.add_argument("--same-fit", nargs="+", type=Path, required=True)
        raise SystemExit(same_fit(q.parse_args().same_fit))
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--same-fit", nargs="+", type=Path, help="only check that these reports share their base fit settings")
    p.add_argument("--world", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--initialization", type=Path)
    p.add_argument("--simulator", choices=("base", "v4"), default="base")
    p.add_argument("--latent", choices=("reweight", "fixed"), default="reweight")
    p.add_argument("--draws", type=int, default=32)
    p.add_argument("--categories", type=int, default=12)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--truth-cache", type=Path, help="JSON of true revenues per action (written if missing)")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    import os
    os.environ["ENERGY_MODEL_DATA_ROOT"] = str((a.world / "model_input_category").resolve())
    from promotion_env import PromotionEnv
    started = time.time()

    meta = json.loads((a.world / "truth.json").read_text())
    test_weeks = list(range(meta["train_weeks"] + meta["validation_weeks"] + 1, meta["weeks"] + 1))
    env = PromotionEnv(a.checkpoint, simulator=a.simulator, initialization=a.initialization, weeks=test_weeks,
                       n_categories=a.categories, trips_per_week=10 ** 9, draws=a.draws, threads=a.threads,
                       latent=a.latent) if a.checkpoint else None
    if env is None:                                          # truth only: the environment's choices without a model
        raise SystemExit("--checkpoint is needed to fix the environment's trips and categories")
    fit = base_fit(a.checkpoint, a.simulator)
    print("[stress_check] base fit: " + " ".join(f"{k}={fit[k]}" for k in BASE_FIT_KEYS), flush=True)
    K, L, levels = len(env.categories), len(env.levels), env.levels
    rows_used = np.concatenate([env.week_trips[w] for w in test_weeks])
    gen_cats = [int(c[3:]) for c in env.categories]          # 'cat017' -> 17

    cache = json.loads(a.truth_cache.read_text()) if a.truth_cache and a.truth_cache.exists() else {}
    key = f"{K}:{len(rows_used)}:{rows_used.sum()}"
    truth_rows = cache.get(key)
    truth = None
    if truth_rows is None:
        truth = Truth(a.world)
        print(f"[stress_check] world rebuilt and verified ({time.time() - started:.0f}s)", flush=True)
        truth_rows = {}
        for name, act in actions(K, L):
            on, off, by_cat = truth.revenue(gen_cats, levels[act], env.units, rows_used)
            truth_rows[name] = {"revenue": on, "new_trips_revenue": off, "by_category": by_cat.tolist()}
        print(f"[stress_check] truth for {len(truth_rows)} actions ({time.time() - started:.0f}s)", flush=True)
        if a.truth_cache:
            cache[key] = truth_rows
            a.truth_cache.write_text(json.dumps(cache) + "\n")

    rows = []
    for name, act in actions(K, L):
        out = {"revenue": 0.0, "by_category": np.zeros(K)}
        for w in test_weeks:
            env._week(w)
            s = env._simulate(w, act)
            out["revenue"] += s["revenue"]; out["by_category"] += s["revenue_by_category"]
        rows.append({"name": name, "levels": levels[act].tolist(), "truth": truth_rows[name],
                     "model": {"revenue": out["revenue"], "by_category": out["by_category"].tolist()}})
        print(f"[stress_check] {name}: truth {truth_rows[name]['revenue'] / truth_rows['none']['revenue'] - 1:+.4f} "
              f"model {out['revenue'] / rows[0]['model']['revenue'] - 1:+.4f}", flush=True)
    summary = summarise(rows, K, L, levels)

    # what the model's combined choice earns in truth, against the truth's own combined choice
    if truth is None:
        truth = Truth(a.world)
    combined = {}
    for side in ("truth", "model"):
        depth = np.array(summary["chosen_levels"][side])
        on, _, _ = truth.revenue(gen_cats, depth, env.units, rows_used)
        combined[side] = on / truth_rows["none"]["revenue"] - 1.0
    summary["combined_choice_true_uplift"] = combined

    report = {"world": str(a.world), "checkpoint": str(a.checkpoint), "simulator": a.simulator,
              "latent": a.latent if a.simulator == "v4" else None, "base_fit": fit, "categories": env.categories,
              "levels": levels.tolist(), "test_weeks": test_weeks, "trips": int(len(rows_used)),
              "summary": summary, "actions": rows, "seconds": round(time.time() - started, 1)}
    a.out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
