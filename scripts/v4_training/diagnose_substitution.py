#!/usr/bin/env python3
"""Step 1: Diagnose price substitution on the 32 stress-world scenarios.

For each of the 32 single-product price scenarios (±20% on 16 products), decomposes the
response of both the causal truth and a Version-4 checkpoint (e.g. runs/stress/hs_rec.pt) into:
  1. Category incidence: P(at least one product of the category, n_c >= 1)
     and total category purchases per trip: E[n_c] = sum_{j in c} P(j in S).
  2. Within-category share: each product's (and product group's: own, same_subcategory,
     rest_of_category) share of the category's purchases, P(j in S) / sum_{i in c} P(i in S),
     as well as incidence per category-present trip, P(j in S) / P(n_c >= 1).

Writes runs/stress/substitution_diagnosis.json.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import sys
import time
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

sys.modules.setdefault("pathlib._local", pathlib)
os.environ.setdefault("V3_AFFINITY", "1")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/version4"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "scripts/synthetic"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(4)
torch.set_default_dtype(torch.float64)

import generate_stress_world as gsw  # noqa: E402
import stress  # noqa: E402
from price_response import changed_price_context  # noqa: E402
from stress import (  # noqa: E402
    Batch,
    basket_field,
    configure_model_from_blob,
    gibbs,
    log_poly_2d_native,
    phi_r,
    row_cat_shift,
    sample_slots,
    setup,
    slot_utilities,
)


def fast_forward_eval_generator(
    gen: torch.Generator,
    n_trips: int = 4096,
    eval_batch: int = 256,
    r: int = 16,
    burn: int = 200,
    components: int = 128,
    thin: int = 4,
    draws: int = 1024,
    sv0_gt_1: bool = True,
) -> None:
    """Advance `gen` through the exact RNG calls made by Step 1 (`is_log_z`) of `stress.py::evaluate`
    when `USE_CPP_SAMPLER=True`, so Step 3 (`price_scenarios`) starts from the identical RNG state."""
    k_per_ch = max(components // 2, 64)
    thin_c = max(thin // 2, 1)
    steps_0 = max(components * thin, k_per_ch * thin_c)
    h = draws // 2
    h2 = draws - h
    for s in range(0, n_trips, eval_batch):
        B = min(eval_batch, n_trips - s)
        # Chain 0: sample_slots (1) + gibbs burn (1) + steps_0 gibbs calls
        for _ in range(2 + steps_0):
            torch.randint(1, 2**62, (1,), generator=gen)
        # Chains 1..3
        for ch in range(1, 4):
            # 1 sample_slots + 4 tempered gibbs + (k_per_ch * thin_c) gibbs
            for _ in range(1 + 4 + k_per_ch * thin_c):
                torch.randint(1, 2**62, (1,), generator=gen)
        torch.rand(B, h, generator=gen)
        torch.rand(B, h, generator=gen)
        torch.rand(B, h2, generator=gen)
        torch.rand(B, h2, generator=gen)
        torch.randn(B, draws, r, generator=gen)


def log_zc_with_cat_presence_shift(
    model, ix, u: torch.Tensor, row_pres_shift: torch.Tensor, z: torch.Tensor | None = None, beta: float = 1.0
) -> torch.Tensor:
    """Compute exact log Z_c(u(z)) with an explicit differentiable shift `row_pres_shift[r]`
    added to log_g[r, 1:] on each row r.
    By differentiating sum_b log Z_c with respect to (u, row_pres_shift):
      - d(sum log Z_c)/du_t = P(slot t in S | z)
      - d(sum log Z_c)/d(row_pres_shift_r) = P(n_r >= 1 | z) = P(at least one product of row r's category | z)
    """
    from poly_degree_native import log_poly_tree_degree_native
    from ragged import esp_log_bucketed, seg_max

    has_f = getattr(model, "f_extra", None) is not None
    has_m2 = getattr(model, "presence_phi", False) and beta != 0.0
    has_m3 = getattr(model, "cat_rho0", False)
    has_m4b = getattr(model, "iv_slope", False) and getattr(model, "iv_slope_raw", None) is not None

    scale = seg_max(u.unsqueeze(0), ix.item_trip, ix.B)[0]
    centred = u - scale[ix.item_trip]
    log_e = esp_log_bucketed(
        centred.unsqueeze(0), ix.row_of, ix.n_rows, model.R,
        ix.row_size, ix.item_pos
    )[0]
    degree_axis = torch.arange(model.R + 1, dtype=u.dtype, device=u.device)
    log_g = log_e - model.rho_c[ix.row_cat].unsqueeze(-1) * model.pair_feature(degree_axis)
    if has_f:
        fe = model.f_extra[ix.row_cat]
        fe_pad = torch.zeros(ix.n_rows, model.R + 1, dtype=u.dtype, device=u.device)
        if model.R >= 2:
            fe_pad[:, 2] = fe[:, 0]
        if model.R >= 3:
            fe_pad[:, 3] = fe[:, 1]
        if model.R >= 4:
            fe_pad[:, 4:] = fe[:, 2:3]
        log_g = log_g - fe_pad
    if has_m2:
        cshift = row_cat_shift(model, ix, z, beta=beta)
        if cshift is not None:
            km1 = (degree_axis - 1.0).clamp_min(0.0)
            log_g = log_g - cshift[:, None] * km1[None, :]
    if has_m4b and hasattr(stress, "row_iv_shift"):
        iv_sh = stress.row_iv_shift(model, ix)
        if iv_sh is not None:
            log_g = torch.cat([log_g[:, :1], log_g[:, 1:] + iv_sh[:, None]], dim=1)

    log_g = torch.cat([log_g[:, :1], log_g[:, 1:] + row_pres_shift[:, None]], dim=1)

    category = torch.full(
        (1, ix.B * ix.Cpad, model.R + 1), -float("inf"), dtype=u.dtype, device=u.device
    )
    category[..., 0] = 0.0
    category = category.index_copy(1, ix.flat_slot, log_g.unsqueeze(0)).view(
        1, ix.B, ix.Cpad, model.R + 1
    )
    degrees = torch.zeros(ix.B * ix.Cpad, dtype=torch.long, device=u.device)
    degrees[ix.flat_slot] = ix.row_size.clamp(max=model.R)
    degrees = degrees.view(ix.B, ix.Cpad)
    slope = (category[..., 1:] / degree_axis[1:]).amax(dim=(2, 3)).clamp_min(0.0)
    tilted = category - slope[:, :, None, None] * degree_axis
    if has_m3:
        log_a2d = log_poly_2d_native(tilted[0].contiguous(), degrees.contiguous(), model.nmax)
        size_axis = torch.arange(model.nmax + 1, dtype=u.dtype, device=u.device)
        log_a2d = log_a2d + slope[0, :, None, None] * size_axis[None, :, None]
        log_joint = (
            log_a2d[:, 1:, 1:]
            + size_axis[None, 1:, None] * scale[:, None, None]
            - model.rho_0()[None, None, 1:model.nmax + 1]
        )
        return torch.logsumexp(log_joint.reshape(ix.B, -1), dim=-1)
    log_a = log_poly_tree_degree_native(tilted.contiguous(), degrees.contiguous(), model.nmax)
    log_a = log_a + slope.unsqueeze(-1) * degree_axis
    size_axis = torch.arange(log_a.shape[-1], dtype=u.dtype, device=u.device)
    log_size = (log_a[0] + size_axis * scale.unsqueeze(-1) - model.rho_0()[:log_a.shape[-1]])[:, 1:]
    return torch.logsumexp(log_size, dim=-1)


def compute_exact_truth_scenarios(world_dir: Path) -> dict[tuple[int, float], dict]:
    """Recompute exact causal truth with common random numbers using generate_stress_world.py's
    causal-truth block, recording exact category incidence P(n_c >= 1) alongside product incidences,
    and verifying exact agreement with saved counterfactual_truth.json."""
    truth_meta = json.loads((world_dir / "truth.json").read_text())
    saved_cf = json.loads((world_dir / "counterfactual_truth.json").read_text())
    obs = np.load(world_dir / "observed.npz")
    kept = obs["original_item"].astype(np.int64)

    w_args = argparse.Namespace(
        seed=int(truth_meta["seed"]),
        households=3000,
        stores=int(truth_meta["stores"]),
        departments=6,
        categories_per_department=8,
        mission_variants=1,
        weeks=int(truth_meta["weeks"]),
        train_weeks=int(truth_meta["train_weeks"]),
        validation_weeks=int(truth_meta["validation_weeks"]),
        scenario_products=len(truth_meta["scenario_products"]),
    )
    rng = np.random.default_rng(w_args.seed)
    world = gsw.World(w_args, rng)
    H, S, T, J, C = w_args.households, w_args.stores, w_args.weeks, world.J, world.C

    visits = rng.random((H, T)) < world.visit_rate[:, None]
    hh, wk = np.nonzero(visits)
    store = np.where(rng.random(len(hh)) < 0.85, world.home[hh], rng.integers(0, S, len(hh)))
    day = wk * 7 + rng.integers(0, 7, len(hh))
    order = np.lexsort((hh, day))
    hh, wk, store, day = hh[order], wk[order], store[order], day[order]
    trip_id = np.arange(len(hh))
    state0 = {
        "loyalty": np.full((H, C), -1),
        "inventory": np.zeros((H, C), dtype=int),
        "taste": world.taste0.copy(),
        "week": 0,
    }
    test_start = w_args.train_weeks + w_args.validation_weeks
    before = wk < test_start
    baskets_a, state_test = gsw.simulate(
        world, (hh[before], store[before], wk[before], trip_id[before]), state0, seed_base=w_args.seed
    )
    test_trips = (hh[~before], store[~before], wk[~before], trip_id[~before])
    baskets_b, _ = gsw.simulate(world, test_trips, state_test, seed_base=w_args.seed)

    counts = np.zeros(J)
    for i in np.flatnonzero(before):
        counts[baskets_a[i]] += 1
    eligible = np.flatnonzero(counts >= 30)
    chosen = np.sort(rng.choice(eligible, size=min(w_args.scenario_products, len(eligible)), replace=False))
    assert chosen.tolist() == truth_meta["scenario_products"], "Scenario products mismatch"

    base_inc = np.zeros(J, dtype=np.float64)
    base_cat_pres = np.zeros(C, dtype=np.float64)
    base_size = []
    for b in baskets_b:
        if b:
            base_inc[b] += 1.0
            cats_in_b = np.unique(world.category[b])
            base_cat_pres[cats_in_b] += 1.0
            base_size.append(len(b))
    base_n = len(base_size)

    # Also check observed test baskets in observed.npz (after train_items & trained_households filtering)
    period_obs = obs["trip_period"]
    test_mask_obs = period_obs > test_start
    mem_test_obs = obs["memberships"][test_mask_obs]
    obs_test_n = int(mem_test_obs.shape[0])
    obs_base_inc_world = np.zeros(J, dtype=np.float64)
    obs_base_inc_world[kept] = mem_test_obs.mean(axis=0)
    obs_base_cat_pres = np.zeros(C, dtype=np.float64)
    for c_id in range(C):
        cat_cols = np.flatnonzero(obs["category"] == c_id)
        obs_base_cat_pres[c_id] = float(mem_test_obs[:, cat_cols].any(axis=1).mean())

    saved_map = {(int(s["product"]), float(s["price_multiplier"])): s for s in saved_cf["scenarios"]}
    out = {}
    for k in chosen:
        c_k = int(world.category[k])
        s_k = int(world.subcategory[k])
        cat_items = np.flatnonzero(world.category == c_k)
        same_sub = np.flatnonzero((world.category == c_k) & (world.subcategory == s_k) & (np.arange(J) != k))
        same_cat = np.flatnonzero((world.category == c_k) & (np.arange(J) != k))
        rest = np.setdiff1d(same_cat, same_sub)

        for multiplier in (0.8, 1.2):
            changed = world.price.copy()
            changed[k, test_start:] = np.round(changed[k, test_start:] * multiplier, 2)
            alt, _ = gsw.simulate(world, test_trips, state_test, price=changed, seed_base=w_args.seed)
            inc = np.zeros(J, dtype=np.float64)
            alt_cat_pres = np.zeros(C, dtype=np.float64)
            size = []
            for b in alt:
                if b:
                    inc[b] += 1.0
                    cats_in_b = np.unique(world.category[b])
                    alt_cat_pres[cats_in_b] += 1.0
                    size.append(len(b))
            alt_n = len(size)

            sc_saved = saved_map[(int(k), float(multiplier))]
            assert [base_n, alt_n] == sc_saved["nonempty_test_baskets"]
            diff_all = (inc / alt_n - base_inc / base_n)
            max_err = float(np.abs(diff_all - np.array(sc_saved["incidence_change_all_products"])).max())
            assert max_err < 1e-15, f"Truth CRN mismatch on ({k}, {multiplier}): {max_err}"

            base_rate = base_inc / base_n
            alt_rate = inc / alt_n
            base_cat_inc = float(base_cat_pres[c_k] / base_n)
            alt_cat_inc = float(alt_cat_pres[c_k] / alt_n)
            base_cat_tot = float(base_rate[cat_items].sum())
            alt_cat_tot = float(alt_rate[cat_items].sum())

            out[(int(k), float(multiplier))] = {
                "category": c_k,
                "subcategory": s_k,
                "nonempty_test_baskets": [int(base_n), int(alt_n)],
                "category_incidence": [base_cat_inc, alt_cat_inc],
                "category_total_purchases": [base_cat_tot, alt_cat_tot],
                "own_incidence": [float(base_rate[k]), float(alt_rate[k])],
                "same_subcategory_incidence": [float(base_rate[same_sub].sum()), float(alt_rate[same_sub].sum())],
                "rest_of_category_incidence": [float(base_rate[rest].sum()), float(alt_rate[rest].sum())],
                "own_baskets": [float(base_inc[k]), float(inc[k])],
                "same_subcategory_baskets": [float(base_inc[same_sub].sum()), float(inc[same_sub].sum())],
                "rest_of_category_baskets": [float(base_inc[rest].sum()), float(inc[rest].sum())],
                "category_present_baskets": [float(base_cat_pres[c_k]), float(alt_cat_pres[c_k])],
                "mean_basket_size": [float(np.mean(base_size)), float(np.mean(size))],
                "obs_filtered_test_trips": obs_test_n,
                "obs_filtered_base_category_incidence": float(obs_base_cat_pres[c_k]),
            }
    return out


def evaluate_model_substitution(
    checkpoint_path: Path,
    world_dir: Path,
    initialization: Path,
    additive: Path,
    threads: int = 4,
    eval_batch: int = 256,
    burn: int = 200,
    components: int = 128,
    thin: int = 4,
    draws: int = 1024,
    scenario_trips: int = 400,
    scenario_draws: int = 64,
    seed: int = 41001,
) -> dict:
    """Evaluate a Version-4 checkpoint on the 32 price scenarios, returning both product
    incidences P(j in S) and exact category incidences P(n_c >= 1) per scenario."""
    stress.USE_CPP_SAMPLER = True
    blob = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    data, model, meta, batcher = setup(
        initialization, additive, threads, recency=bool(blob.get("recency", False))
    )
    configure_model_from_blob(model, blob)
    model.eval()
    for q in model.parameters():
        q.requires_grad_(False)
    r = int(blob["rank"])
    J = int(data["n_item"])
    C = int(data["n_cat"])

    gen = torch.Generator().manual_seed(seed + 5)
    sv0 = float(torch.linalg.svdvals(phi_r(model, r).detach())[0])
    fast_forward_eval_generator(
        gen, n_trips=4096, eval_batch=eval_batch, r=r,
        burn=burn, components=components, thin=thin, draws=draws,
        sv0_gt_1=(sv0 > 1.0),
    )

    o = np.load(world_dir / "observed.npz")
    truth_meta = json.loads((world_dir / "truth.json").read_text())
    period = o["trip_period"]
    split = np.where(
        period <= truth_meta["train_weeks"], 0,
        np.where(period <= truth_meta["train_weeks"] + truth_meta["validation_weeks"], 1, 2),
    )
    rows = np.sort(
        np.random.default_rng(77).choice(np.flatnonzero(split == 2), size=scenario_trips, replace=False)
    )
    to_model = {int(orig): i for i, orig in enumerate(o["original_item"])}
    truth_cf = json.loads((world_dir / "counterfactual_truth.json").read_text())
    cat, sub = o["category"], o["subcategory"]

    sets = []
    for s in range(0, len(rows), eval_batch):
        bt = Batch(batcher, rows[s:s + eval_batch], J)
        bt.bind(model)
        X = sample_slots(model, bt.ix, slot_utilities(model, bt.ix, None, r, beta=0.0), gen, beta=0.0)
        X = gibbs(model, bt.ix, X, r, gen, sweeps=burn)
        zs = []
        for k_step in range(scenario_draws * thin):
            X = gibbs(model, bt.ix, X, r, gen)
            if (k_step + 1) % thin == 0:
                zs.append(basket_field(model, bt.ix, X, r) + torch.randn(bt.ix.B, r, generator=gen))
        sets.append((bt, zs))

    def incidence_and_cat_presence(product=None, multiplier=1.0):
        inc = np.zeros(J, dtype=np.float64)
        cat_inc = np.zeros(C, dtype=np.float64)
        size = 0.0
        for bt, zs in sets:
            ix = bt.ix
            ctx = bt.ctx
            if product is not None:
                change = torch.where(ix.item == product, math.log(multiplier), 0.0)
                ctx = changed_price_context(bt.ctx, ix.item_trip, change)
            lzs, grads_u, grads_c, lzb = [], [], [], []
            for z in zs:
                bt.bind(model, ctx)
                u = slot_utilities(model, ix, z, r).detach().requires_grad_(True)
                p_sh = torch.zeros(ix.n_rows, dtype=u.dtype, requires_grad=True)
                with torch.enable_grad():
                    lz = log_zc_with_cat_presence_shift(model, ix, u, p_sh, z=z)
                    gu, gc = torch.autograd.grad(lz.sum(), (u, p_sh))
                lzs.append(lz.detach())
                grads_u.append(gu.detach())
                grads_c.append(gc.detach())
                if product is not None:
                    bt.bind(model)
                    p_sh0 = torch.zeros(ix.n_rows, dtype=u.dtype)
                    lzb.append(
                        log_zc_with_cat_presence_shift(
                            model, ix, slot_utilities(model, ix, z, r), p_sh0, z=z
                        ).detach()
                    )
            lzs = torch.stack(lzs, 1)  # B x K
            logw = (lzs - torch.stack(lzb, 1)) if product is not None else torch.zeros_like(lzs)
            w = torch.softmax(logw, 1)  # B x K
            Gu = torch.stack(grads_u, 1)  # T x K
            Gc = torch.stack(grads_c, 1)  # n_rows x K
            m_u = (Gu * w[ix.item_trip]).sum(1)
            m_c = (Gc * w[ix.row_trip]).sum(1)
            np.add.at(inc, ix.item.numpy(), m_u.numpy())
            np.add.at(cat_inc, ix.row_cat.numpy(), m_c.numpy())
            size += float(m_u.sum())
        bt.bind(model)
        n_tr = float(len(rows))
        return inc / n_tr, cat_inc / n_tr, size / n_tr

    base_inc, base_cat_inc, base_size = incidence_and_cat_presence()
    out_scenarios = {}
    for sc in truth_cf["scenarios"]:
        orig_p = int(sc["product"])
        mult = float(sc["price_multiplier"])
        k = to_model.get(orig_p)
        if k is None:
            continue
        alt_inc, alt_cat_inc, alt_size = incidence_and_cat_presence(k, mult)
        c_k = int(cat[k])
        s_k = int(sub[k])
        cat_items = np.flatnonzero(cat == c_k)
        same_sub = np.flatnonzero((cat == c_k) & (sub == s_k) & (np.arange(J) != k))
        rest_cat = np.flatnonzero((cat == c_k) & (sub != s_k))

        out_scenarios[(orig_p, mult)] = {
            "model_item_id": int(k),
            "category": c_k,
            "subcategory": s_k,
            "category_incidence": [float(base_cat_inc[c_k]), float(alt_cat_inc[c_k])],
            "category_total_purchases": [float(base_inc[cat_items].sum()), float(alt_inc[cat_items].sum())],
            "own_incidence": [float(base_inc[k]), float(alt_inc[k])],
            "same_subcategory_incidence": [float(base_inc[same_sub].sum()), float(alt_inc[same_sub].sum())],
            "rest_of_category_incidence": [float(base_inc[rest_cat].sum()), float(alt_inc[rest_cat].sum())],
            "has_rest_of_category": len(rest_cat) > 0,
            "mean_basket_size": [float(base_size), float(alt_size)],
        }
    return out_scenarios


def rel_change(pair: list[float]) -> float | None:
    a, b = pair
    return float(b / a - 1.0) if a > 0 else None


def build_diagnosis_report(
    truth_sc: dict[tuple[int, float], dict],
    model_sc: dict[tuple[int, float], dict],
    checkpoint_label: str,
    eval_json_path: Path | None = None,
) -> dict:
    """Assemble per-scenario and summary metrics splitting each scenario's change into
    (1) Category incidence P(n_c >= 1) (plus total category purchases E[n_c]) and
    (2) Within-category share P(j in S) / sum_{i in c} P(i in S) (plus P(j in S) / P(n_c >= 1))."""
    saved_eval_map = {}
    if eval_json_path is not None and eval_json_path.exists():
        ev = json.loads(eval_json_path.read_text())
        for row in ev.get("price_scenarios", []):
            saved_eval_map[(int(row["product"]), float(row["price_multiplier"]))] = row

    scenarios_out = []
    max_repro_diff = 0.0

    for (prod, mult), tr in truth_sc.items():
        md = model_sc[(prod, mult)]
        has_rest = md["has_rest_of_category"]

        # Category incidence P(n_c >= 1) and category total purchases sum_{i in c} P(i in S)
        tr_cat_inc_rel = rel_change(tr["category_incidence"])
        md_cat_inc_rel = rel_change(md["category_incidence"])
        tr_cat_tot_rel = rel_change(tr["category_total_purchases"])
        md_cat_tot_rel = rel_change(md["category_total_purchases"])

        # Within-category share of category purchases: s_G = P(G) / sum_{i in c} P(i in S)
        tr_tot0, tr_tot1 = tr["category_total_purchases"]
        md_tot0, md_tot1 = md["category_total_purchases"]
        tr_cinc0, tr_cinc1 = tr["category_incidence"]
        md_cinc0, md_cinc1 = md["category_incidence"]

        def _group_metrics(tr_pair, md_pair, valid=True):
            if not valid:
                return {
                    "incidence": {"truth_base": 0.0, "truth_alt": 0.0, "truth_rel_change": None,
                                  "model_base": 0.0, "model_alt": 0.0, "model_rel_change": None},
                    "within_category_share": {"truth_base": 0.0, "truth_alt": 0.0, "truth_rel_change": None, "truth_diff": None,
                                              "model_base": 0.0, "model_alt": 0.0, "model_rel_change": None, "model_diff": None},
                    "per_category_present_trip": {"truth_base": 0.0, "truth_alt": 0.0, "truth_rel_change": None,
                                                  "model_base": 0.0, "model_alt": 0.0, "model_rel_change": None},
                }
            t0, t1 = tr_pair
            m0, m1 = md_pair
            ts0, ts1 = (t0 / tr_tot0 if tr_tot0 > 0 else 0.0), (t1 / tr_tot1 if tr_tot1 > 0 else 0.0)
            ms0, ms1 = (m0 / md_tot0 if md_tot0 > 0 else 0.0), (m1 / md_tot1 if md_tot1 > 0 else 0.0)
            tc0, tc1 = (t0 / tr_cinc0 if tr_cinc0 > 0 else 0.0), (t1 / tr_cinc1 if tr_cinc1 > 0 else 0.0)
            mc0, mc1 = (m0 / md_cinc0 if md_cinc0 > 0 else 0.0), (m1 / md_cinc1 if md_cinc1 > 0 else 0.0)
            return {
                "incidence": {
                    "truth_base": t0, "truth_alt": t1, "truth_rel_change": rel_change([t0, t1]),
                    "model_base": m0, "model_alt": m1, "model_rel_change": rel_change([m0, m1]),
                },
                "within_category_share": {
                    "truth_base": ts0, "truth_alt": ts1,
                    "truth_rel_change": rel_change([ts0, ts1]),
                    "truth_diff": float(ts1 - ts0),
                    "model_base": ms0, "model_alt": ms1,
                    "model_rel_change": rel_change([ms0, ms1]),
                    "model_diff": float(ms1 - ms0),
                },
                "per_category_present_trip": {
                    "truth_base": tc0, "truth_alt": tc1,
                    "truth_rel_change": rel_change([tc0, tc1]),
                    "model_base": mc0, "model_alt": mc1,
                    "model_rel_change": rel_change([mc0, mc1]),
                },
            }

        own_m = _group_metrics(tr["own_incidence"], md["own_incidence"], valid=True)
        sub_m = _group_metrics(tr["same_subcategory_incidence"], md["same_subcategory_incidence"], valid=True)
        rest_m = _group_metrics(tr["rest_of_category_incidence"], md["rest_of_category_incidence"], valid=has_rest)

        tr_sib_pair = [
            tr["same_subcategory_incidence"][0] + tr["rest_of_category_incidence"][0],
            tr["same_subcategory_incidence"][1] + tr["rest_of_category_incidence"][1],
        ]
        md_sib_pair = [
            md["same_subcategory_incidence"][0] + md["rest_of_category_incidence"][0],
            md["same_subcategory_incidence"][1] + md["rest_of_category_incidence"][1],
        ]
        all_other_m = _group_metrics(tr_sib_pair, md_sib_pair, valid=True)

        if (prod, mult) in saved_eval_map:
            sv = saved_eval_map[(prod, mult)]
            for k_chk, m_obj in (("own", own_m), ("same_subcategory", sub_m), ("rest_of_category", rest_m)):
                if sv[k_chk]["model"] is not None and m_obj["incidence"]["model_rel_change"] is not None:
                    max_repro_diff = max(
                        max_repro_diff,
                        abs(float(sv[k_chk]["model"]) - float(m_obj["incidence"]["model_rel_change"])),
                    )

        scenarios_out.append({
            "product": prod,
            "model_item_id": md["model_item_id"],
            "category": tr["category"],
            "subcategory": tr["subcategory"],
            "price_multiplier": mult,
            "direction": "cut" if mult < 1.0 else "rise",
            "category_incidence": {
                "truth_base": tr["category_incidence"][0],
                "truth_alt": tr["category_incidence"][1],
                "truth_rel_change": tr_cat_inc_rel,
                "truth_abs_change": float(tr["category_incidence"][1] - tr["category_incidence"][0]),
                "model_base": md["category_incidence"][0],
                "model_alt": md["category_incidence"][1],
                "model_rel_change": md_cat_inc_rel,
                "model_abs_change": float(md["category_incidence"][1] - md["category_incidence"][0]),
            },
            "category_total_purchases": {
                "truth_base": tr["category_total_purchases"][0],
                "truth_alt": tr["category_total_purchases"][1],
                "truth_rel_change": tr_cat_tot_rel,
                "model_base": md["category_total_purchases"][0],
                "model_alt": md["category_total_purchases"][1],
                "model_rel_change": md_cat_tot_rel,
            },
            "own": own_m,
            "same_subcategory": sub_m,
            "rest_of_category": rest_m,
            "all_other_in_category": all_other_m,
            "basket_size_change": {
                "truth": float(tr["mean_basket_size"][1] - tr["mean_basket_size"][0]),
                "model": float(md["mean_basket_size"][1] - md["mean_basket_size"][0]),
            },
        })

    def _agg_subset(sub_rows):
        cat_tr = np.array([r["category_incidence"]["truth_rel_change"] for r in sub_rows])
        cat_md = np.array([r["category_incidence"]["model_rel_change"] for r in sub_rows])
        ctot_tr = np.array([r["category_total_purchases"]["truth_rel_change"] for r in sub_rows])
        ctot_md = np.array([r["category_total_purchases"]["model_rel_change"] for r in sub_rows])

        def _stat_pair(t_arr, m_arr):
            corr = float(np.corrcoef(t_arr, m_arr)[0, 1]) if len(t_arr) > 1 and np.std(t_arr) > 0 and np.std(m_arr) > 0 else None
            return {
                "scenarios": int(len(t_arr)),
                "mean_truth": float(t_arr.mean()),
                "mean_model": float(m_arr.mean()),
                "mean_abs_truth": float(np.abs(t_arr).mean()),
                "mean_abs_model": float(np.abs(m_arr).mean()),
                "max_abs_truth": float(np.abs(t_arr).max()),
                "max_abs_model": float(np.abs(m_arr).max()),
                "correlation": corr,
                "mean_abs_error": float(np.abs(t_arr - m_arr).mean()),
            }

        out_sub = {
            "category_incidence_rel_change": _stat_pair(cat_tr, cat_md),
            "category_total_purchases_rel_change": _stat_pair(ctot_tr, ctot_md),
        }
        for grp in ("own", "same_subcategory", "rest_of_category", "all_other_in_category"):
            for part_key in ("incidence", "within_category_share", "per_category_present_trip"):
                valid_r = [
                    r for r in sub_rows
                    if r[grp][part_key]["truth_rel_change"] is not None
                    and r[grp][part_key]["model_rel_change"] is not None
                ]
                t_a = np.array([r[grp][part_key]["truth_rel_change"] for r in valid_r])
                m_a = np.array([r[grp][part_key]["model_rel_change"] for r in valid_r])
                out_sub[f"{grp}_{part_key}_rel_change"] = _stat_pair(t_a, m_a)
                if part_key == "within_category_share":
                    td_a = np.array([r[grp][part_key]["truth_diff"] for r in valid_r])
                    md_a = np.array([r[grp][part_key]["model_diff"] for r in valid_r])
                    out_sub[f"{grp}_within_category_share_diff_pp"] = _stat_pair(td_a, md_a)
        return out_sub

    cuts = [r for r in scenarios_out if r["direction"] == "cut"]
    rises = [r for r in scenarios_out if r["direction"] == "rise"]

    summary_all = _agg_subset(scenarios_out)
    summary_cuts = _agg_subset(cuts)
    summary_rises = _agg_subset(rises)

    # Check the claim: truth's category incidence barely moves, while the model's moves a lot
    truth_mean_abs_cat_inc = summary_all["category_incidence_rel_change"]["mean_abs_truth"]
    model_mean_abs_cat_inc = summary_all["category_incidence_rel_change"]["mean_abs_model"]
    claim_verified = bool(truth_mean_abs_cat_inc < 0.005 and model_mean_abs_cat_inc > 5.0 * truth_mean_abs_cat_inc)

    return {
        "checkpoint": checkpoint_label,
        "max_reproduction_diff_vs_eval_json": max_repro_diff,
        "claim_check": {
            "claim": "The truth's category incidence barely moves, while the model's moves a lot.",
            "verified": claim_verified,
            "truth_category_incidence_mean_abs_rel_change": truth_mean_abs_cat_inc,
            "model_category_incidence_mean_abs_rel_change": model_mean_abs_cat_inc,
            "ratio_model_over_truth_mean_abs_change": float(
                model_mean_abs_cat_inc / max(truth_mean_abs_cat_inc, 1e-12)
            ),
            "cuts_16_mean_rel_change": {
                "truth_category_incidence": summary_cuts["category_incidence_rel_change"]["mean_truth"],
                "model_category_incidence": summary_cuts["category_incidence_rel_change"]["mean_model"],
                "truth_own_share": summary_cuts["own_within_category_share_rel_change"]["mean_truth"],
                "model_own_share": summary_cuts["own_within_category_share_rel_change"]["mean_model"],
                "truth_same_subcategory_share": summary_cuts["same_subcategory_within_category_share_rel_change"]["mean_truth"],
                "model_same_subcategory_share": summary_cuts["same_subcategory_within_category_share_rel_change"]["mean_model"],
                "truth_rest_of_category_share": summary_cuts["rest_of_category_within_category_share_rel_change"]["mean_truth"],
                "model_rest_of_category_share": summary_cuts["rest_of_category_within_category_share_rel_change"]["mean_model"],
            },
            "rises_16_mean_rel_change": {
                "truth_category_incidence": summary_rises["category_incidence_rel_change"]["mean_truth"],
                "model_category_incidence": summary_rises["category_incidence_rel_change"]["mean_model"],
                "truth_own_share": summary_rises["own_within_category_share_rel_change"]["mean_truth"],
                "model_own_share": summary_rises["own_within_category_share_rel_change"]["mean_model"],
                "truth_same_subcategory_share": summary_rises["same_subcategory_within_category_share_rel_change"]["mean_truth"],
                "model_same_subcategory_share": summary_rises["same_subcategory_within_category_share_rel_change"]["mean_model"],
                "truth_rest_of_category_share": summary_rises["rest_of_category_within_category_share_rel_change"]["mean_truth"],
                "model_rest_of_category_share": summary_rises["rest_of_category_within_category_share_rel_change"]["mean_model"],
            },
        },
        "summary_all_32_scenarios": summary_all,
        "summary_16_price_cuts": summary_cuts,
        "summary_16_price_rises": summary_rises,
        "scenarios": scenarios_out,
    }


def main():
    A = ROOT.parent / "energy_basket_intelligence/artifacts/stress_clean_confirm"
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", type=Path, default=ROOT / "runs/stress/hs_rec.pt")
    p.add_argument("--eval-json", type=Path, default=ROOT / "runs/stress/hs_rec.eval.json")
    p.add_argument("--world", type=Path, default=ROOT / "data/stress_world")
    p.add_argument("--initialization", type=Path, default=A / "artifacts/initialization.pt")
    p.add_argument("--additive", type=Path, default=A / "out/v3_pipeline_additive_best.pt")
    p.add_argument("--output", type=Path, default=ROOT / "runs/stress/substitution_diagnosis.json")
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()

    t0 = time.time()
    os.environ["ENERGY_MODEL_DATA_ROOT"] = str(args.world / "model_input_category")
    truth_sc = compute_exact_truth_scenarios(args.world)
    print(f"[diagnose] exact CRN truth scenarios computed in {time.time() - t0:.1f}s", flush=True)

    t1 = time.time()
    model_sc = evaluate_model_substitution(
        args.checkpoint, args.world, args.initialization, args.additive, threads=args.threads
    )
    print(f"[diagnose] model scenarios computed in {time.time() - t1:.1f}s", flush=True)

    report = build_diagnosis_report(
        truth_sc, model_sc, str(args.checkpoint.relative_to(ROOT) if args.checkpoint.is_relative_to(ROOT) else args.checkpoint),
        eval_json_path=args.eval_json,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["claim_check"], indent=2), flush=True)
    if not report["claim_check"]["verified"]:
        raise RuntimeError("Claim check failed: truth category incidence vs model category incidence claim is false.")


if __name__ == "__main__":
    main()
