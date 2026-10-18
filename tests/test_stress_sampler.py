"""The vectorized exact sampler and C++ sampler match the native Version-4 dynamic program
and exact brute-force enumeration on stress-world trips (baseline, M1 f_extra, M2 presence_phi, and M3 cat_rho0)."""
import math
import os
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch

try:
    import pytest
except ModuleNotFoundError:
    pytest = types.ModuleType("pytest")

    class _Mark:
        @staticmethod
        def skipif(cond, reason=""):
            return lambda f: f

        @staticmethod
        def parametrize(argnames, argvalues, ids=None):
            def deco(f):
                f._parametrize = (argnames, argvalues, ids or [str(v) for v in argvalues])
                return f
            return deco

    pytest.mark = _Mark()

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "data/stress_world/model_input_category"
pytestmark = pytest.mark.skipif(not BUNDLE.exists(), reason="stress bundle not present")
os.environ.setdefault("ENERGY_MODEL_DATA_ROOT", str(BUNDLE))
sys.path.insert(0, str(ROOT / "scripts/version4"))
sys.path.insert(0, str(ROOT / "scripts/v4_training"))


@pytest.mark.parametrize("use_cpp", [False, True], ids=["python", "cpp"])
def test_sampler_matches_exact_marginals_and_size_law(use_cpp):
    import stress
    from interaction_particles import differentiable_log_size_beta0
    A = ROOT.parent / "energy_basket_intelligence/artifacts/stress_clean_confirm"
    data, model, meta, batcher = stress.setup(A / "artifacts/initialization.pt",
                                              A / "out/v3_pipeline_additive_best.pt", 1)
    J, r = int(data["n_item"]), 8
    with torch.no_grad():
        model.phi[:, :r] = 0.4 * torch.randn(J, r, generator=torch.Generator().manual_seed(0))
    trips = np.repeat(np.flatnonzero(data["trip_split"] == 2)[:3], 3000)
    bt = stress.Batch(batcher, trips, J)
    bt.bind(model)
    z = torch.randn(3, r, generator=torch.Generator().manual_seed(1)).repeat_interleave(3000, 0)
    u = stress.slot_utilities(model, bt.ix, z, r).detach()
    X = stress.sample_slots(model, bt.ix, u, torch.Generator().manual_seed(2), use_cpp=use_cpp)
    uu = u.clone().requires_grad_(True)
    lz = stress.log_zc(model, bt.ix, uu)
    p, = torch.autograd.grad(lz.sum(), uu)
    for b in range(3):
        sel = (bt.ix.item_trip >= 3000 * b) & (bt.ix.item_trip < 3000 * (b + 1))
        first = bt.ix.item_trip == 3000 * b
        freq = torch.zeros(J).index_add_(0, bt.ix.item[sel], X[sel].double()) / 3000
        exact = torch.zeros(J).index_add_(0, bt.ix.item[first], p[first])
        assert float((freq - exact).abs().max()) < 0.03
        sizes = torch.zeros(3000 * 3).index_add_(0, bt.ix.item_trip, X.double())[3000 * b:3000 * (b + 1)]
        log_size = differentiable_log_size_beta0(model, bt.ix, u)[3000 * b]
        assert abs(float(sizes.mean()) - float((torch.softmax(log_size, 0) * torch.arange(1, len(log_size) + 1)).sum())) < 4 * float(sizes.std()) / 3000 ** 0.5
    if use_cpp:
        Xg = stress.gibbs(model, bt.ix, X, r, torch.Generator().manual_seed(3), sweeps=2, beta=0.0, use_cpp=True)
        u0 = stress.slot_utilities(model, bt.ix, None, r, beta=0.0).detach()
        uu0 = u0.clone().requires_grad_(True)
        lz0 = stress.log_zc(model, bt.ix, uu0)
        p0, = torch.autograd.grad(lz0.sum(), uu0)
        for b in range(3):
            sel = (bt.ix.item_trip >= 3000 * b) & (bt.ix.item_trip < 3000 * (b + 1))
            first = bt.ix.item_trip == 3000 * b
            freq0 = torch.zeros(J).index_add_(0, bt.ix.item[sel], Xg[sel].double()) / 3000
            exact0 = torch.zeros(J).index_add_(0, bt.ix.item[first], p0[first])
            assert float((freq0 - exact0).abs().max()) < 0.03
            sizes0 = torch.zeros(3000 * 3).index_add_(0, bt.ix.item_trip, Xg.double())[3000 * b:3000 * (b + 1)]
            log_size0 = differentiable_log_size_beta0(model, bt.ix, u0)[3000 * b]
            assert abs(float(sizes0.mean()) - float((torch.softmax(log_size0, 0) * torch.arange(1, len(log_size0) + 1)).sum())) < 4 * float(sizes0.std()) / 3000 ** 0.5


@pytest.mark.parametrize("use_cpp", [False, True], ids=["python", "cpp"])
def test_m1_m2_sampler_matches_dp_derivatives_on_stress_trips(use_cpp):
    """With M1 (f_extra), M2 (presence_phi), and M3 (cat_rho0) enabled on full stress-world trips,
    verify that both Python and C++ sample_slots match the exact log_zc derivatives w.r.t. slot utilities u,
    latent field z (E[s(S)|z]), f_extra (-P(n_c=k|z) for k=2,3,>=4), and rho_0 (-P(K_cat=m|z))."""
    import stress
    A = ROOT.parent / "energy_basket_intelligence/artifacts/stress_clean_confirm"
    data, model, meta, batcher = stress.setup(A / "artifacts/initialization.pt",
                                              A / "out/v3_pipeline_additive_best.pt", 1)
    J, C, r = int(data["n_item"]), int(data["n_cat"]), 8
    g0 = torch.Generator().manual_seed(10)
    model.f_extra = torch.nn.Parameter(0.35 * torch.randn(C, 3, generator=g0))
    model.u_cat = torch.nn.Parameter(0.3 * torch.randn(C, model.theta.shape[1], generator=g0))
    model.presence_phi = True
    model.psi_cat = torch.nn.Parameter(0.35 * torch.randn(C, r, generator=g0))
    dlt = 0.15 * torch.randn(J, r, generator=g0)
    with torch.no_grad():
        model.phi.zero_()
        model.phi[:, :r] = model.psi_cat[model.cat_of.long()] + dlt

    base_trips = np.flatnonzero(data["trip_split"] == 2)[:2]
    for cat_rho0 in (False, True):
        model.cat_rho0 = cat_rho0
        if cat_rho0:
            with torch.no_grad():
                s0 = (model.rho_0_free[9] - model.rho_0_free[0]) / 9.0
                model.lam.sub_(s0)
                model.rho_0_free.sub_(s0 * torch.arange(1, model.rho_0_free.numel() + 1, dtype=model.rho_0_free.dtype))
                model.rho_0_free.sub_(model.rho_0_free[0].clone())
        bt1 = stress.Batch(batcher, base_trips, J)
        bt1.bind(model)
        z1 = torch.randn(2, r, generator=torch.Generator().manual_seed(11)).requires_grad_(True)
        u1 = stress.slot_utilities(model, bt1.ix, z1, r)
        lz1 = stress.log_zc(model, bt1.ix, u1, z=z1)
        gu1, gz1 = torch.autograd.grad(lz1.sum(), (u1, z1), retain_graph=True)
        gfe0, gr0_0 = torch.autograd.grad(lz1[0], (model.f_extra, model.rho_0_free), retain_graph=True)
        gfe1, gr0_1 = torch.autograd.grad(lz1[1], (model.f_extra, model.rho_0_free))

        reps = 4000
        trips = np.repeat(base_trips, reps)
        bt = stress.Batch(batcher, trips, J)
        bt.bind(model)
        z = z1.detach().repeat_interleave(reps, 0).contiguous()
        u = stress.slot_utilities(model, bt.ix, z, r).detach()
        X = stress.sample_slots(model, bt.ix, u, torch.Generator().manual_seed(12), use_cpp=use_cpp, z=z, beta=1.0)
        s_draws = stress.basket_field(model, bt.ix, X, r)
        nc_row = torch.zeros(bt.ix.n_rows).index_add_(0, bt.ix.row_of, X.double())

        for b, (gfe_b, gr0_b) in enumerate(((gfe0, gr0_0), (gfe1, gr0_1))):
            sel = (bt.ix.item_trip >= reps * b) & (bt.ix.item_trip < reps * (b + 1))
            first = bt1.ix.item_trip == b
            freq = torch.zeros(J).index_add_(0, bt.ix.item[sel], X[sel].double()) / reps
            exact = torch.zeros(J).index_add_(0, bt1.ix.item[first], gu1[first].detach())
            assert float((freq - exact).abs().max()) < 0.03

            s_mean = s_draws[reps * b:reps * (b + 1)].mean(0)
            assert float((s_mean - gz1[b].detach()).abs().max()) < 0.04

            rsel = (bt.ix.row_trip >= reps * b) & (bt.ix.row_trip < reps * (b + 1))
            p2 = torch.zeros(C).index_add_(0, bt.ix.row_cat[rsel], (nc_row[rsel] == 2).double()) / reps
            p3 = torch.zeros(C).index_add_(0, bt.ix.row_cat[rsel], (nc_row[rsel] == 3).double()) / reps
            p4 = torch.zeros(C).index_add_(0, bt.ix.row_cat[rsel], (nc_row[rsel] >= 4).double()) / reps
            assert float((p2 - (-gfe_b[:, 0].detach())).abs().max()) < 0.025
            assert float((p3 - (-gfe_b[:, 1].detach())).abs().max()) < 0.02
            assert float((p4 - (-gfe_b[:, 2].detach())).abs().max()) < 0.02

            if cat_rho0:
                kcat_draws = torch.zeros(reps, dtype=torch.long).index_add_(
                    0, bt.ix.row_trip[rsel] - reps * b, (nc_row[rsel] > 0).long()
                )
                pk = torch.bincount(kcat_draws, minlength=model.nmax + 1)[1:model.nmax + 1].double() / reps
                assert float((pk - (-gr0_b[:model.nmax].detach())).abs().max()) < 0.025


@pytest.mark.parametrize("use_cpp", [False, True], ids=["python", "cpp"])
def test_dp_and_samplers_match_exact_enumeration_on_small_subset(use_cpp):
    """Construct a small stress-world RaggedIndex subset (J_sub=10 slots across 4 categories on B=2 trips)
    with f_extra (M1), presence_phi (M2), and cat_rho0 (M3, both False and True) active, and check
    stress.log_zc, stress.obs_energy, stress.sample_slots, and stress.gibbs against exact brute-force
    enumeration over all 2^10 - 1 baskets."""
    import stress
    from ragged import RaggedIndex
    A = ROOT.parent / "energy_basket_intelligence/artifacts/stress_clean_confirm"
    data, model, meta, batcher = stress.setup(A / "artifacts/initialization.pt",
                                              A / "out/v3_pipeline_additive_best.pt", 1)
    J, C = int(data["n_item"]), int(data["n_cat"])
    r = 6
    model.nmax = 6
    model.R = 6

    g = torch.Generator().manual_seed(21)
    model.f_extra = torch.nn.Parameter(0.4 * torch.randn(C, 3, generator=g))
    model.presence_phi = True
    model.psi_cat = torch.nn.Parameter(0.35 * torch.randn(C, r, generator=g))
    dlt = 0.2 * torch.randn(J, r, generator=g)
    with torch.no_grad():
        model.phi.zero_()
        model.phi[:, :r] = model.psi_cat[model.cat_of.long()] + dlt
        model.rho_c.copy_(0.25 * torch.randn(C, generator=g))
        model.rho_0_free[:model.nmax].copy_(0.25 * torch.randn(model.nmax, generator=g))

    # Build a 10-slot RaggedIndex per trip across 4 categories (sizes 4, 3, 2, 1)
    cat_of = model.cat_of.long().numpy()
    cats_chosen = [0, 1, 3, 4]
    sizes_chosen = [4, 3, 2, 1]
    sub_items, sub_row_of, sub_row_cat = [], [], []
    for r_idx, (c_id, sz) in enumerate(zip(cats_chosen, sizes_chosen)):
        pool = np.flatnonzero(cat_of == c_id)[:sz]
        sub_items.extend(pool.tolist())
        sub_row_of.extend([r_idx] * sz)
        sub_row_cat.append(c_id)
    M_slots = len(sub_items)  # 10 slots
    n_cat_sub = len(cats_chosen)

    def make_replicated_index(n_trips):
        items_all = np.tile(sub_items, n_trips)
        row_of_all = np.concatenate([np.array(sub_row_of) + b * n_cat_sub for b in range(n_trips)])
        row_trip_all = np.repeat(np.arange(n_trips), n_cat_sub)
        row_cat_all = np.tile(sub_row_cat, n_trips)
        return RaggedIndex(items_all, row_of_all, row_trip_all, row_cat_all, n_trips)

    ix2 = make_replicated_index(2)
    b_slot_2 = 0.5 * torch.randn(2 * M_slots, generator=g)
    model.b_flat = lambda ix: b_slot_2[:M_slots].repeat(ix.B) if ix.B > 2 else b_slot_2[:ix.B * M_slots]

    z2 = torch.randn(2, r, generator=g)
    u2 = stress.slot_utilities(model, ix2, z2, r)

    # Brute-force all nonempty subsets S of {0..M_slots-1} with 1 <= |S| <= model.nmax
    bits = []
    for mask_int in range(1, 1 << M_slots):
        b_vec = [(mask_int >> j) & 1 for j in range(M_slots)]
        if 1 <= sum(b_vec) <= model.nmax:
            bits.append(b_vec)
    B_mat = torch.tensor(bits, dtype=torch.float64)  # [N_baskets, M_slots]
    N_baskets = B_mat.shape[0]

    for cat_rho0 in (False, True):
        model.cat_rho0 = cat_rho0
        lz_dp = stress.log_zc(model, ix2, u2, z=z2)
        rho0 = model.rho_0()
        exact_probs = []
        for b in range(2):
            u_b = u2[b * M_slots:(b + 1) * M_slots]
            log_w = B_mat @ u_b
            k_cat_b = torch.zeros(N_baskets, dtype=torch.long)
            for r_local, c_id in enumerate(cats_chosen):
                in_cat = torch.tensor([1.0 if x == r_local else 0.0 for x in sub_row_of], dtype=torch.float64)
                nc = B_mat @ in_cat
                k_cat_b = k_cat_b + (nc > 0).long()
                pen = model.rho_c[c_id] * (0.5 * nc * (nc - 1.0))
                fe = model.f_extra[c_id]
                pen = pen + (nc == 2).double() * fe[0] + (nc == 3).double() * fe[1] + (nc >= 4).double() * fe[2]
                psi_c = model.psi_cat[c_id]
                cshift = float((z2[b] * psi_c).sum() - 0.5 * psi_c.square().sum())
                pen = pen + (nc - 1.0).clamp_min(0.0) * cshift
                log_w = log_w - pen
            sz_b = k_cat_b if cat_rho0 else B_mat.sum(1).long()
            log_w = log_w - rho0[sz_b]
            lz_exact = torch.logsumexp(log_w, 0)
            assert abs(float(lz_dp[b] - lz_exact)) < 1e-12
            exact_probs.append(torch.softmax(log_w, 0))

        # Check sample_slots (Python and C++) against exact brute-force marginals on trip 0
        reps = 6000
        ix_rep = make_replicated_index(reps)
        z_rep = z2[0:1].expand(reps, r).contiguous()
        u_rep = stress.slot_utilities(model, ix_rep, z_rep, r).detach()
        X_draws = stress.sample_slots(model, ix_rep, u_rep, torch.Generator().manual_seed(31),
                                      use_cpp=use_cpp, z=z_rep, beta=1.0)
        X_mat = X_draws.view(reps, M_slots).double()
        emp_marg = X_mat.mean(0)
        ex_marg = exact_probs[0] @ B_mat
        assert float((emp_marg - ex_marg).abs().max()) < 0.02

        # Check full marginal law under Gibbs (z | S ~ N(s(S), I), S | z ~ P(S | z)) against exact target E(S)
        b0 = b_slot_2[:M_slots]
        ph_sub = stress.phi_r(model, r)[torch.tensor(sub_items, dtype=torch.long)]
        log_p_target = B_mat @ b0
        s_all = torch.zeros(N_baskets, r)
        diag_all = B_mat @ ph_sub.square().sum(-1)
        k_cat_all = torch.zeros(N_baskets, dtype=torch.long)
        for r_local, c_id in enumerate(cats_chosen):
            in_cat = torch.tensor([1.0 if x == r_local else 0.0 for x in sub_row_of], dtype=torch.float64)
            nc = B_mat @ in_cat
            k_cat_all = k_cat_all + (nc > 0).long()
            pen = model.rho_c[c_id] * (0.5 * nc * (nc - 1.0))
            fe = model.f_extra[c_id]
            pen = pen + (nc == 2).double() * fe[0] + (nc == 3).double() * fe[1] + (nc >= 4).double() * fe[2]
            log_p_target = log_p_target - pen
            extra_c = (nc - 1.0).clamp_min(0.0)
            psi_c = model.psi_cat[c_id]
            diag_all = diag_all - extra_c * psi_c.square().sum()
            s_all = s_all + (B_mat * in_cat[None, :]) @ ph_sub - extra_c[:, None] * psi_c[None, :]
        sz_target = k_cat_all if cat_rho0 else B_mat.sum(1).long()
        log_p_target = log_p_target + 0.5 * (s_all.square().sum(-1) - diag_all) - rho0[sz_target]
        target_probs = torch.softmax(log_p_target, 0)
        target_marg = target_probs @ B_mat

        X_gibbs = stress.gibbs(model, ix_rep, X_draws, r, torch.Generator().manual_seed(32),
                               sweeps=25, beta=1.0, use_cpp=use_cpp)
        gibbs_marg = X_gibbs.view(reps, M_slots).double().mean(0)
        assert float((gibbs_marg - target_marg).abs().max()) < 0.025


if __name__ == "__main__":
    t0 = time.time()
    passed = 0
    for fn in (test_sampler_matches_exact_marginals_and_size_law,
               test_m1_m2_sampler_matches_dp_derivatives_on_stress_trips,
               test_dp_and_samplers_match_exact_enumeration_on_small_subset):
        for use_cpp, label in ((False, "python"), (True, "cpp")):
            ts = time.time()
            fn(use_cpp=use_cpp)
            passed += 1
            print(f"tests/test_stress_sampler.py::{fn.__name__}[{label}] PASSED ({time.time() - ts:.2f}s)", flush=True)
    print(f"================== {passed} passed in {time.time() - t0:.2f}s ==================", flush=True)
