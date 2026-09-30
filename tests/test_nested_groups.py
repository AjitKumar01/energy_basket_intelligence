"""Nested substitution groups: the exact program against brute-force enumeration."""
import itertools
import math

import numpy as np
import torch

from ragged import RaggedIndex, RaggedModel, log_f_nested, log_f_ragged, smolyak_grid

torch.set_default_dtype(torch.float64)
J, NMAX = 8, 5
LEAF = np.asarray([0, 0, 1, 1, 2, 2, 3, 3])          # 4 leaf groups (e.g. subcategories)
PARENT = [0, 0, 1, 1]                               # 2 parents (e.g. categories)
BASKETS = [b for n in range(1, NMAX + 1) for b in itertools.combinations(range(J), n)]


def nested_model(group_parent=PARENT, rho_c=(0.3, 0.1, 0.5, -0.2), rho_p=(0.4, 0.7), seed=0):
    model = RaggedModel(J=J, N=1, C=4, K=2, Kz=2, nmax=NMAX, R=NMAX, seed=seed,
                        group_parent=group_parent).double()
    g = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.lam.copy_(torch.linspace(-0.9, 0.4, J))
        model.rho_0_free.copy_(torch.tensor([0.0, 0.3, 0.7, 1.2, 1.8]))
        model.rho_c.copy_(torch.tensor(rho_c))
        if model.P:
            model.rho_p.copy_(torch.tensor(rho_p))
        model.phi.copy_(0.5 * torch.randn(J, 2, generator=g))
    model.cat_of.copy_(torch.as_tensor(LEAF))
    model.ctx = None
    model.house = torch.zeros(1, dtype=torch.long)
    model._esp_native = True
    model._poly_degree_native = True
    return model


def one_trip_index():
    return RaggedIndex(np.arange(J), LEAF, np.zeros(4, dtype=int), np.arange(4), 1)


def exact_terms(model, z=None):
    """Enumerated energies: E(S) (z is None) or the z-conditional weight log of f(z)."""
    lam, phi = model.lam.detach().numpy(), model.phi.detach().numpy()
    rho0, rhoc = model.rho_0().detach().numpy(), model.rho_c.detach().numpy()
    rhop = model.rho_p.detach().numpy() if model.P else None
    out = []
    for b in BASKETS:
        b = np.asarray(b)
        leaf = np.bincount(LEAF[b], minlength=4)
        e = lam[b].sum() - np.sum(rhoc * leaf * (leaf - 1) / 2) - rho0[len(b)]
        if model.P:
            par = np.bincount(np.asarray(PARENT)[LEAF[b]], minlength=model.P)
            e -= np.sum(rhop * par * (par - 1) / 2)
        if z is None:
            v = phi[b].sum(0)
            e += 0.5 * (v @ v - (phi[b] ** 2).sum())
        else:
            e += (-0.5 * (phi[b] ** 2).sum(1) + phi[b] @ z).sum()
        out.append(e)
    return np.asarray(out)


def test_nested_f_of_z_matches_enumeration(native_dp):
    model = nested_model()
    ix = one_trip_index()
    for z in ([0.0, 0.0], [0.8, -0.5], [-1.2, 0.9]):
        zt = torch.tensor([[z]])
        got = float(log_f_nested(model, zt, ix, drop_empty=True))
        want = np.logaddexp.reduce(exact_terms(model, np.asarray(z)))
        assert abs(got - want) < 1e-10


def test_nested_normaliser_and_energy_give_a_probability_law(native_dp):
    model = nested_model()
    ix = one_trip_index()
    model.quad = smolyak_grid(2, 9)
    log_z = float(model.log_Z(ix, drop_empty=True))
    exact = np.logaddexp.reduce(exact_terms(model))
    assert abs(log_z - exact) < 1e-3                      # quadrature error only
    line_item = torch.tensor([0, 1, 4])
    energy = float(model.energy(line_item, torch.zeros(3, dtype=torch.long),
                                torch.as_tensor(LEAF[[0, 1, 4]]), 1))
    lookup = {b: i for i, b in enumerate(BASKETS)}
    assert abs(energy - exact_terms(model)[lookup[(0, 1, 4)]]) < 1e-12


def test_zero_parent_penalty_is_the_flat_model(native_dp):
    nested = nested_model(rho_p=(0.0, 0.0))
    flat = nested_model(group_parent=None)
    ix = one_trip_index()
    for z in ([0.3, 0.2], [-0.7, 1.1]):
        zt = torch.tensor([[z]])
        a = float(log_f_nested(nested, zt, ix, drop_empty=True))
        b = float(log_f_ragged(flat, zt, ix, drop_empty=True))
        assert abs(a - b) < 1e-10


def test_parent_penalty_alone_is_a_flat_model_on_parents(native_dp):
    nested = nested_model(rho_c=(0.0, 0.0, 0.0, 0.0), rho_p=(0.6, 0.25))
    flat = RaggedModel(J=J, N=1, C=2, K=2, Kz=2, nmax=NMAX, R=NMAX).double()
    flat.load_state_dict({k: v for k, v in nested.state_dict().items()
                          if k not in ("rho_c", "rho_p", "group_parent", "cat_of")}, strict=False)
    with torch.no_grad():
        flat.rho_c.copy_(nested.rho_p)
    flat.ctx, flat.house = None, nested.house
    parent_of_item = np.asarray(PARENT)[LEAF]
    ix_flat = RaggedIndex(np.arange(J), parent_of_item, np.zeros(2, dtype=int), np.arange(2), 1)
    zt = torch.tensor([[[0.4, -0.3]]])
    a = float(log_f_nested(nested, zt, one_trip_index(), drop_empty=True))
    b = float(log_f_ragged(flat, zt, ix_flat, drop_empty=True))
    assert abs(a - b) < 1e-10


def test_nested_conditional_law_matches_enumeration(native_dp):
    """Revealed basket R = {0}: at fixed z, f(z) of the remainder law is the enumerated sum over
    remainders T of exp(sum_{j in T} w_j(z) - [pen(R u T) - pen(R)] - [rho_0(|R|+|T|) - rho_0(|R|)]),
    with the revealed product's interaction phi_0'phi_j entering every w_j (as fit.py does)."""
    model = nested_model()
    z = np.asarray([0.5, -0.4])
    lam, phi = model.lam.detach().numpy(), model.phi.detach().numpy()
    rho0, rhoc, rhop = (model.rho_0().detach().numpy(), model.rho_c.detach().numpy(),
                        model.rho_p.detach().numpy())
    parent = np.asarray(PARENT)

    def penalty(items):
        leaf = np.bincount(LEAF[items], minlength=4)
        par = np.bincount(parent[LEAF[items]], minlength=2)
        return np.sum(rhoc * leaf * (leaf - 1) / 2) + np.sum(rhop * par * (par - 1) / 2)

    others = np.arange(1, J)
    w = lam + phi @ phi[0] - 0.5 * (phi ** 2).sum(1) + phi @ z
    terms = []
    for n in range(0, NMAX):                                     # |R| + |T| <= NMAX
        for T in itertools.combinations(others, n):
            T = np.asarray(T, dtype=int)
            RT = np.concatenate([[0], T]).astype(int)
            terms.append(w[T].sum() - (penalty(RT) - penalty(np.asarray([0])))
                         - (rho0[1 + n] - rho0[1]))
    want = np.logaddexp.reduce(np.asarray(terms))
    keep = np.arange(J) != 0
    ix = RaggedIndex(np.arange(J)[keep], LEAF[keep], np.zeros(4, dtype=int), np.arange(4), 1)
    model._condition_cat_count = torch.as_tensor(np.bincount(LEAF[[0]], minlength=4)).unsqueeze(0)
    model._condition_size = torch.tensor([1])
    model._b_override = (model.b_flat(ix) + model.phi.detach()[ix.item] @ model.phi.detach()[0]).detach()
    try:
        got = float(log_f_nested(model, torch.tensor([[z]]), ix, drop_empty=False))
    finally:
        model._condition_cat_count = model._condition_size = model._b_override = None
    assert abs(got - want) < 1e-10


def test_parent_penalty_gradient_matches_finite_differences(native_dp):
    model = nested_model()
    ix = one_trip_index()
    zt = torch.tensor([[[0.2, 0.6]]])
    value = log_f_nested(model, zt, ix, drop_empty=True).sum()
    grad, = torch.autograd.grad(value, model.rho_p)
    eps = 1e-6
    for p in range(model.P):
        with torch.no_grad():
            model.rho_p[p] += eps
        up = float(log_f_nested(model, zt, ix, drop_empty=True))
        with torch.no_grad():
            model.rho_p[p] -= 2 * eps
        down = float(log_f_nested(model, zt, ix, drop_empty=True))
        with torch.no_grad():
            model.rho_p[p] += eps
        assert abs(float(grad[p]) - (up - down) / (2 * eps)) < 1e-6


def test_no_interaction_size_law_matches_enumeration(native_dp):
    """differentiable_log_size_beta0 (the additive stage's exact normaliser) on a nested model."""
    from interaction_particles import differentiable_log_size_beta0
    model = nested_model()
    with torch.no_grad():
        model.phi.zero_()
    ix = one_trip_index()
    got = differentiable_log_size_beta0(model, ix)[0].detach().numpy()          # sizes 1..NMAX
    terms = exact_terms(model)
    sizes = np.asarray([len(b) for b in BASKETS])
    want = np.asarray([np.logaddexp.reduce(terms[sizes == n]) for n in range(1, NMAX + 1)])
    assert np.allclose(got, want, atol=1e-10)


def test_add_one_log_odds_include_the_parent_term(native_dp):
    """E(S + j) - E(S) = b_j + phi_j'sum phi_S - rho_c[leaf]*rest_leaf - rho_p[parent]*rest_parent
    - (rho_0(|S|+1) - rho_0(|S|)), for every basket S and product j outside it."""
    from interaction_particles import rest_parent_penalty
    model = nested_model()
    phi, lam = model.phi.detach(), model.lam.detach()
    rho0, rhoc = model.rho_0().detach(), model.rho_c.detach()
    slot_cat = torch.as_tensor(LEAF)

    def energy(items):
        items = torch.as_tensor(items)
        return float(model.energy(items, torch.zeros(len(items), dtype=torch.long),
                                  torch.as_tensor(LEAF)[items], 1))

    for S in [(0,), (0, 2), (1, 4, 6), (2, 3)]:
        old = torch.zeros(J)
        old[list(S)] = 1.0
        parent_pen = rest_parent_penalty(model, torch.zeros(J, dtype=torch.long), slot_cat, old, 1)
        leaf_counts = torch.bincount(slot_cat[list(S)], minlength=4).double()
        for j in set(range(J)) - set(S):
            logit = (lam[j] + phi[j] @ phi[list(S)].sum(0) - rhoc[LEAF[j]] * leaf_counts[LEAF[j]]
                     - parent_pen[j] - (rho0[len(S) + 1] - rho0[len(S)]))
            assert abs(float(logit) - (energy(list(S) + [j]) - energy(list(S)))) < 1e-12


def test_refinement_energy_and_design_include_the_parent_penalty(native_dp):
    """basket_energy and the sufficient-statistic design_energy equal the enumerated energy."""
    from joint_refinement import (bank_design, basket_energy, baskets_from_items, design_energy,
                                  orthonormal_basis, slot_table, taste_utility)
    model = nested_model()
    ix = one_trip_index()
    table = slot_table(ix, J)
    items = [list(b) for b in BASKETS[::7]]
    baskets = baskets_from_items(table, [0] * len(items), items)
    lookup = {b: i for i, b in enumerate(BASKETS)}
    want = exact_terms(model)[[lookup[tuple(b)] for b in items]]
    with torch.no_grad():
        got = basket_energy(model, ix, baskets).numpy()
        U, C = orthonormal_basis(model.phi, 2)
        frozen = model.b_flat(ix) - taste_utility(model, ix)
        design = bank_design(model, ix, baskets, U, frozen)
        via_design = design_energy(model, design, C).numpy()
    assert np.allclose(got, want, atol=1e-10)
    assert np.allclose(via_design, want, atol=1e-10)


def _tv_against(law, members, draws, rng):
    lookup = {b: i for i, b in enumerate(BASKETS)}
    counts = np.bincount([lookup[tuple(sorted(m))] for m in members], minlength=len(BASKETS))
    tv = 0.5 * np.abs(counts / counts.sum() - law).sum()
    floor = np.mean([0.5 * np.abs(np.bincount(rng.choice(len(BASKETS), size=draws, p=law),
                                              minlength=len(BASKETS)) / draws - law).sum()
                     for _ in range(20)])
    return tv, floor


def test_nested_conditional_sampler_draws_the_exact_law(native_dp):
    from tempered_block_gibbs import conditional_slots_repeated
    model = nested_model()
    ix = one_trip_index()
    z = np.asarray([0.6, -0.3])
    draws = 40000
    states = conditional_slots_repeated(model, ix, torch.tensor([z]), 1.0, draws,
                                        torch.Generator().manual_seed(7))
    members = [[int(ix.item[s]) for s in states[d][0]] for d in range(draws)]
    terms = exact_terms(model, z)
    law = np.exp(terms - np.logaddexp.reduce(terms))
    tv, floor = _tv_against(law, members, draws, np.random.default_rng(8))
    assert tv < 1.5 * floor + 0.005


def test_nested_fixed_size_sampler_draws_the_size_conditional_law(native_dp):
    from tempered_block_gibbs import conditional_slots_fixed_sizes
    model = nested_model()
    ix = one_trip_index()
    z = np.asarray([-0.4, 0.5])
    draws = 30000
    states = conditional_slots_fixed_sizes(model, ix, torch.tensor([z]), 1.0,
                                           np.full((draws, 1), 3), torch.Generator().manual_seed(9))
    members = [[int(ix.item[s]) for s in states[d][0]] for d in range(draws)]
    assert all(len(m) == 3 for m in members)
    terms = exact_terms(model, z)
    sizes = np.asarray([len(b) for b in BASKETS])
    terms = np.where(sizes == 3, terms, -np.inf)
    law = np.exp(terms - np.logaddexp.reduce(terms))
    tv, floor = _tv_against(law, members, draws, np.random.default_rng(10))
    assert tv < 1.5 * floor + 0.005


def test_nested_compiled_entry_point_draws_the_exact_law(native_dp):
    from compiled_backtrack import compiled_draws
    from tempered_block_gibbs import conditional_log_tables_levels
    model = nested_model()
    ix = one_trip_index()
    z = np.asarray([0.2, 0.9])
    tables = conditional_log_tables_levels(model, ix, torch.tensor([[z]]), [1.0])
    draws = 30000
    slots, basket = compiled_draws(*tables, ix, draws, torch.Generator().manual_seed(11))
    members = [[] for _ in range(draws)]
    for s, b in zip(slots.tolist(), basket.tolist()):
        members[b].append(int(ix.item[s]))
    terms = exact_terms(model, z)
    law = np.exp(terms - np.logaddexp.reduce(terms))
    tv, floor = _tv_against(law, members, draws, np.random.default_rng(12))
    assert tv < 1.5 * floor + 0.005


def test_nested_gibbs_bank_draws_the_joint_law(native_dp):
    """Blocked Gibbs on (S, z) with the nested S | z sampler targets the joint law with phi."""
    from joint_refinement import draw_bank, slot_table
    model = nested_model()
    chains = 3000
    item = np.tile(np.arange(J), chains)
    row_of = np.concatenate([LEAF + 4 * c for c in range(chains)])
    ix_rep = RaggedIndex(item, row_of, np.repeat(np.arange(chains), 4), np.tile(np.arange(4), chains), chains)
    model.house = torch.zeros(chains, dtype=torch.long)
    bank = draw_bank(model, ix_rep, np.zeros(chains, dtype=int), slot_table(one_trip_index(), J), 3, 10,
                     torch.Generator().manual_seed(13))
    members = [[] for _ in range(bank.n)]
    for s, b in zip(bank.slots.tolist(), bank.basket.tolist()):
        members[b].append(int(s))                                    # base-table slot == product
    terms = exact_terms(model)
    law = np.exp(terms - np.logaddexp.reduce(terms))
    tv, floor = _tv_against(law, members, bank.n, np.random.default_rng(14))
    assert tv < 1.5 * floor + 0.01


def test_embedding_audit_pair_penalty_includes_the_parent(native_dp):
    from audit_interaction_embeddings import group_pair_penalty
    group = np.asarray([0, 0, 1, 2])
    rho, parent, rho_parent = np.asarray([0.3, 0.5, 0.2]), np.asarray([0, 0, 1]), np.asarray([0.4, 0.9])
    assert group_pair_penalty(0, 1, group, rho, parent, rho_parent) == -(0.3 + 0.4)   # same leaf
    assert group_pair_penalty(0, 2, group, rho, parent, rho_parent) == -0.4           # same parent
    assert group_pair_penalty(0, 3, group, rho, parent, rho_parent) == 0.0            # different
    assert group_pair_penalty(0, 1, group, rho) == -0.3                               # flat model
