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
