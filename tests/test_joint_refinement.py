"""Exactness checks for the joint refinement stage on an enumerable toy model with Phi != 0."""
import itertools
import math

import numpy as np
import torch

from joint_refinement import (SlotBaskets, baskets_from_items, basket_energy, damped_round,
                              draw_bank, orthonormal_basis, slot_table)
from ragged import RaggedIndex, RaggedModel

torch.set_default_dtype(torch.float64)
J, NMAX = 6, 4
CATEGORY = np.asarray([0, 0, 0, 1, 1, 1])
BASKETS = [b for n in range(1, NMAX + 1) for b in itertools.combinations(range(J), n)]


def toy_model():
    model = RaggedModel(J=J, N=1, C=2, K=2, Kz=2, nmax=NMAX, R=NMAX).double()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.lam.copy_(torch.tensor([-0.8, -0.4, -0.1, 0.1, 0.3, 0.6]))
        model.rho_0_free.copy_(torch.tensor([0.0, 0.2, 0.6, 1.1]))
        model.rho_c.copy_(torch.tensor([0.15, -0.05]))
        model.phi.copy_(torch.tensor([[0.6, 0.1], [0.5, -0.2], [-0.3, 0.4],
                                      [0.2, 0.5], [-0.4, -0.3], [0.1, 0.6]]))
    model.ctx = None
    return model


def replicated_index(copies: int) -> RaggedIndex:
    """`copies` identical one-trip contexts (all products, two category rows each)."""
    item = np.tile(np.arange(J), copies)
    row_of = np.concatenate([np.repeat([2 * c, 2 * c + 1], 3) for c in range(copies)])
    return RaggedIndex(item, row_of, np.repeat(np.arange(copies), 2), np.tile([0, 1], copies), copies)


def exact_energy(model, basket):
    basket = np.asarray(basket)
    lam = model.lam.detach().numpy()
    phi = model.phi.detach().numpy()
    rho0 = model.rho_0().detach().numpy()
    rhoc = model.rho_c.detach().numpy()
    counts = np.bincount(CATEGORY[basket], minlength=2)
    v = phi[basket].sum(0)
    pair = 0.5 * (v @ v - (phi[basket] ** 2).sum())
    return lam[basket].sum() + pair - np.sum(rhoc * counts * (counts - 1) / 2) - rho0[len(basket)]


def exact_law(model):
    energy = np.asarray([exact_energy(model, b) for b in BASKETS])
    return energy, np.exp(energy - np.logaddexp.reduce(energy))


def test_basket_energy_matches_enumeration_through_phi_and_through_basis(native_dp):
    model = toy_model()
    index = replicated_index(1)
    model.house = torch.zeros(1, dtype=torch.long)
    table = slot_table(index, J)
    baskets = baskets_from_items(table, [0] * len(BASKETS), [list(b) for b in BASKETS])
    exact, _ = exact_law(model)
    with torch.no_grad():
        via_phi = basket_energy(model, index, baskets).numpy()
        U, C = orthonormal_basis(model.phi.detach(), 2)
        via_basis = basket_energy(model, index, baskets, U, C).numpy()
    assert np.abs(via_phi - exact).max() < 1e-12
    assert np.abs(via_basis - exact).max() < 1e-12


def test_blocked_gibbs_bank_draws_the_phi_nonzero_law(native_dp):
    model = toy_model()
    chains = 4000
    ix_rep = replicated_index(chains)
    model.house = torch.zeros(chains, dtype=torch.long)
    base = replicated_index(1)
    bank = draw_bank(model, ix_rep, np.zeros(chains, dtype=int), slot_table(base, J), 3, 25,
                     torch.Generator().manual_seed(4))
    _, probability = exact_law(model)
    lookup = {b: i for i, b in enumerate(BASKETS)}
    members = [[] for _ in range(bank.n)]
    for slot, b in zip(bank.slots.tolist(), bank.basket.tolist()):
        members[b].append(int(base.item[slot]))
    counts = np.bincount([lookup[tuple(sorted(m))] for m in members], minlength=len(BASKETS))
    tv = 0.5 * np.abs(counts / counts.sum() - probability).sum()
    rng = np.random.default_rng(0)
    floor = np.mean([0.5 * np.abs(np.bincount(rng.choice(len(BASKETS), size=bank.n, p=probability),
                                              minlength=len(BASKETS)) / bank.n - probability).sum()
                     for _ in range(20)])
    assert tv < 1.5 * floor + 0.005


def test_fixed_bank_log_ratio_converges_to_exact_normalizer_ratio(native_dp):
    parent = toy_model()
    chains = 3000
    ix_rep = replicated_index(chains)
    parent.house = torch.zeros(chains, dtype=torch.long)
    base = replicated_index(1)
    bank = draw_bank(parent, ix_rep, np.zeros(chains, dtype=int), slot_table(base, J), 4, 25,
                     torch.Generator().manual_seed(9))
    parent.house = torch.zeros(1, dtype=torch.long)
    child = toy_model()
    child.house = torch.zeros(1, dtype=torch.long)
    with torch.no_grad():
        child.lam.add_(torch.tensor([0.2, -0.1, 0.0, 0.15, -0.2, 0.05]))
        child.phi.mul_(1.2)
        child.rho_c.add_(torch.tensor([0.1, 0.05]))
        dE = basket_energy(child, base, bank) - basket_energy(parent, base, bank)
    estimate = float(torch.logsumexp(dE, 0) - math.log(bank.n))
    e_parent, _ = exact_law(parent)
    e_child, _ = exact_law(child)
    exact = float(np.logaddexp.reduce(e_child) - np.logaddexp.reduce(e_parent))
    assert abs(estimate - exact) < 0.02


def test_one_damped_round_raises_the_exact_likelihood(native_dp):
    truth = toy_model()
    with torch.no_grad():
        truth.lam.add_(torch.tensor([0.4, -0.3, 0.2, 0.0, -0.4, 0.3]))
        truth.rho_c.add_(torch.tensor([0.3, 0.2]))
    _, p_truth = exact_law(truth)
    rng = np.random.default_rng(1)
    contexts = 400
    observed_items = [list(BASKETS[i]) for i in rng.choice(len(BASKETS), size=contexts, p=p_truth)]
    model = toy_model()
    for name in ("mu", "delta", "zeta", "xi", "gamma", "beta", "price_kappa", "w_dsp", "w_mlr", "psi", "phi"):
        if hasattr(model, name):
            getattr(model, name).requires_grad_(False)
    ix = replicated_index(contexts)
    table = slot_table(ix, J)
    observed = baskets_from_items(table, list(range(contexts)), observed_items)
    draws = 16
    ix_rep = replicated_index(contexts * draws)
    model.house = torch.zeros(contexts * draws, dtype=torch.long)
    bank = draw_bank(model, ix_rep, np.tile(np.arange(contexts), draws), table, 1, 20,
                     torch.Generator().manual_seed(2))
    model.house = torch.zeros(contexts, dtype=torch.long)

    def exact_loglik(m):
        energy, _ = exact_law(m)
        log_z = np.logaddexp.reduce(energy)
        lookup = {b: i for i, b in enumerate(BASKETS)}
        return np.mean([energy[lookup[tuple(sorted(b))]] - log_z for b in observed_items])

    before = exact_loglik(model)
    U, C = orthonormal_basis(model.phi.detach(), 2)
    ok, record = damped_round(model, ix, observed, bank, draws, U, C, 2, [1, 10, 100, 1000],
                              lambda ess: (float(ess.quantile(0.05)) >= 0.3, float(ess.quantile(0.05))),
                              cycles=2, pool_prod=0.0, cap=10.0, log=lambda *_: None)
    assert ok
    after = exact_loglik(model)
    assert after > before + 0.01
