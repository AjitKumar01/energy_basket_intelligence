"""Exact checks of the HS training components against enumeration."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/v4_training"))
from hs import V4, gibbs_sweep, is_log_z, log_zc, pseudo_likelihood, sample_given_u  # noqa: E402
from world import Support, energy_table, exact_log_z, make_truth  # noqa: E402

torch.set_default_dtype(torch.float64)


def _setup(strength=3.0):
    truth, price = make_truth(5, strength, H=6, W=2)
    sup = Support(18, 5)
    h = torch.tensor([0, 1, 2, 3])
    pr = price[torch.tensor([0, 1, 0, 1])]
    return truth, sup, h, pr


def _additive_table(truth, sup, u):
    n_c = sup.M @ truth.part.onehot
    rest = -(n_c * (n_c - 1) / 2) @ truth.p["rho_c"] - truth.rho0()[sup.size - 1]
    return u @ sup.M.T + rest[None, :]


def test_conditional_log_z_and_marginals_match_enumeration():
    truth, sup, h, pr = _setup()
    z = torch.randn(4, 3)
    u = truth.utilities(h, pr, z).detach().requires_grad_(True)
    lz = log_zc(u, truth.part, truth.p["rho_c"], truth.rho0())
    E = _additive_table(truth, sup, u.detach())
    assert torch.allclose(lz.detach(), torch.logsumexp(E, 1), atol=1e-10)
    lz.sum().backward()
    assert torch.allclose(u.grad, torch.softmax(E, 1) @ sup.M, atol=1e-10)


def test_conditional_sampler_matches_enumeration():
    truth, sup, h, pr = _setup()
    z = torch.randn(1, 3).repeat(60000, 1)
    u = truth.utilities(h[:1].repeat(60000), pr[:1].repeat(60000, 1), z)
    X = sample_given_u(u, truth.part, truth.p["rho_c"], truth.rho0(), torch.Generator().manual_seed(1))
    P = torch.softmax(_additive_table(truth, sup, u[:1]), 1)[0]
    assert (X.mean(0) - P @ sup.M).abs().max() < 0.01
    sizes = torch.bincount(X.sum(1).long(), minlength=6)[1:] / len(X)
    exact = torch.stack([P[sup.size == k].sum() for k in range(1, 6)])
    assert (sizes - exact).abs().max() < 0.01


def test_hs_identity_and_gibbs_target_the_v4_law():
    truth, sup, h, pr = _setup(strength=3.0)
    exact = exact_log_z(truth, sup, h, pr)
    est, ess = is_log_z(truth, h, pr, torch.Generator().manual_seed(2), burn=100, components=64, draws=2000)
    assert (est - exact).abs().max() < 0.02
    # Gibbs marginals vs the exact basket law
    n = 20000
    hh, pp = h[:1].repeat(n), pr[:1].repeat(n, 1)
    X = sample_given_u(truth.b(hh, pp), truth.part, truth.p["rho_c"], truth.rho0(), torch.Generator().manual_seed(3))
    X, _ = gibbs_sweep(truth, X, hh, pp, torch.Generator().manual_seed(4), sweeps=60)
    P = torch.softmax(energy_table(truth, sup, h[:1], pr[:1]), 1)[0]
    assert (X.mean(0) - P @ sup.M).abs().max() < 0.015


def test_pseudo_likelihood_conditionals_match_enumeration():
    truth, sup, h, pr = _setup()
    rng = np.random.default_rng(0)
    rows = rng.choice(sup.M.shape[0], size=40)
    X = sup.M[torch.as_tensor(rows)]
    hh, pp = h[:1].repeat(40), pr[:1].repeat(40, 1)
    got = pseudo_likelihood(truth, X, hh, pp)
    E = energy_table(truth, sup, h[:1], pr[:1])[0]
    total = 0.0
    for r, row in enumerate(rows):
        x = sup.M[row]
        for j in range(18):
            y = x.clone(); y[j] = 1 - y[j]
            n = int(y.sum())
            if n == 0 or n > 5:
                continue
            key = tuple(np.flatnonzero(y.numpy()))
            other = sup.row[key]
            total += float(E[row] - torch.logaddexp(E[row], E[other]))
    assert abs(float(got) - total) < 1e-8
