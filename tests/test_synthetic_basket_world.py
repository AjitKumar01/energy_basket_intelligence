"""The synthetic truth's sampler and oracle normalizer against brute-force enumeration."""
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "synthetic"))
from basket_world import World, energy, gibbs_sample, log_normalizer  # noqa: E402


def tiny_world(seed=3):
    rng = np.random.default_rng(seed)
    J, nmax = 7, 4
    return World(lam=rng.normal(-0.5, 0.6, J), alpha=rng.normal(0, 0.5, (J, 2)),
                 phi=rng.normal(0, 0.45, (J, 2)), category=np.array([0, 0, 0, 1, 1, 2, 2]),
                 rho_c=np.array([0.7, 0.3, 1.2]), rho0=np.array([0, 0, 0.3, 0.9, 1.8]),
                 price_sensitivity=np.array([1.0, 1.5, 0.5])), J, nmax


def exact_law(world, b, J, nmax):
    baskets = [s for n in range(1, nmax + 1) for s in itertools.combinations(range(J), n)]
    membership = np.zeros((len(baskets), J), dtype=bool)
    for row, basket in enumerate(baskets):
        membership[row, list(basket)] = True
    valid = ~(membership & ~np.isfinite(b[0])[None, :]).any(1)
    membership = membership[valid]
    e = energy(world, np.repeat(b, len(membership), 0), membership)
    return membership, np.exp(e - np.logaddexp.reduce(e))


def test_quadrature_normalizer_matches_enumeration():
    world, J, nmax = tiny_world()
    b = np.random.default_rng(1).normal(-0.3, 0.8, (1, J))
    b[0, 4] = -np.inf  # unstocked
    membership, _ = exact_law(world, b, J, nmax)
    e = energy(world, np.repeat(b, len(membership), 0), membership)
    assert abs(float(log_normalizer(world, b, nodes=30)[0]) - np.logaddexp.reduce(e)) < 1e-8


def test_gibbs_sampler_matches_exact_basket_law():
    world, J, nmax = tiny_world()
    b = np.random.default_rng(2).normal(-0.3, 0.8, (1, J))
    b[0, 4] = -np.inf
    membership, probability = exact_law(world, b, J, nmax)
    draws = gibbs_sample(world, np.repeat(b, 60000, 0), np.random.default_rng(5), sweeps=12)
    assert not draws[:, 4].any()
    key = {tuple(row): i for i, row in enumerate(membership)}
    counts = np.bincount([key[tuple(row)] for row in draws], minlength=len(membership))
    total_variation = 0.5 * np.abs(counts / counts.sum() - probability).sum()
    assert total_variation < 0.03, total_variation
