"""Known-truth energy basket world, independent of the pipeline implementation.

For a trip x (household h, store s, week t) and a nonempty basket S,

    E(S, x) = sum_{j in S} b_j(x) + sum_{j<k in S} phi_j . phi_k
              - sum_c rho_c C(n_c, 2) - rho0(|S|),
    b_j(x)  = lam_j + kappa_h + theta_h . alpha_j - g_{c(j)} dlp_{jt}   if j stocked at (s, t)
    p(S|x)  = exp E(S, x) / Z_+(x),     1 <= |S| <= nmax.

Unstocked products have weight zero. With the Hubbard-Stratonovich identity
exp(sum_{j<k} phi_j phi_k) = E_z exp(z . sum_j phi_j - 1/2 sum_j |phi_j|^2), z ~ N(0, I),
the law is the z-marginal of p(S, z) proportional to N(z) prod_{j in S} w_j(z) pen(S), with
w_j(z) = exp(b_j - |phi_j|^2 / 2 + z . phi_j). This gives

* exact conditional samplers: z | S ~ N(sum_{j in S} phi_j, I), and S | z by a
  category/size dynamic program with backward sampling (Gibbs on (S, z)); and
* the oracle normalizer Z_+(x) = E_z[D(z)] by tensor Gauss-Hermite quadrature, where D(z)
  is the same dynamic program's total.
"""
from __future__ import annotations

from dataclasses import dataclass
import itertools

import numpy as np


@dataclass
class World:
    lam: np.ndarray              # [J]
    alpha: np.ndarray            # [J, Kt]
    phi: np.ndarray              # [J, Kz]
    category: np.ndarray         # [J] int
    rho_c: np.ndarray            # [C]
    rho0: np.ndarray             # [nmax + 1], rho0[0] unused
    price_sensitivity: np.ndarray  # [C]

    @property
    def nmax(self) -> int:
        return len(self.rho0) - 1

    @property
    def n_categories(self) -> int:
        return len(self.rho_c)

    def category_items(self) -> list[np.ndarray]:
        return [np.flatnonzero(self.category == c) for c in range(self.n_categories)]


def utilities(world: World, taste: np.ndarray, size_shift: np.ndarray, dlp: np.ndarray,
              available: np.ndarray) -> np.ndarray:
    """b [B, J] with -inf for unstocked products. taste [B, Kt], size_shift [B], dlp [B, J]."""
    b = (world.lam[None, :] + size_shift[:, None] + taste @ world.alpha.T
         - world.price_sensitivity[world.category][None, :] * dlp)
    return np.where(available, b, -np.inf)


def _log_weights(world: World, b: np.ndarray, z: np.ndarray) -> np.ndarray:
    return b - 0.5 * (world.phi ** 2).sum(1)[None, :] + z @ world.phi.T


def _category_tables(world: World, logw: np.ndarray):
    """Per-category ESP tables in a per-trip scaled space.

    Weights are divided by the trip maximum M, so each size-n term carries M^n, returned as
    log_scale = log M.
    """
    finite = np.where(np.isfinite(logw), logw, -np.inf)
    log_scale = finite.max(1)
    w = np.exp(finite - log_scale[:, None])
    R = world.nmax
    tables = []
    for c, items in enumerate(world.category_items()):
        depth = min(R, len(items))
        esp = np.zeros((w.shape[0], depth + 1))
        esp[:, 0] = 1.0
        for item in items:
            previous = esp.copy()
            esp[:, 1:] += w[:, item:item + 1] * previous[:, :-1]
        k = np.arange(depth + 1)
        tables.append(esp * np.exp(-world.rho_c[c] * k * (k - 1) / 2.0)[None, :])
    return w, tables, log_scale


def _prefix(tables, nmax):
    B = tables[0].shape[0]
    prefix = [np.zeros((B, nmax + 1))]
    prefix[0][:, 0] = 1.0
    for table in tables:
        current = np.zeros((B, nmax + 1))
        for k in range(table.shape[1]):
            current[:, k:] += table[:, k:k + 1] * prefix[-1][:, :nmax + 1 - k]
        prefix.append(current)
    return prefix


def log_total(world: World, b: np.ndarray, z: np.ndarray) -> np.ndarray:
    """log D(z) = log sum_{1<=|S|<=nmax} prod w_j(z) pen(S) for each row."""
    _, tables, log_scale = _category_tables(world, _log_weights(world, b, z))
    total = _prefix(tables, world.nmax)[-1]
    n = np.arange(world.nmax + 1)
    with np.errstate(divide="ignore"):
        terms = np.log(total[:, 1:]) - world.rho0[None, 1:] + n[None, 1:] * log_scale[:, None]
    return np.logaddexp.reduce(terms, axis=1)


def sample_given_z(world: World, b: np.ndarray, z: np.ndarray, rng: np.random.Generator
                   ) -> np.ndarray:
    """Exact S | z by backward sampling. Returns membership [B, J] bool."""
    w, tables, log_scale = _category_tables(world, _log_weights(world, b, z))
    prefix = _prefix(tables, world.nmax)
    B, J = w.shape
    n_axis = np.arange(world.nmax + 1)
    with np.errstate(divide="ignore"):
        size_log = (np.log(prefix[-1][:, 1:]) - world.rho0[None, 1:]
                    + n_axis[None, 1:] * log_scale[:, None])
    size_log -= size_log.max(1, keepdims=True)
    size_p = np.exp(size_log)
    size = 1 + _categorical(size_p, rng)
    membership = np.zeros((B, J), dtype=bool)
    remaining = size.copy()
    items_by_category = world.category_items()
    for c in range(world.n_categories - 1, -1, -1):
        table = tables[c]
        depth = table.shape[1] - 1
        k = np.arange(depth + 1)
        index = remaining[:, None] - k[None, :]
        valid = index >= 0
        before = np.take_along_axis(prefix[c], np.clip(index, 0, world.nmax), axis=1)
        p = np.where(valid, before * table, 0.0)
        count = _categorical(p, rng)
        remaining -= count
        items = items_by_category[c]
        # suffix ESP within category for sequential inclusion
        L = len(items)
        suffix = np.zeros((B, L + 1, depth + 1))
        suffix[:, L, 0] = 1.0
        for position in range(L - 1, -1, -1):
            suffix[:, position] = suffix[:, position + 1]
            suffix[:, position, 1:] += (w[:, items[position]][:, None]
                                        * suffix[:, position + 1, :-1])
        need = count.copy()
        rows = np.arange(B)
        for position in range(L):
            active = need > 0
            numerator = w[:, items[position]] * suffix[rows, position + 1,
                                                        np.clip(need - 1, 0, depth)]
            denominator = suffix[rows, position, np.clip(need, 0, depth)]
            probability = np.where(active & (denominator > 0), numerator / np.where(
                denominator > 0, denominator, 1.0), 0.0)
            take = rng.random(B) < probability
            membership[take, items[position]] = True
            need -= take
    return membership


def _categorical(p: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    cumulative = np.cumsum(p, axis=1)
    total = cumulative[:, -1:]
    if (total <= 0).any():
        raise RuntimeError("a trip has no admissible basket")
    u = rng.random((p.shape[0], 1)) * total
    return np.minimum((cumulative < u).sum(1), p.shape[1] - 1)


def gibbs_sample(world: World, b: np.ndarray, rng: np.random.Generator, sweeps: int = 30
                 ) -> np.ndarray:
    Kz = world.phi.shape[1]
    z = np.zeros((b.shape[0], Kz))
    membership = sample_given_z(world, b, z, rng)
    for _ in range(sweeps):
        mean = membership.astype(np.float64) @ world.phi
        z = mean + rng.standard_normal(mean.shape)
        membership = sample_given_z(world, b, z, rng)
    return membership


def quadrature(Kz: int, nodes: int):
    x, weight = np.polynomial.hermite_e.hermegauss(nodes)
    weight = weight / weight.sum()
    grid = np.array(list(itertools.product(x, repeat=Kz)))
    log_weight = np.log(np.array([np.prod(c) for c in itertools.product(weight, repeat=Kz)]))
    return grid, log_weight


def log_normalizer(world: World, b: np.ndarray, nodes: int = 24) -> np.ndarray:
    """log Z_+(x) for each row of b by tensor Gauss-Hermite quadrature over z."""
    grid, log_weight = quadrature(world.phi.shape[1], nodes)
    B = b.shape[0]
    values = np.empty((B, len(grid)))
    for q, point in enumerate(grid):
        values[:, q] = log_total(world, b, np.broadcast_to(point, (B, len(point))))
    return np.logaddexp.reduce(values + log_weight[None, :], axis=1)


def energy(world: World, b: np.ndarray, membership: np.ndarray) -> np.ndarray:
    m = membership.astype(np.float64)
    if np.any(membership & ~np.isfinite(b)):
        raise ValueError("a basket contains an unstocked product")
    linear = np.where(membership, b, 0.0).sum(1)
    v = m @ world.phi
    pair = 0.5 * ((v ** 2).sum(1) - (m @ (world.phi ** 2).sum(1)))
    counts = np.stack([m[:, items].sum(1) for items in world.category_items()], axis=1)
    category = (world.rho_c[None, :] * counts * (counts - 1) / 2.0).sum(1)
    size = world.rho0[membership.sum(1)]
    return linear + pair - category - size


def log_probability(world: World, b: np.ndarray, membership: np.ndarray, nodes: int = 24
                    ) -> np.ndarray:
    return energy(world, b, membership) - log_normalizer(world, b, nodes)
