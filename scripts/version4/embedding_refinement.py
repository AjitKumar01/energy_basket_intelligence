"""Experimental free-product Gram refinement on an exact-base stratified bank.

Unlike U C U.T fitting, every row of Phi may move. This is a nonconvex sampled
likelihood, NOT exact MLE. Anchor shrinkage and a singular-value cap constrain
the update; independent draw banks and normalizer checks are mandatory.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.special import logsumexp

from basket_incidence import pair_energy, pair_adjoint


@dataclass
class IncidenceBank:
    observed: sparse.csr_matrix
    generated: sparse.csr_matrix
    log_weight: np.ndarray
    band: np.ndarray
    size_basis: np.ndarray

    def __post_init__(self):
        self.observed = sparse.csr_matrix(self.observed, dtype=np.float64)
        self.generated = sparse.csr_matrix(self.generated, dtype=np.float64)
        self.log_weight = np.asarray(self.log_weight, dtype=np.float64)
        self.band = np.asarray(self.band, dtype=np.int64)
        self.size_basis = np.asarray(self.size_basis, dtype=np.float64)
        if self.log_weight.ndim != 2 or self.log_weight.shape[0] != self.observed.shape[0]:
            raise ValueError("draw weights must be context-major")
        if self.generated.shape != (self.log_weight.size, self.observed.shape[1]):
            raise ValueError("generated incidence must be flattened context-major")
        if self.band.shape != (self.log_weight.shape[1],):
            raise ValueError("band indices must have one entry per draw")
        if not all(np.all(x.data == 1) and x.has_canonical_format
                   for x in (self.observed, self.generated)):
            raise ValueError("canonical binary incidence is required")
        self.observed_size = np.asarray(self.observed.sum(1)).ravel().astype(int) - 1
        self.generated_size = np.asarray(self.generated.sum(1)).ravel().astype(int) - 1
        if (self.size_basis.ndim != 2 or not np.isfinite(self.size_basis).all()
                or min(self.observed_size.min(), self.generated_size.min()) < 0
                or max(self.observed_size.max(), self.generated_size.max()) >= len(self.size_basis)):
            raise ValueError("basket size outside declared support")
        if (np.isnan(self.log_weight).any() or np.isposinf(self.log_weight).any()
                or not np.allclose(logsumexp(self.log_weight, axis=1), 0, atol=1e-10)):
            raise ValueError("exact-base stratum weights must sum to one per context")

    def increments(self, phi, size):
        curve = self.size_basis @ size
        observed = pair_energy(self.observed, phi) - curve[self.observed_size]
        generated = (pair_energy(self.generated, phi) - curve[self.generated_size]
                     ).reshape(self.log_weight.shape)
        return observed, generated

    def gain(self, phi, size):
        observed, generated = self.increments(phi, size)
        return observed - logsumexp(self.log_weight + generated, axis=1)


def objective_gradient(bank, phi, size, anchor, *, anchor_ridge=0.01,
                       gram_ridge=0.001, size_ridge=0.001, size_smoothness=0.1):
    observed, generated = bank.increments(phi, size)
    logits = bank.log_weight + generated
    normalizer = logsumexp(logits, axis=1)
    probability = np.exp(logits - normalizer[:, None])
    contexts = len(observed)
    gradient_phi = (pair_adjoint(bank.observed, phi, np.ones(contexts) / contexts)
                    - pair_adjoint(bank.generated, phi, probability.ravel() / contexts))
    size_mass = (np.bincount(bank.generated_size, weights=probability.ravel(),
                             minlength=len(bank.size_basis))
                 - np.bincount(bank.observed_size, minlength=len(bank.size_basis))) / contexts
    gradient_size = bank.size_basis.T @ size_mass
    delta, gram = phi - anchor, phi.T @ phi
    difference = np.diff(size, n=2)
    penalty = (0.5 * anchor_ridge * np.square(delta).sum()
               + 0.5 * gram_ridge * np.square(gram).sum()
               + 0.5 * size_ridge * np.square(size).sum()
               + 0.5 * size_smoothness * np.square(difference).sum())
    gradient_phi -= anchor_ridge * delta + 2 * gram_ridge * phi @ gram
    gradient_size -= size_ridge * size
    gradient_size[:-2] -= size_smoothness * difference
    gradient_size[1:-1] += 2 * size_smoothness * difference
    gradient_size[2:] -= size_smoothness * difference
    return float(np.mean(observed - normalizer) - penalty), gradient_phi, gradient_size


def spectral_project(phi, maximum):
    left, value, right = np.linalg.svd(phi, full_matrices=False)
    return (left * np.minimum(value, maximum)) @ right


def compact_factor(phi):
    """Equivalent Gram factor with active coordinates first, required by quadrature."""
    left, value, _ = np.linalg.svd(phi, full_matrices=False)
    active = value > max(float(value.max(initial=0)) * 1e-10, 1e-12)
    return left[:, active] * value[active]


def fit_embeddings(bank, initial_phi, *, basis=None, initial_size=None,
                   anchor=None, steps=200, tolerance=1e-5, spectral_max=1.0,
                   size_bound=12.0, fit_size=True, label="refinement", **penalties):
    """Monotone projected ascent; basis=None releases all product coordinates.

    The fixed-basis arm uses the identical factor objective and penalties. It is
    an experimental matched control, not a replacement for the convex C solver.
    No validation observations are used in either fit or stopping decision.
    """
    if steps < 1 or spectral_max <= 0 or size_bound <= 0 or tolerance <= 0:
        raise ValueError("invalid refinement bounds")
    if any(not np.isfinite(v) or v < 0 for v in penalties.values()):
        raise ValueError("regularization must be finite and nonnegative")
    anchor = initial_phi.copy() if anchor is None else anchor.copy()
    coordinate = initial_phi.copy() if basis is None else basis.T @ initial_phi
    if basis is not None and not np.allclose(basis.T @ basis, np.eye(basis.shape[1])):
        raise ValueError("the constrained basis must be orthonormal")
    size = np.zeros(bank.size_basis.shape[1]) if initial_size is None else initial_size.copy()
    size[0] = 0
    coordinate = spectral_project(coordinate, spectral_max)
    step, records, converged = 1.0, [], False
    for iteration in range(steps + 1):
        phi = coordinate if basis is None else basis @ coordinate
        value, gp, gs = objective_gradient(bank, phi, size, anchor, **penalties)
        gc = gp if basis is None else basis.T @ gp
        gs[0] = 0
        if not fit_size:
            gs[:] = 0
        pc = spectral_project(coordinate + gc, spectral_max)
        ps = np.clip(size + gs, -size_bound, size_bound); ps[0] = 0
        residual = float(np.sqrt(np.square(pc - coordinate).sum() + np.square(ps - size).sum()))
        records.append({"iteration": iteration, "objective": value,
                        "projected_gradient_norm": residual})
        if iteration % 20 == 0 or iteration == steps:
            print(f"[{label}] step={iteration} objective={value:.8f} residual={residual:.6g}", flush=True)
        if not np.isfinite(value) or not np.isfinite(residual):
            raise FloatingPointError("nonfinite refinement objective/gradient")
        if residual <= tolerance:
            converged = True
            break
        if iteration == steps:
            break
        accepted = False
        for _ in range(40):
            trial_c = spectral_project(coordinate + step * gc, spectral_max)
            trial_s = np.clip(size + step * gs, -size_bound, size_bound); trial_s[0] = 0
            trial_phi = trial_c if basis is None else basis @ trial_c
            trial_value = objective_gradient(bank, trial_phi, trial_s, anchor, **penalties)[0]
            improvement = float(np.sum(gc * (trial_c - coordinate)) + gs @ (trial_s - size))
            if np.isfinite(trial_value) and trial_value >= value + 1e-4 * improvement:
                coordinate, size = trial_c, trial_s
                step = min(step * 1.5, 1000.0)
                accepted = True
                break
            step *= 0.5
        if not accepted:
            break
    phi = coordinate if basis is None else basis @ coordinate
    return phi, size, {
        "converged": converged, "iterations": records[-1]["iteration"],
        "projected_gradient_norm": records[-1]["projected_gradient_norm"],
        "monotone": bool(np.all(np.diff([row["objective"] for row in records]) >= -1e-12)),
        "records": records, "spectral_norm": float(np.linalg.norm(phi, 2)),
        "outside_initial_span_norm": None if basis is None else float(np.linalg.norm(phi - basis @ (basis.T @ phi))),
    }


def evaluation(bank, phi, size, household):
    from uncertainty import paired_score_summary
    from stratified_natural import within_band_ess_passes
    _, increment = bank.increments(phi, size)
    rows = []
    for band in np.unique(bank.band):
        selected = bank.band == band
        active = np.isfinite(bank.log_weight[:, selected]).any(axis=1)
        logits = increment[active][:, selected]
        if len(logits) == 0:
            continue
        weight = np.exp(logits - logsumexp(logits, axis=1, keepdims=True))
        ess = 1 / np.square(weight).sum(1)
        rows.append({"band": int(band), "draws": int(selected.sum()),
                     "ess_min": float(ess.min()),
                     "ess_fraction_min": float(ess.min() / selected.sum())})
    logits = bank.log_weight + increment
    weight = np.exp(logits - logsumexp(logits, axis=1, keepdims=True))
    sizes = bank.generated_size.reshape(weight.shape) + 1
    result = {
        "gain": paired_score_summary(bank.gain(phi, size), household), "bands": rows,
        "minimum_within_band_ess": min((r["ess_min"] for r in rows), default=0.0),
        "minimum_within_band_ess_fraction": min((r["ess_fraction_min"] for r in rows), default=0.0),
        "observed_mean_size": float(np.mean(bank.observed_size + 1)),
        "model_mean_size": float(np.mean((weight * sizes).sum(1))),
        "observed_tail_60": float(np.mean(bank.observed_size + 1 >= 60)),
        "model_tail_60": float(np.mean((weight * (sizes >= 60)).sum(1))),
    }
    result["ess_passed"] = within_band_ess_passes(result, 0.2, 2.0)
    return result
