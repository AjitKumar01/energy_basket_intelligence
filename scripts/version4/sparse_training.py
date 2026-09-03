"""Training-side manager for certified sparse rules and Phi-only score correction.

The manager is opt-in and contains no optimizer loop.  It installs deterministic rules,
computes a population low/high normalizer-score correction on declared calibration batches,
and reports when parameter drift leaves the audited anchor envelope.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from pathlib import Path

import torch

from adaptive_sparse import sparse_rule


@torch.no_grad()
def rotate_phi_to_natural_basis(model) -> torch.Tensor:
    """Right-rotate Phi so ``Phi'Phi`` is diagonal in descending natural order.

    The returned orthogonal matrix satisfies ``Phi_new = Phi_old @ Q``.  This must be
    called before constructing an optimizer; rotating Phi without its saved optimizer
    moments would change the optimization state even though it preserves the model law.
    """
    gram = model.phi.T @ model.phi
    gram = 0.5 * (gram + gram.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    order = torch.argsort(eigenvalues, descending=True)
    rotation = eigenvectors.index_select(1, order)
    model.phi.copy_(model.phi @ rotation)
    return rotation


class SparseRuleManager:
    def __init__(self, sequence: Iterable[tuple[int, ...]], *, dimension: int,
                 low_budget: int, high_budget: int, audit_budget: int,
                 phi_relative_gate: float = 0.10, rho_c_rms_gate: float = 0.02,
                 utility_rms_gate: float = 0.15):
        self.sequence = tuple(tuple(int(level) for level in index) for index in sequence)
        if not self.sequence or any(len(index) != dimension for index in self.sequence):
            raise ValueError("sparse rule sequence has the wrong latent dimension")
        if not 0 < low_budget < high_budget < audit_budget <= len(self.sequence):
            raise ValueError("require 0 < low < high < audit <= sequence length")
        if min(phi_relative_gate, rho_c_rms_gate, utility_rms_gate) <= 0:
            raise ValueError("sparse rule drift gates must be positive")
        self.dimension = int(dimension)
        self.budgets = {"low": int(low_budget), "high": int(high_budget),
                        "audit": int(audit_budget)}
        self.rules = {
            name: sparse_rule(self.sequence[:budget])
            for name, budget in self.budgets.items()
        }
        self.phi_relative_gate = float(phi_relative_gate)
        self.rho_c_rms_gate = float(rho_c_rms_gate)
        self.utility_rms_gate = float(utility_rms_gate)
        self.training_fidelity = "low"
        self.phi_correction = None
        self.anchor_phi = self.anchor_rho_c = self.anchor_lam = None
        self.anchor_context_b = None

    @classmethod
    def from_json(cls, path: str | Path, **kwargs):
        payload = json.loads(Path(path).read_text())
        sequence = payload.get("accepted_sequence")
        if sequence is None:
            raise ValueError("sparse-rule JSON has no accepted_sequence")
        dimension = len(sequence[0])
        return cls(sequence, dimension=dimension, **kwargs)

    @classmethod
    def from_artifact(cls, payload: dict, **kwargs):
        sequence = payload.get("accepted_sequence")
        if sequence is None:
            raise ValueError("sparse initialization artifact has no accepted_sequence")
        dimension = len(sequence[0])
        return cls(sequence, dimension=dimension, **kwargs)

    def install(self, model, fidelity: str = "low"):
        if fidelity not in self.rules:
            raise ValueError(f"unknown sparse fidelity {fidelity}")
        if int(model.Kz) != self.dimension:
            raise ValueError("model rank does not match sparse-rule dimension")
        nodes, weights = self.rules[fidelity]
        model.quad = (nodes.to(dtype=model.lam.dtype, device=model.lam.device),
                      weights.to(dtype=model.lam.dtype, device=model.lam.device))
        model.quad_a = None
        model.quad_mix_a = None
        return len(nodes)

    @property
    def reference_fidelity(self):
        return "high" if self.training_fidelity == "low" else "audit"

    def install_training(self, model):
        return self.install(model, self.training_fidelity)

    def escalate(self, model) -> bool:
        """Move monotonically from low to high training fidelity."""
        if self.training_fidelity == "high":
            return False
        self.training_fidelity = "high"
        self.install_training(model)
        return True

    def set_anchor(self, model, phi_correction: torch.Tensor,
                   context_b: torch.Tensor | None = None):
        if phi_correction.shape != model.phi.shape:
            raise ValueError("Phi correction has the wrong shape")
        self.phi_correction = phi_correction.detach().clone()
        self.anchor_phi = model.phi.detach().clone()
        self.anchor_rho_c = model.rho_c.detach().clone()
        self.anchor_lam = model.lam.detach().clone()
        self.anchor_context_b = (None if context_b is None
                                 else context_b.detach().clone())

    def apply_phi_loss_correction(self, model):
        """Add the high-minus-low log-Z score to an existing loss gradient."""
        if self.phi_correction is None:
            raise RuntimeError("sparse Phi correction has not been calibrated")
        if model.phi.grad is None:
            raise RuntimeError("loss produced no Phi gradient")
        model.phi.grad.add_(self.phi_correction.to(
            dtype=model.phi.grad.dtype, device=model.phi.grad.device))

    def drift(self, model, context_b: torch.Tensor | None = None):
        if self.anchor_phi is None:
            raise RuntimeError("sparse rule manager has no anchor")
        phi_denominator = self.anchor_phi.norm().clamp_min(1e-30)
        if self.anchor_context_b is not None:
            if context_b is None:
                context_b = self.anchor_context_b
            if context_b.shape != self.anchor_context_b.shape:
                raise ValueError("context-utility drift requires the calibrated b signature")
            utility_rms = float((context_b.detach() - self.anchor_context_b)
                                .square().mean().sqrt())
        else:
            # Backward-compatible unit-test/fallback path.  Production sparse artifacts
            # always calibrate an actual b(x) signature over fixed contexts.
            utility_rms = float((model.lam.detach() - self.anchor_lam)
                                .square().mean().sqrt())
        return {
            "phi_relative": float(((model.phi.detach() - self.anchor_phi).norm()
                                   / phi_denominator)),
            "rho_c_rms": float((model.rho_c.detach() - self.anchor_rho_c)
                               .square().mean().sqrt()),
            "utility_rms": utility_rms,
        }

    def needs_refresh(self, model, context_b: torch.Tensor | None = None):
        value = self.drift(model, context_b)
        return bool(value["phi_relative"] >= self.phi_relative_gate
                    or value["rho_c_rms"] >= self.rho_c_rms_gate
                    or value["utility_rms"] >= self.utility_rms_gate), value


def calibrate_population_phi_correction(
        model, manager: SparseRuleManager, batches: Iterable,
        install_batch: Callable[[object], None]) -> tuple[torch.Tensor, dict]:
    """Average ``grad_phi logZ_high - grad_phi logZ_low`` on fixed batches.

    ``install_batch(batch)`` must set ``model.house`` and ``model.ctx`` and return the
    corresponding ragged index.  Calibration differentiates only log Z; observed energy and
    every non-Phi objective term cancel exactly from the low/high difference.
    """
    corrections = []
    context_values = []
    context_count = 0
    for batch in batches:
        ix = install_batch(batch)
        context_count += int(ix.B)
        with torch.no_grad():
            context_values.append(model.b_flat(ix).detach().reshape(-1))
        manager.install(model, manager.training_fidelity)
        low = torch.autograd.grad(
            model.log_Z(ix, drop_empty=True).mean(), model.phi)[0]
        manager.install(model, manager.reference_fidelity)
        high = torch.autograd.grad(
            model.log_Z(ix, drop_empty=True).mean(), model.phi)[0]
        corrections.append((high - low).detach())
    if not corrections:
        raise ValueError("Phi correction calibration needs at least one batch")
    stacked = torch.stack(corrections)
    correction = stacked.mean(0)
    deviation = (stacked - correction).flatten(1).norm(dim=1)
    se_norm = (deviation.square().mean().sqrt() / len(corrections) ** 0.5
               if len(corrections) > 1 else torch.tensor(float("nan")))
    manager.set_anchor(model, correction, torch.cat(context_values))
    manager.install_training(model)
    return correction, {
        "batches": len(corrections),
        "contexts": context_count,
        "correction_l2": float(correction.norm()),
        "between_batch_se_l2": float(se_norm),
        "base_fidelity": manager.training_fidelity,
        "reference_fidelity": manager.reference_fidelity,
        "low_nodes": len(manager.rules[manager.training_fidelity][0]),
        "high_nodes": len(manager.rules[manager.reference_fidelity][0]),
    }
