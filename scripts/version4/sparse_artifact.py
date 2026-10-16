"""The untrained initialization file that every Version-4 fit starts from.

It stores the model dimensions (``metadata``) and the untrained model state.  No optimizer
state or fitted parameters are stored.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

import torch


ARTIFACT_FORMAT = 1


@torch.no_grad()
def initialize_nested_trace_class_phi(model, *, active_rank: int,
                                      row_rms: float = 0.03,
                                      decay: float = 0.84,
                                      seed: int = 823) -> dict:
    """Initialize an exact nested Gram-interaction submodel.

    Every catalogue row is active, while columns ``active_rank:Kz`` are exactly zero.
    The nonzero singular values follow a trace-class geometric prior and are normalized so
    ``||Phi||_F / sqrt(J) == row_rms``.  This is an initialization choice only: the model
    remains ``W = Phi Phi'`` and zero-padded columns can later be activated by the
    second-order rank-continuation score.
    """
    if not 1 <= int(active_rank) <= int(model.Kz):
        raise ValueError("active rank must lie in 1..Kz")
    if row_rms <= 0 or not 0 < decay < 1:
        raise ValueError("row_rms must be positive and decay must lie in (0,1)")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    left, _ = torch.linalg.qr(torch.randn(
        model.J, active_rank, generator=generator, dtype=model.phi.dtype,
        device="cpu"), mode="reduced")
    singular = decay ** torch.arange(active_rank, dtype=model.phi.dtype)
    singular *= (model.J ** 0.5 * float(row_rms)) / singular.norm()
    phi = torch.zeros(model.J, model.Kz, dtype=model.phi.dtype)
    phi[:, :active_rank] = left * singular
    model.phi.copy_(phi.to(device=model.phi.device))
    gram_eigenvalues = torch.linalg.eigvalsh(model.phi.T @ model.phi).flip(0)
    return {
        "active_rank": int(active_rank),
        "row_rms": float(model.phi.norm() / model.J ** 0.5),
        "nonzero_rows": int((model.phi.norm(dim=1) > 0).sum()),
        "decay": float(decay),
        "seed": int(seed),
        "gram_eigenvalues_desc": gram_eigenvalues.cpu().tolist(),
    }


def save_sparse_initialization_artifact(path: str | Path, model: torch.nn.Module, *,
                                        metadata: Mapping, sequence: Iterable,
                                        calibration_trips: Iterable[int]) -> dict:
    """Save an untrained model state with its metadata."""
    path = Path(path)
    state = {name: value.detach().cpu().clone()
             for name, value in model.state_dict().items()}
    accepted = [list(map(int, index)) for index in sequence]
    if not accepted:
        raise ValueError("artifact requires a nonempty sparse sequence")
    payload = {
        "format": ARTIFACT_FORMAT,
        "kind": "version4_sparse_untrained_initialization",
        "metadata": dict(metadata),
        "accepted_sequence": accepted,
        "calibration_trips": [int(value) for value in calibration_trips],
        "model_state": state,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return {key: value for key, value in payload.items() if key != "model_state"}


def load_sparse_initialization_artifact(path: str | Path, model: torch.nn.Module,
                                        *, expected_metadata: Mapping | None = None) -> dict:
    """Load the untrained state into ``model`` and return the file's other fields."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("format") != ARTIFACT_FORMAT or payload.get("kind") != \
            "version4_sparse_untrained_initialization":
        raise ValueError("not a supported Version-4 sparse initialization artifact")
    state = payload.get("model_state")
    if not isinstance(state, dict):
        raise ValueError("sparse initialization artifact has no model state")
    if expected_metadata:
        got = payload.get("metadata", {})
        mismatch = {key: (got.get(key), value) for key, value in expected_metadata.items()
                    if got.get(key) != value}
        if mismatch:
            raise ValueError(f"sparse initialization metadata mismatch: {mismatch}")
    model.load_state_dict(state, strict=True)
    return {key: value for key, value in payload.items() if key != "model_state"}

