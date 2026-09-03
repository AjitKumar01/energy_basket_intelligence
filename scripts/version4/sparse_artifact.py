"""Reproducible initialization artifacts for Version-4 sparse training.

An adaptive rule is part of the numerical method, not a universal table of nodes.  Its
index set depends on the integrand at the state where it was constructed.  This module
therefore binds a rule to the exact *untrained* model state and data support used by the
preflight audit.  No optimizer state or learned checkpoint is stored.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np
import torch


ARTIFACT_FORMAT = 1


def _tensor_bytes(value: torch.Tensor) -> bytes:
    tensor = value.detach().cpu().contiguous()
    return (str(tensor.dtype).encode() + b"\0"
            + np.asarray(tensor.shape, dtype=np.int64).tobytes()
            + tensor.numpy().tobytes())


def tensor_mapping_sha256(values: Mapping[str, torch.Tensor]) -> str:
    """Stable digest of a named tensor mapping, including names, shapes and dtypes."""
    digest = hashlib.sha256()
    for name in sorted(values):
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(_tensor_bytes(values[name]))
    return digest.hexdigest()


def model_state_sha256(model: torch.nn.Module) -> str:
    return tensor_mapping_sha256(model.state_dict())


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


def select_calibration_trips(data, training: np.ndarray, count: int,
                             *, seed: int = 2718) -> np.ndarray:
    """Deterministic coverage sample over assortment and observed basket-size tails.

    Half the slots cover the joint two-dimensional rank of assortment and basket size;
    one quarter explicitly covers the largest assortments and one quarter the largest
    observed baskets.  Duplicates are filled by a fixed random permutation.  Selection
    uses training data only.
    """
    training = np.asarray(training, dtype=np.int64)
    if count <= 0 or count > len(training):
        raise ValueError("calibration count must lie in 1..number of training trips")
    C, S = int(data["n_cat"]), int(data["n_store"])
    ptr = data["store_cat_ptr"]
    stocked = np.asarray([ptr[(store + 1) * C] - ptr[store * C]
                          for store in range(S)])
    assortment = stocked[data["trip_store"][training]]
    basket = data["trip_nlines"][training]
    n_assort = count // 4
    n_basket = count // 4
    n_joint = count - n_assort - n_basket
    chosen: list[int] = []

    def extend(candidates: Iterable[int]):
        seen = set(chosen)
        for index in candidates:
            value = int(index)
            if value not in seen:
                chosen.append(value)
                seen.add(value)

    extend(np.argsort(assortment, kind="stable")[::-1][:n_assort])
    extend(np.argsort(basket, kind="stable")[::-1][:n_basket])
    # Equal-weight empirical ranks avoid arbitrary unit scaling between the two axes.
    arank = np.empty(len(training), dtype=np.float64)
    brank = np.empty(len(training), dtype=np.float64)
    arank[np.argsort(assortment, kind="stable")] = np.linspace(0, 1, len(training))
    brank[np.argsort(basket, kind="stable")] = np.linspace(0, 1, len(training))
    joint_order = np.argsort(arank + brank, kind="stable")
    positions = np.linspace(0, len(joint_order) - 1, max(n_joint, 1)).astype(int)
    extend(joint_order[positions])
    extend(np.random.default_rng(seed).permutation(len(training)))
    return training[np.asarray(chosen[:count], dtype=np.int64)]


def save_sparse_initialization_artifact(path: str | Path, model: torch.nn.Module, *,
                                        metadata: Mapping, sequence: Iterable,
                                        calibration_trips: Iterable[int]) -> dict:
    """Save an untrained state and its adaptive rule as one fail-closed artifact."""
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
        "model_state_sha256": tensor_mapping_sha256(state),
        "model_state": state,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return {key: value for key, value in payload.items() if key != "model_state"}


def load_sparse_initialization_artifact(path: str | Path, model: torch.nn.Module,
                                        *, expected_metadata: Mapping | None = None) -> dict:
    """Restore and verify the exact untrained state bound to a sparse rule."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("format") != ARTIFACT_FORMAT or payload.get("kind") != \
            "version4_sparse_untrained_initialization":
        raise ValueError("not a supported Version-4 sparse initialization artifact")
    state = payload.get("model_state")
    if not isinstance(state, dict):
        raise ValueError("sparse initialization artifact has no model state")
    digest = tensor_mapping_sha256(state)
    if digest != payload.get("model_state_sha256"):
        raise ValueError("sparse initialization artifact failed its state digest")
    if expected_metadata:
        got = payload.get("metadata", {})
        mismatch = {key: (got.get(key), value) for key, value in expected_metadata.items()
                    if got.get(key) != value}
        if mismatch:
            raise ValueError(f"sparse initialization metadata mismatch: {mismatch}")
    model.load_state_dict(state, strict=True)
    if model_state_sha256(model) != digest:
        raise RuntimeError("restored sparse initialization differs from certified state")
    return {key: value for key, value in payload.items() if key != "model_state"}


def write_artifact_manifest(path: str | Path, summary: Mapping):
    Path(path).write_text(json.dumps(dict(summary), indent=2, sort_keys=True) + "\n")
