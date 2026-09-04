"""Neutral checkpoint loading and cross-clone relocation helpers."""
from __future__ import annotations

from pathlib import Path

import torch

from provenance import require_fingerprint
from ragged import RaggedModel
from sparse_artifact import load_sparse_initialization_artifact


ROOT = Path(__file__).resolve().parents[2]


def resolve_initialization_artifact(checkpoint: Path, configured: str | Path) -> Path:
    configured = Path(configured)
    if configured.is_file():
        return configured
    checkpoint_root = checkpoint.resolve().parents[1]
    candidates = (
        checkpoint_root / "artifacts" / configured.name,
        checkpoint.resolve().parent / configured.name,
        ROOT / "artifacts" / configured.name,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"checkpoint refers to missing initialization artifact {configured}; "
        f"restore it as {checkpoint_root / 'artifacts' / configured.name}")


def require_capabilities(blob: dict, *required: str) -> dict[str, bool]:
    capabilities = blob.get("trained_capabilities")
    if not isinstance(capabilities, dict):
        raise ValueError(
            "checkpoint predates trained-capability declarations; restart the pipeline")
    missing = [name for name in required if capabilities.get(name) is not True]
    if missing:
        raise ValueError(
            "checkpoint does not support required capabilities: " + ", ".join(missing))
    return {str(name): bool(value) for name, value in capabilities.items()}


def load_checkpoint(path: Path, data, *, required_capabilities=()):
    path = Path(path)
    blob = torch.load(path, map_location="cpu", weights_only=False)
    artifact = resolve_initialization_artifact(path, blob["config"]["artifact"])
    raw = torch.load(artifact, map_location="cpu", weights_only=False)
    meta = raw["metadata"]
    if blob.get("fresh_artifact_digest") != raw.get("model_state_sha256"):
        raise ValueError("checkpoint and initialization state digests differ")
    fingerprint = meta.get("data_fingerprint_sha256")
    require_fingerprint(fingerprint, ROOT)
    if blob.get("data_fingerprint_sha256") != fingerprint:
        raise ValueError("checkpoint and initialization data fingerprints differ")
    require_capabilities(blob, *required_capabilities)
    model = RaggedModel(
        int(data["n_item"]), int(data["n_user"]), int(data["n_cat"]),
        K=int(meta["K"]), Kz=int(meta["Kz"]), nmax=int(meta["nmax"]),
        R=int(meta["R"]), seed=int(meta["seed"]), S=int(data["n_store"]),
        Kp=int(meta["Kp"]), phi_init=0.0,
        household_size_rank1=bool(meta.get("household_size_rank1", False)))
    load_sparse_initialization_artifact(artifact, model)
    model.load_state_dict(blob["model"], strict=True)
    model._poly_degree_native = True
    model._esp_native = True
    model._esp_log_blocked = True
    model.double().eval()
    return model, blob, meta
