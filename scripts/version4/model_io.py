"""Load a fitted Version-4 model file written by one of the pipeline stages."""
from __future__ import annotations

from pathlib import Path
import os

import torch

from ragged import RaggedModel
from sparse_artifact import load_sparse_initialization_artifact


ROOT = Path(__file__).resolve().parents[2]


def require_capabilities(blob: dict, *required: str) -> dict[str, bool]:
    """Refuse a model file that was not trained for what the caller needs."""
    capabilities = blob.get("trained_capabilities")
    if not isinstance(capabilities, dict):
        raise ValueError("model file declares no trained capabilities")
    missing = [name for name in required if capabilities.get(name) is not True]
    if missing:
        raise ValueError(
            "model file does not support required capabilities: " + ", ".join(missing))
    return {str(name): bool(value) for name, value in capabilities.items()}


def floating_state_dtype(state: dict[str, torch.Tensor]) -> torch.dtype:
    """Return the unique floating dtype of a saved model state.

    Model construction follows PyTorch's process-wide default dtype, so the model is cast
    to the saved dtype before loading; otherwise float64 values could be silently cast
    into a float32 model.
    """
    dtypes = {value.dtype for value in state.values()
              if isinstance(value, torch.Tensor) and value.is_floating_point()}
    if len(dtypes) != 1:
        raise ValueError(
            "initialization must contain exactly one floating dtype; "
            f"found {sorted(map(str, dtypes))}")
    return next(iter(dtypes))


def load_model(path: Path, data, *, required_capabilities=()):
    """Return ``(model, blob, meta)`` for a fitted model file.

    ``blob`` is the saved dictionary, and ``meta`` the dimensions recorded by the
    initialization file the fit started from (``blob["config"]["artifact"]``).
    """
    blob = torch.load(Path(path), map_location="cpu", weights_only=False)
    artifact = Path(blob["config"]["artifact"])
    artifact = artifact if artifact.is_absolute() else ROOT / artifact
    raw = torch.load(artifact, map_location="cpu", weights_only=False)
    meta = raw["metadata"]
    require_capabilities(blob, *required_capabilities)
    price_component = blob.get("supported_price_component") or {}
    price_contract = price_component.get("price_feature_contract", "chain_and_store")
    if price_contract not in {"chain_and_store", "chain_product_week_only"}:
        raise ValueError(f"model file has unknown price feature contract {price_contract!r}")
    # Feature objects are built right after loading; this makes every process score the
    # same price panel the model was fitted on.
    os.environ["ENERGY_PRICE_FEATURE_CONTRACT"] = price_contract
    model = RaggedModel(
        int(data["n_item"]), int(data["n_user"]), int(data["n_cat"]),
        K=int(meta["K"]), Kz=int(meta["Kz"]), nmax=int(meta["nmax"]),
        R=int(meta["R"]), seed=int(meta["seed"]), S=int(data["n_store"]),
        Kp=int(meta["Kp"]), phi_init=0.0,
        household_size_rank1=bool(meta.get("household_size_rank1", False)))
    model.to(dtype=floating_state_dtype(raw["model_state"]))
    load_sparse_initialization_artifact(artifact, model)
    model.load_state_dict(blob["model"], strict=True)
    model._poly_degree_native = True
    model._esp_native = True
    model._esp_log_blocked = True
    model.double().eval()
    return model, blob, meta
