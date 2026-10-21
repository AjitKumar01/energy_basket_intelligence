"""Reading and writing the three kinds of saved model.

    initialization   untrained starting values and the model's sizes (scripts/initialize.py)
    additive         the exact base model, phi = 0 (scripts/fit_additive.py)
    version-4        the full model with interactions (scripts/fit_interactions.py)

Each later file records the earlier ones it was built from, so loading a Version-4 model
needs only its own path.  Older checkpoints written by the research code (branch
v4-training) load too: their extra keys are ignored and their flags are translated.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from .batches import Batcher, PurchaseHistory
from .bundle import data_root, require_fingerprint
from .features import Features
from .model import BasketModel

INIT_KINDS = ("version4_sparse_untrained_initialization", "version4_untrained_initialization")


def _load(path):
    return torch.load(Path(path), map_location="cpu", weights_only=False)


def state_digest(state) -> str:
    """sha256 over the names, dtypes, shapes and values of a state dict."""
    digest = hashlib.sha256()
    for name in sorted(state):
        t = state[name].detach().cpu().contiguous()
        digest.update(name.encode() + b"\0")
        digest.update(str(t.dtype).encode() + b"\0" + np.asarray(t.shape, dtype=np.int64).tobytes()
                      + t.numpy().tobytes())
    return digest.hexdigest()


# ---- initialization ---------------------------------------------------------------------------
def save_initialization(path, model, metadata):
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    torch.save({"format": 1, "kind": INIT_KINDS[1], "metadata": metadata,
                "model_state_sha256": state_digest(state), "model_state": state}, Path(path))


def load_initialization(path, data, root=None):
    """The untrained model of an initialization file, with phi = 0, and its metadata."""
    raw = _load(path)
    if raw.get("kind") not in INIT_KINDS:
        raise ValueError(f"{path} is not an initialization artifact")
    if state_digest(raw["model_state"]) != raw["model_state_sha256"]:
        raise ValueError(f"{path} fails its state digest")
    meta = raw["metadata"]
    require_fingerprint(meta.get("data_fingerprint_sha256"), root)
    if not meta.get("household_size_rank1", False):
        raise ValueError("this pipeline needs household_size_rank1 initializations")
    model = BasketModel(int(data["n_item"]), int(data["n_user"]), int(data["n_cat"]),
                        K=int(meta["K"]), Kz=int(meta["Kz"]), Kp=int(meta["Kp"]),
                        nmax=int(meta["nmax"]), R=int(meta["R"]), S=int(data["n_store"]),
                        seed=int(meta["seed"]))
    model.load_fitted_state(raw["model_state"])
    with torch.no_grad():
        model.phi.zero_()
    return model, meta


# ---- additive base model ------------------------------------------------------------------------
def price_contract(blob) -> str:
    """The price panel a checkpoint was fitted with (old checkpoints: chain_and_store)."""
    return (blob.get("price_feature_contract")
            or (blob.get("supported_price_component") or {}).get("price_feature_contract")
            or "chain_and_store")


def discount_panel(blob):
    """The discount panel a checkpoint was fitted with, or None (no discount split)."""
    return (blob.get("discount_split") or {}).get("discount_panel") or blob.get("discount_panel")


def load_additive(initialization, additive, data, root=None):
    """The fitted base model (phi = 0) and its blob."""
    model, meta = load_initialization(initialization, data, root)
    blob = _load(additive)
    model.load_fitted_state(blob["model"])
    if float(model.phi.detach().abs().max()) != 0.0:
        raise ValueError(f"{additive} is not an additive (phi = 0) model")
    return model, meta, blob


def batcher_for(data, root, additive_blob, *, recency=False, purchase_history=False,
                contract=None, panel=None):
    """A Batcher with the features the base model was fitted on, plus optional history.
    ``contract`` and ``panel`` override the recorded price contract and discount panel."""
    features = Features(data_root(root), int(data["n_item"]), int(data["n_store"]),
                        recency=recency, price_contract=contract or price_contract(additive_blob),
                        discount_panel=panel or discount_panel(additive_blob))
    return Batcher(data, features, recency=recency,
                   purchase_history=PurchaseHistory(data) if purchase_history else None)


# ---- Version-4 model -------------------------------------------------------------------------------
def version4_flags(blob) -> dict:
    """Rank and the non-parameter switches of a Version-4 checkpoint."""
    state = blob["model"]
    if "iv_slope_raw" in state and not blob.get("iv_price_only", False):
        raise ValueError("M4b (full inclusive value) checkpoints are not supported; only M4c")
    return {"rank": int(blob["rank"]), "size_on_categories": bool(blob.get("cat_rho0", False)),
            "recency": bool(blob.get("recency", False)), "max_per_category": blob.get("max_per_category")}


def load_model(path, data, root=None, *, initialization=None, additive=None):
    """An additive or a Version-4 checkpoint: (model, meta, parent blob, flags, its blob).

    For a Version-4 checkpoint the parent files are read from it; old checkpoints that do
    not record them need ``initialization`` and ``additive``.  An additive model has rank 0."""
    blob = _load(path)
    if "rank" not in blob:                                        # additive base model
        model, meta, parent = load_additive(initialization or blob.get("initialization"), path, data, root)
        flags = {"rank": 0, "size_on_categories": False, "recency": False, "max_per_category": None}
        return model, meta, parent, flags, blob
    model, meta, parent = load_additive(initialization or blob["initialization"],
                                        additive or blob["additive"], data, root)
    flags = version4_flags(blob)
    model.load_fitted_state(blob["model"])
    model.rank = flags["rank"]
    model.size_on_categories = flags["size_on_categories"]
    if flags["max_per_category"] is not None:
        model.R = int(flags["max_per_category"])
    return model, meta, parent, flags, blob


def load_version4(path, data, root=None, *, initialization=None, additive=None):
    """A trained Version-4 model, its Batcher (the base model's features plus history),
    its blob and metadata."""
    model, meta, parent, flags, blob = load_model(path, data, root, initialization=initialization,
                                                  additive=additive)
    batcher = batcher_for(data, root, parent, recency=flags["recency"], purchase_history=flags["recency"])
    return model, batcher, blob, meta


def freeze(model):
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model
