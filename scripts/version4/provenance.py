"""Content-addressed provenance for the selected Version-4 pipeline."""
from __future__ import annotations

import hashlib
import json
import math
import numbers
import os
from pathlib import Path
from typing import Any, Mapping


FINGERPRINT_SCHEMA = 1
OPTIONAL_FINGERPRINT_FILES = frozenset({"observed_promotion_panel", "price_evidence_panel"})


def model_data_root(default: str | Path) -> Path:
    """Resolve an isolated model-data bundle without changing source-code paths."""
    configured = os.environ.get("ENERGY_MODEL_DATA_ROOT")
    return Path(configured).expanduser().resolve() if configured else Path(default).resolve()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False).encode("utf-8")


def payload_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def json_safe(value: Any) -> Any:
    """Recursively map non-finite diagnostics to JSON null."""
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        converted = float(value)
        return converted if math.isfinite(converted) else None
    return value


def strict_json_dumps(value: Any, *, indent: int = 2) -> str:
    return json.dumps(json_safe(value), indent=indent, allow_nan=False) + "\n"


def build_data_fingerprint(root: str | Path, *, write: bool = True,
                           optional_names: set[str] | None = None) -> dict[str, Any]:
    """Build one immutable identity for all model-facing data and support files.

    Optional panels are included when present.  Verification passes the optional names a
    stored fingerprint recorded, so a bundle written before an optional panel was added to
    the identity still verifies, while every file it did record is re-hashed.
    """
    root = model_data_root(root)
    basket = root / "basket_input"
    data = root / "data"
    paths = {
        "preprocessing_manifest": basket / "preprocessing_manifest.json",
        "affinity_manifest": basket / "affinity_manifest.json",
        "affinity_partition": basket / "items_affinity.parquet",
        "basket_meta": basket / "meta.json",
        "base_build_meta": data / "build_meta.json",
        "ragged_index": basket / "v3_index_affinity.npz",
    }
    optional_paths = {
        "observed_promotion_panel": basket / "promo_observed.npz",
        "price_evidence_panel": data / "price_week.parquet",
    }
    if optional_names is None:
        paths.update({name: path for name, path in optional_paths.items()
                      if path.is_file()})
    else:
        unknown = set(optional_names).difference(optional_paths)
        if unknown:
            raise ValueError(f"unknown optional fingerprint files: {sorted(unknown)}")
        paths.update({name: optional_paths[name] for name in optional_names})
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "cannot build model-data fingerprint; missing: " + ", ".join(missing))
    preprocessing = json.loads(paths["preprocessing_manifest"].read_text())
    affinity = json.loads(paths["affinity_manifest"].read_text())
    basket_meta = json.loads(paths["basket_meta"].read_text())
    build_meta = json.loads(paths["base_build_meta"].read_text())
    price_basis = basket_meta.get("price_basis")
    if not isinstance(price_basis, str) or not price_basis.strip():
        raise ValueError("basket metadata has no declared price_basis")
    if build_meta.get("price_basis") != price_basis:
        raise ValueError("Stage-01 and basket metadata price bases differ")
    affinity_digest = file_sha256(paths["affinity_partition"])
    if affinity.get("partition_sha256") != affinity_digest:
        raise ValueError("affinity partition does not match affinity_manifest.json")
    body = {
        "schema_version": FINGERPRINT_SCHEMA,
        "price_basis": price_basis,
        "cohort": preprocessing.get("cohort"),
        "raw_sha256": preprocessing.get("raw_sha256"),
        "derived_sha256": preprocessing.get("derived_sha256"),
        "files": {name: file_sha256(path) for name, path in paths.items()},
    }
    result = {**body, "fingerprint_sha256": payload_sha256(body)}
    if write:
        output = basket / "model_data_fingerprint.json"
        temporary = output.with_name(f".{output.name}.pending-{os.getpid()}")
        temporary.write_text(strict_json_dumps(result))
        os.replace(temporary, output)
    return result


def load_data_fingerprint(root: str | Path, *, verify_files: bool = False) -> dict[str, Any]:
    """Load and self-verify the current model-data identity.

    The pipeline regenerates this file after auditing and rebuilding the ragged index on
    every invocation. ``verify_files`` is available to standalone consumers that did not
    enter through the driver.
    """
    root = model_data_root(root)
    path = root / "basket_input" / "model_data_fingerprint.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"missing {path}; rerun the pipeline data stage before loading checkpoints")
    result = json.loads(path.read_text())
    digest = result.pop("fingerprint_sha256", None)
    if result.get("schema_version") != FINGERPRINT_SCHEMA:
        raise ValueError("unsupported model-data fingerprint schema")
    if payload_sha256(result) != digest:
        raise ValueError("model-data fingerprint failed its self digest")
    result["fingerprint_sha256"] = digest
    if verify_files:
        recorded_optional = set(result.get("files", {})).intersection(
            OPTIONAL_FINGERPRINT_FILES)
        rebuilt = build_data_fingerprint(
            root, write=False, optional_names=recorded_optional)
        if rebuilt["fingerprint_sha256"] != digest:
            raise ValueError("model-facing files differ from model_data_fingerprint.json")
    return result


def require_fingerprint(recorded: str | None, root: str | Path) -> dict[str, Any]:
    current = load_data_fingerprint(root, verify_files=True)
    if not recorded:
        raise ValueError(
            "artifact predates data-fingerprint enforcement; restart from initialization")
    if recorded != current["fingerprint_sha256"]:
        raise ValueError(
            "artifact data fingerprint differs from the current audited dataset")
    return current


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    result = build_data_fingerprint(root)
    print(f"[provenance] data fingerprint {result['fingerprint_sha256']}", flush=True)
