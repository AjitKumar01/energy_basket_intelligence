#!/usr/bin/env python3
"""Hardware discovery and execution policy for the certified Version-4 pipeline.

The likelihood is not an ordinary dense neural-network loss. Its exact
category/cardinality normalizer and probability adjoint are implemented by
``poly_degree_native.cpp`` and currently accept CPU float64 tensors only. This module
keeps hardware discovery separate from backend eligibility: seeing a CUDA device must not
cause the pipeline to move tensors there and either fail or change the estimator.
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class RuntimeCapabilities:
    python: str
    platform: str
    machine: str
    logical_cpu_count: int
    recommended_cpu_threads: int
    memory_gib: float | None
    workspace_free_gib: float
    torch: str
    cuda_built: str | None
    cuda_available: bool
    cuda_device_count: int
    cuda_devices: list[dict[str, object]]
    mps_available: bool
    certified_training_backend: str
    accelerator_eligible_stages: list[str]
    accelerator_ineligible_reason: str


def _memory_gib() -> float | None:
    """Best-effort physical-memory query without an optional dependency."""
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        pages = int(os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, KeyError, OSError, ValueError):
        return None
    return round(page_size * pages / 2**30, 2)


def recommended_cpu_threads(logical_cpu_count: int | None = None) -> int:
    """Conservative thread count for the small native DP kernels."""
    logical = max(1, int(logical_cpu_count or os.cpu_count() or 1))
    likely_physical = logical if logical <= 4 else max(1, logical // 2)
    return min(8, likely_physical)


def detect_runtime(workspace: Path | None = None) -> RuntimeCapabilities:
    logical = max(1, int(os.cpu_count() or 1))
    cuda_available = bool(torch.cuda.is_available())
    cuda_devices: list[dict[str, object]] = []
    if cuda_available:
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            cuda_devices.append({
                "index": index,
                "name": properties.name,
                "compute_capability": list(torch.cuda.get_device_capability(index)),
                "memory_gib": round(properties.total_memory / 2**30, 2),
            })
    mps = getattr(torch.backends, "mps", None)
    mps_available = bool(mps is not None and mps.is_available())
    return RuntimeCapabilities(
        python=platform.python_version(),
        platform=platform.platform(),
        machine=platform.machine(),
        logical_cpu_count=logical,
        recommended_cpu_threads=recommended_cpu_threads(logical),
        memory_gib=_memory_gib(),
        workspace_free_gib=round(
            shutil.disk_usage(workspace or Path.cwd()).free / 2**30, 2),
        torch=torch.__version__,
        cuda_built=torch.version.cuda,
        cuda_available=cuda_available,
        cuda_device_count=len(cuda_devices),
        cuda_devices=cuda_devices,
        mps_available=mps_available,
        certified_training_backend="cpu",
        accelerator_eligible_stages=[],
        accelerator_ineligible_reason=(
            "The exact float64 ESP/category-polynomial normalizer and its custom "
            "probability adjoint are CPU-only. Rank fitting also uses SciPy sparse and "
            "convex CPU solvers. Moving only dense utility operations to an accelerator "
            "would add host/device transfers without moving the dominant work."
        ),
    )


def resolve_backend(requested: str, capabilities: RuntimeCapabilities) -> str:
    """Return a mathematically supported pipeline backend or fail explicitly."""
    requested = requested.lower()
    if requested not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError(f"unknown device policy: {requested}")
    if requested in {"auto", "cpu"}:
        return "cpu"
    available = (capabilities.cuda_available if requested == "cuda"
                 else capabilities.mps_available)
    availability = "available" if available else "not available"
    raise RuntimeError(
        f"--device {requested} was requested and is {availability}, but the certified "
        f"Version-4 training backend cannot run there. "
        f"{capabilities.accelerator_ineligible_reason} Use --device auto or cpu."
    )


def write_runtime_report(path: Path, capabilities: RuntimeCapabilities,
                         *, requested_device: str, selected_device: str,
                         cpu_threads: int) -> None:
    payload = asdict(capabilities)
    payload.update({
        "requested_device": requested_device,
        "selected_device": selected_device,
        "cpu_threads": int(cpu_threads),
        "argv": sys.argv,
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
