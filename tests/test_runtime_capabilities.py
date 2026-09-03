import dataclasses
import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "runtime_capabilities", ROOT / "scripts" / "runtime_capabilities.py")
runtime = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = runtime
SPEC.loader.exec_module(runtime)


def capability(**overrides):
    base = runtime.detect_runtime()
    return dataclasses.replace(base, **overrides)


def test_thread_policy_is_bounded_and_uses_small_machines_fully():
    assert runtime.recommended_cpu_threads(1) == 1
    assert runtime.recommended_cpu_threads(4) == 4
    assert runtime.recommended_cpu_threads(16) == 8
    assert runtime.recommended_cpu_threads(128) == 8


def test_auto_keeps_exact_cpu_backend_even_when_cuda_is_visible():
    caps = capability(cuda_available=True, cuda_device_count=1)
    assert runtime.resolve_backend("auto", caps) == "cpu"


def test_forced_accelerator_fails_instead_of_changing_normalizer():
    caps = capability(cuda_available=True, cuda_device_count=1)
    with pytest.raises(RuntimeError, match="exact float64 ESP"):
        runtime.resolve_backend("cuda", caps)


def test_runtime_report_records_decision(tmp_path):
    output = tmp_path / "runtime.json"
    caps = capability()
    runtime.write_runtime_report(
        output, caps, requested_device="auto", selected_device="cpu", cpu_threads=3)
    contents = output.read_text()
    assert '"selected_device": "cpu"' in contents
    assert '"cpu_threads": 3' in contents
