#!/usr/bin/env python3
"""Build and load the native C++ exact conditional sampler and HS block-Gibbs kernel."""
from __future__ import annotations

import ctypes
import platform
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SRC = HERE / "stress_sampler.cpp"
EXT = ".dylib" if platform.system() == "Darwin" else ".so"
OUT = ROOT / "artifacts" / "native" / "lib" / f"libstress_sampler{EXT}"

_HANDLE = None


def build_stress_sampler(force: bool = False) -> Path:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not force and OUT.exists() and OUT.stat().st_mtime >= SRC.stat().st_mtime:
        return OUT
    flags = ["-O3", "-std=c++17", "-shared", "-fPIC"]
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        flags.append("-mcpu=native")
    elif platform.system() != "Windows":
        flags.append("-march=native")
    cmd = ["clang++", *flags, str(SRC), "-o", str(OUT)]
    subprocess.run(cmd, check=True)
    return OUT


def load_stress_sampler():
    global _HANDLE
    if _HANDLE is not None:
        return _HANDLE
    path = build_stress_sampler(force=False)
    lib = ctypes.CDLL(str(path))
    lib.sample_slots_cpp.argtypes = (
        [ctypes.c_int64] * 6 + [ctypes.c_uint64] + [ctypes.c_void_p] * 11
    )
    lib.sample_slots_cpp.restype = ctypes.c_int
    lib.gibbs_sweeps_cpp.argtypes = (
        [ctypes.c_int64] * 8 + [ctypes.c_double, ctypes.c_uint64] + [ctypes.c_void_p] * 14
    )
    lib.gibbs_sweeps_cpp.restype = ctypes.c_int
    _HANDLE = lib
    return _HANDLE


if __name__ == "__main__":
    built = build_stress_sampler(force=True)
    print(f"[build] compiled {built}")
