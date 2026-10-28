"""Build (on first use) and load the two native libraries in v4/native/.

    exact_dp    PyTorch C++ extension: per-row elementary symmetric polynomials and the
                product of category polynomials over sizes, with backward passes
    sampler     plain C library (ctypes): exact sampling given z, log Z_z, the 2-D size DP

Both are written to build/native/ in the repository and rebuilt when their source changes.
The compiler is $CXX if set, else clang++, else g++.
"""
from __future__ import annotations

import ctypes
import importlib
import importlib.machinery
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "native"
# V4_BUILD_DIR keeps the builds of different Python installs apart (they share extension suffixes).
BUILD = Path(os.environ["V4_BUILD_DIR"]) if os.environ.get("V4_BUILD_DIR") else HERE.parent / "build" / "native"

_exact_dp = None
_sampler = None


def _stale(target: Path, source: Path) -> bool:
    return not target.exists() or target.stat().st_mtime < source.stat().st_mtime


def exact_dp():
    """The exact_dp extension module."""
    global _exact_dp
    if _exact_dp is None:
        BUILD.mkdir(parents=True, exist_ok=True)
        # Only a build for this Python (its extension suffix, e.g. .cpython-313-darwin.so).
        built = BUILD / f"v4_exact_dp{importlib.machinery.EXTENSION_SUFFIXES[0]}"
        if _stale(built, SOURCE / "exact_dp.cpp"):
            subprocess.run([sys.executable, str(SOURCE / "setup_exact_dp.py"), "build_ext",
                            "--build-lib", str(BUILD), "--build-temp", str(BUILD / "temp")],
                           check=True, cwd=SOURCE, stdout=subprocess.DEVNULL)
        if str(BUILD) not in sys.path:
            sys.path.insert(0, str(BUILD))
        _exact_dp = importlib.import_module("v4_exact_dp")
    return _exact_dp


def sampler():
    """The sampler library with its argument types declared."""
    global _sampler
    if _sampler is None:
        BUILD.mkdir(parents=True, exist_ok=True)
        ext = ".dylib" if platform.system() == "Darwin" else ".so"
        lib = BUILD / f"libsampler{ext}"
        if _stale(lib, SOURCE / "sampler.cpp"):
            flags = ["-O3", "-std=c++17", "-shared", "-fPIC",
                     "-mcpu=native" if platform.machine() in ("arm64", "aarch64") else "-march=native"]
            compiler = os.environ.get("CXX") or shutil.which("clang++") or shutil.which("g++")
            if not compiler:
                raise RuntimeError("no C++ compiler found; install clang++ or g++, or set CXX")
            subprocess.run([compiler, *flags, str(SOURCE / "sampler.cpp"), "-o", str(lib), "-pthread"], check=True)
        so = ctypes.CDLL(str(lib))
        I, U, D, P = ctypes.c_int64, ctypes.c_uint64, ctypes.c_double, ctypes.c_void_p
        so.sample_slots_cpp.argtypes = [I] * 8 + [U] + [P] * 12
        so.log_partition_cpp.argtypes = [I] * 8 + [P] * 13
        so.log_degree_2d_forward_cpp.argtypes = [I] * 4 + [P] * 7
        so.log_degree_2d_grad_cpp.argtypes = [I] * 4 + [P] * 5
        for f in (so.sample_slots_cpp, so.log_partition_cpp, so.log_degree_2d_forward_cpp,
                  so.log_degree_2d_grad_cpp):
            f.restype = ctypes.c_int
        _sampler = so
    return _sampler


def ptr(tensor):
    """Data pointer of a contiguous tensor, or 0 (null) for None."""
    return 0 if tensor is None else tensor.data_ptr()


def check(status, name):
    if status != 0:
        raise RuntimeError(f"native {name} failed with status {status}")


# ---- autograd wrappers ----------------------------------------------------------------------
class RowESP(torch.autograd.Function):
    """log e_0..e_R of each row of log weights [rows, items] (rows padded with -inf)."""

    @staticmethod
    def forward(ctx, log_weights, lengths, R, block_size):
        value, weights, boundary = exact_dp().esp_blocked_log_forward(
            log_weights.contiguous(), lengths.contiguous(), int(R), int(block_size))
        ctx.save_for_backward(weights, lengths, boundary)
        ctx.R, ctx.block_size = int(R), int(block_size)
        return value

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad):
        weights, lengths, boundary = ctx.saved_tensors
        return (exact_dp().esp_blocked_log_backward(grad.contiguous(), weights, lengths, boundary,
                                                    ctx.R, ctx.block_size), None, None, None)


class RowESPRagged(torch.autograd.Function):
    """log e_0..e_R of each ragged row of log weights [T] given row_start [n_rows+1] and row_size [n_rows]."""

    @staticmethod
    def forward(ctx, log_weights, row_start, row_size, R):
        value, weights = exact_dp().esp_ragged_log_forward(
            log_weights.contiguous(), row_start.contiguous(), row_size.contiguous(), int(R))
        ctx.save_for_backward(weights, row_start, row_size)
        ctx.R = int(R)
        return value

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad):
        weights, row_start, row_size = ctx.saved_tensors
        return (exact_dp().esp_ragged_log_backward(grad.contiguous(), weights, row_start, row_size,
                                                   ctx.R), None, None, None)


class CategoryProduct(torch.autograd.Function):
    """log coefficients over total size n of the product of category polynomials.
    log_polys [1, B, Cpad, R+1], degree [B, Cpad] -> [1, B, nmax+1]."""

    @staticmethod
    def forward(ctx, log_polys, degree, nmax):
        value, normalized, prefix, normalized_exp, prefix_exp = exact_dp().log_degree_forward(
            log_polys.contiguous(), degree.contiguous(), int(nmax))
        ctx.save_for_backward(normalized, degree, prefix, normalized_exp, prefix_exp)
        ctx.nmax = int(nmax)
        return value

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad):
        normalized, degree, prefix, normalized_exp, prefix_exp = ctx.saved_tensors
        return (exact_dp().log_degree_backward(grad.contiguous(), normalized, degree, prefix,
                                               normalized_exp, prefix_exp, ctx.nmax), None, None)


class CategoryProduct2D(torch.autograd.Function):
    """log coefficients over (size n, categories present m) of the product of category
    polynomials.  log_polys [B, Cpad, R+1], degree [B, Cpad] -> [B, nmax+1, nmax+1].

    The backward pass needs the DP's intermediate tables, B x (Cpad+1) x (nmax+1)^2 numbers
    (about 9 GB for 256 Dunnhumby trips).  So the forward pass keeps only its inputs, and the
    backward pass (log_degree_2d_grad_cpp) recomputes them per trip from a table kept every
    sqrt(Cpad) categories, one segment at a time, and only for the sizes the incoming gradient
    weighs (all but TAIL of its weight)."""

    TAIL = 1e-13

    @staticmethod
    def forward(ctx, log_polys, degree, nmax):
        lp, deg = log_polys.contiguous(), degree.contiguous()
        B, Cpad, stride = lp.shape
        out = torch.empty((B, nmax + 1, nmax + 1))
        check(sampler().log_degree_2d_forward_cpp(B, Cpad, stride, nmax, ptr(lp), ptr(deg),
                                                  ptr(out), 0, 0, 0, 0), "2-D DP")
        ctx.save_for_backward(lp, deg)
        ctx.nmax = nmax
        return out

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad):
        lp, deg = ctx.saved_tensors
        B, Cpad, stride = lp.shape
        g = grad.contiguous()
        # Per trip, the largest size n whose rows n.. still carry more than TAIL of the gradient's
        # total weight; the backward runs on sizes <= that cap (exact there; what is dropped is at
        # most TAIL of the gradient).
        rows = g.abs().sum(2)
        tail = rows.flip(1).cumsum(1).flip(1)
        caps = ((tail > CategoryProduct2D.TAIL * rows.sum(1, keepdim=True)).sum(1) - 1).clamp(0, ctx.nmax).contiguous()
        grad_in = torch.empty_like(lp)
        check(sampler().log_degree_2d_grad_cpp(B, Cpad, stride, ctx.nmax, ptr(lp), ptr(deg), ptr(g), ptr(caps),
                                               ptr(grad_in)), "2-D DP backward")
        return grad_in, None, None
