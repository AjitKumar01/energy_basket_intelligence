"""Build (on first use) and load the two native libraries in v4/native/.

    exact_dp    PyTorch C++ extension: per-row elementary symmetric polynomials and the
                product of category polynomials over sizes, with backward passes
    sampler     plain C library (ctypes): exact sampling, Gibbs sweeps, the 2-D size DP

Both are written to build/native/ in the repository and rebuilt when their source changes.
"""
from __future__ import annotations

import ctypes
import importlib
import platform
import subprocess
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "native"
BUILD = HERE.parent / "build" / "native"

_exact_dp = None
_sampler = None


def _stale(target: Path, source: Path) -> bool:
    return not target.exists() or target.stat().st_mtime < source.stat().st_mtime


def exact_dp():
    """The exact_dp extension module."""
    global _exact_dp
    if _exact_dp is None:
        BUILD.mkdir(parents=True, exist_ok=True)
        built = sorted(BUILD.glob("v4_exact_dp*.so"))
        if not built or _stale(built[0], SOURCE / "exact_dp.cpp"):
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
                     "-mcpu=native" if platform.machine() == "arm64" else "-march=native"]
            subprocess.run(["clang++", *flags, str(SOURCE / "sampler.cpp"), "-o", str(lib)], check=True)
        so = ctypes.CDLL(str(lib))
        I, U, D, P = ctypes.c_int64, ctypes.c_uint64, ctypes.c_double, ctypes.c_void_p
        so.sample_slots_cpp.argtypes = [I] * 7 + [U] + [P] * 12
        so.gibbs_sweeps_cpp.argtypes = [I] * 9 + [D, U] + [P] * 15
        so.log_degree_2d_forward_cpp.argtypes = [I] * 4 + [P] * 5
        so.log_degree_2d_backward_cpp.argtypes = [I] * 4 + [P] * 5
        for f in (so.sample_slots_cpp, so.gibbs_sweeps_cpp,
                  so.log_degree_2d_forward_cpp, so.log_degree_2d_backward_cpp):
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


class CategoryProduct(torch.autograd.Function):
    """log coefficients over total size n of the product of category polynomials.
    log_polys [1, B, Cpad, R+1], degree [B, Cpad] -> [1, B, nmax+1]."""

    @staticmethod
    def forward(ctx, log_polys, degree, nmax):
        value, normalized, prefix = exact_dp().log_degree_forward(
            log_polys.contiguous(), degree.contiguous(), int(nmax))
        ctx.save_for_backward(normalized, degree, prefix)
        ctx.nmax = int(nmax)
        return value

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad):
        normalized, degree, prefix = ctx.saved_tensors
        return (exact_dp().log_degree_backward(grad.contiguous(), normalized, degree, prefix,
                                               ctx.nmax), None, None)


class CategoryProduct2D(torch.autograd.Function):
    """log coefficients over (size n, categories present m) of the product of category
    polynomials.  log_polys [B, Cpad, R+1], degree [B, Cpad] -> [B, nmax+1, nmax+1].

    The backward pass needs the DP's intermediate tables, B x (Cpad+1) x (nmax+1)^2 numbers
    (about 9 GB for 256 Dunnhumby trips).  So the forward pass keeps only its inputs, and the
    backward pass recomputes the tables for ``CHUNK`` trips at a time.  Trips are independent,
    so the result is the same as keeping every table."""

    CHUNK = 16

    @staticmethod
    def forward(ctx, log_polys, degree, nmax):
        lp, deg = log_polys.contiguous(), degree.contiguous()
        B, Cpad, stride = lp.shape
        out = torch.empty((B, nmax + 1, nmax + 1))
        check(sampler().log_degree_2d_forward_cpp(B, Cpad, stride, nmax, ptr(lp), ptr(deg),
                                                  ptr(out), 0, 0), "2-D DP")
        ctx.save_for_backward(lp, deg)
        ctx.nmax = nmax
        return out

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad):
        lp, deg = ctx.saved_tensors
        B, Cpad, stride = lp.shape
        n = ctx.nmax + 1
        g = grad.contiguous()
        grad_in = torch.empty_like(lp)
        step = CategoryProduct2D.CHUNK
        states = torch.empty((min(step, B), Cpad + 1, n, n))
        for s in range(0, B, step):
            e = min(s + step, B)
            lp_c, deg_c, g_c = lp[s:e], deg[s:e], g[s:e]
            q_c, out_c, grad_c = torch.empty_like(lp_c), torch.empty((e - s, n, n)), torch.empty_like(lp_c)
            check(sampler().log_degree_2d_forward_cpp(e - s, Cpad, stride, ctx.nmax, ptr(lp_c), ptr(deg_c),
                                                      ptr(out_c), ptr(q_c), ptr(states)), "2-D DP")
            check(sampler().log_degree_2d_backward_cpp(e - s, Cpad, stride, ctx.nmax, ptr(g_c), ptr(q_c),
                                                       ptr(deg_c), ptr(states), ptr(grad_c)), "2-D DP backward")
            grad_in[s:e] = grad_c
        return grad_in, None, None


def category_product_2d(log_polys, degree, nmax):
    """CategoryProduct2D, skipping the saved states when no gradient is needed."""
    if torch.is_grad_enabled() and log_polys.requires_grad:
        return CategoryProduct2D.apply(log_polys, degree, int(nmax))
    lp, deg = log_polys.contiguous(), degree.contiguous()
    B, Cpad, stride = lp.shape
    out = torch.empty((B, nmax + 1, nmax + 1))
    check(sampler().log_degree_2d_forward_cpp(B, Cpad, stride, int(nmax), ptr(lp), ptr(deg),
                                              ptr(out), 0, 0), "2-D DP")
    return out
