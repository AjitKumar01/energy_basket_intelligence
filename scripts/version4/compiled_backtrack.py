"""Compiled exact reverse sampler for Version-4 conditional basket laws.

Given the forward tables of an exact S | z law (tempered_block_gibbs.conditional_log_tables_levels:
category log polynomials log_g, centred slot log weights, and the nonempty size law), a basket is
drawn exactly by the probability chain rule:

  1. size n ~ size law;
  2. category counts by reverse sampling through log-domain prefix products of the category
     polynomials: P(t_c = t | remaining) oc g_c[t] * prefix_{c-1}[remaining - t];
  3. within each category, the t chosen slots by exact sequential conditional Bernoulli using the
     suffix elementary-symmetric table of the slots' log weights (the category penalty is constant
     at fixed count, so it cancels here).

This is the same law as the NumPy reverse samplers in tempered_block_gibbs.py; the loops are
compiled (numba), in log space throughout, with no rejection step.  The pure-Python samplers stay the
reference implementation.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from numba import njit, prange


@njit(cache=True)
def _logaddexp(a, b):
    if a == -np.inf:
        return b
    if b == -np.inf:
        return a
    m = a if a > b else b
    return m + math.log(math.exp(a - m) + math.exp(b - m))


@njit(cache=True)
def _categorical(logw, n):
    top = -np.inf
    for i in range(n):
        if logw[i] > top:
            top = logw[i]
    total = 0.0
    for i in range(n):
        if logw[i] > -np.inf:
            total += math.exp(logw[i] - top)
    u = np.random.random() * total
    acc = 0.0
    last = -1
    for i in range(n):
        if logw[i] > -np.inf:
            acc += math.exp(logw[i] - top)
            last = i
            if u < acc:
                return i
    return last


@njit(cache=True)
def _backtrack(lg, cw, ls, trip_rows_ptr, trip_rows, row_start, row_end, draws, seed):
    np.random.seed(seed)
    B = ls.shape[0]
    nmax = ls.shape[1]
    width = lg.shape[1]                      # R + 1
    total_cap = B * draws * nmax
    out_slot = np.empty(total_cap, dtype=np.int64)
    out_basket = np.empty(total_cap, dtype=np.int64)
    count = 0
    for b in range(B):
        r0, r1 = trip_rows_ptr[b], trip_rows_ptr[b + 1]
        m = r1 - r0
        # log prefix products over this trip's categories, degrees 0..nmax
        prefix = np.full((m + 1, nmax + 1), -np.inf)
        prefix[0, 0] = 0.0
        for c in range(m):
            row = trip_rows[r0 + c]
            size = row_end[row] - row_start[row]
            deg = min(size, width - 1)
            for k in range(nmax + 1):
                acc = -np.inf
                for t in range(min(k, deg) + 1):
                    g = lg[row, t]
                    p = prefix[c, k - t]
                    if g > -np.inf and p > -np.inf:
                        acc = _logaddexp(acc, g + p)
                prefix[c + 1, k] = acc
        weights = np.empty(nmax + 1)
        for d in range(draws):
            n = _categorical(ls[b], nmax) + 1
            left = n
            basket_id = d * B + b
            for c in range(m - 1, -1, -1):
                if left == 0:
                    break
                row = trip_rows[r0 + c]
                size = row_end[row] - row_start[row]
                deg = min(size, width - 1)
                hi = min(left, deg)
                for t in range(hi + 1):
                    g = lg[row, t]
                    p = prefix[c, left - t]
                    weights[t] = g + p if (g > -np.inf and p > -np.inf) else -np.inf
                take = _categorical(weights, hi + 1)
                left -= take
                if take == 0:
                    continue
                s0 = row_start[row]
                if take == size:
                    for i in range(size):
                        out_slot[count] = s0 + i
                        out_basket[count] = basket_id
                        count += 1
                    continue
                # suffix[i, k] = log e_k(logits[i:]) for k <= take
                suffix = np.full((size + 1, take + 1), -np.inf)
                for i in range(size, -1, -1):
                    suffix[i, 0] = 0.0
                for i in range(size - 1, -1, -1):
                    for k in range(1, take + 1):
                        suffix[i, k] = _logaddexp(suffix[i + 1, k], cw[s0 + i] + suffix[i + 1, k - 1])
                need = take
                for i in range(size):
                    if need == 0:
                        break
                    if size - i == need:
                        for j in range(i, size):
                            out_slot[count] = s0 + j
                            out_basket[count] = basket_id
                            count += 1
                        need = 0
                        break
                    log_include = cw[s0 + i] + suffix[i + 1, need - 1] - suffix[i, need]
                    if log_include > 0.0:
                        log_include = 0.0
                    if np.random.random() < math.exp(log_include):
                        out_slot[count] = s0 + i
                        out_basket[count] = basket_id
                        count += 1
                        need -= 1
            if left != 0:
                return out_slot[:0], out_basket[:0], -1
    return out_slot[:count], out_basket[:count], count


@njit(cache=True, inline="always")
def _uniform(state):
    """splitmix64 step on a one-element uint64 array; returns a double in [0, 1)."""
    state[0] += np.uint64(0x9E3779B97F4A7C15)
    z = state[0]
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    z = z ^ (z >> np.uint64(31))
    return (z >> np.uint64(11)) * (1.0 / 9007199254740992.0)


@njit(cache=True)
def _categorical_s(logw, n, state):
    top = -np.inf
    for i in range(n):
        if logw[i] > top:
            top = logw[i]
    total = 0.0
    for i in range(n):
        if logw[i] > -np.inf:
            total += math.exp(logw[i] - top)
    u = _uniform(state) * total
    acc = 0.0
    last = -1
    for i in range(n):
        if logw[i] > -np.inf:
            acc += math.exp(logw[i] - top)
            last = i
            if u < acc:
                return i
    return last


@njit(cache=True, parallel=True)
def _backtrack_parallel(lg, cw, ls, trip_rows_ptr, trip_rows, row_start, row_end, draws, seed):
    """Same law as _backtrack; trips run in parallel, each with its own splitmix64 stream
    (seeded by seed and the trip index), so the output does not depend on the thread count."""
    B = ls.shape[0]
    nmax = ls.shape[1]
    width = lg.shape[1]
    per_trip = draws * nmax
    out_slot = np.empty(B * per_trip, dtype=np.int64)
    out_basket = np.empty(B * per_trip, dtype=np.int64)
    filled = np.zeros(B, dtype=np.int64)
    failed = np.zeros(B, dtype=np.int64)
    for b in prange(B):
        state = np.empty(1, dtype=np.uint64)
        state[0] = np.uint64(seed) * np.uint64(0x100000001B3) + np.uint64(b)
        _uniform(state)
        base = b * per_trip
        count = 0
        r0, r1 = trip_rows_ptr[b], trip_rows_ptr[b + 1]
        m = r1 - r0
        prefix = np.full((m + 1, nmax + 1), -np.inf)
        prefix[0, 0] = 0.0
        for c in range(m):
            row = trip_rows[r0 + c]
            size = row_end[row] - row_start[row]
            deg = min(size, width - 1)
            for k in range(nmax + 1):
                acc = -np.inf
                for t in range(min(k, deg) + 1):
                    g = lg[row, t]
                    p = prefix[c, k - t]
                    if g > -np.inf and p > -np.inf:
                        acc = _logaddexp(acc, g + p)
                prefix[c + 1, k] = acc
        weights = np.empty(nmax + 1)
        for d in range(draws):
            n = _categorical_s(ls[b], nmax, state) + 1
            left = n
            basket_id = d * B + b
            for c in range(m - 1, -1, -1):
                if left == 0:
                    break
                row = trip_rows[r0 + c]
                size = row_end[row] - row_start[row]
                deg = min(size, width - 1)
                hi = min(left, deg)
                for t in range(hi + 1):
                    g = lg[row, t]
                    p = prefix[c, left - t]
                    weights[t] = g + p if (g > -np.inf and p > -np.inf) else -np.inf
                take = _categorical_s(weights, hi + 1, state)
                left -= take
                if take == 0:
                    continue
                s0 = row_start[row]
                if take == size:
                    for i in range(size):
                        out_slot[base + count] = s0 + i
                        out_basket[base + count] = basket_id
                        count += 1
                    continue
                suffix = np.full((size + 1, take + 1), -np.inf)
                for i in range(size, -1, -1):
                    suffix[i, 0] = 0.0
                for i in range(size - 1, -1, -1):
                    for k in range(1, take + 1):
                        suffix[i, k] = _logaddexp(suffix[i + 1, k], cw[s0 + i] + suffix[i + 1, k - 1])
                need = take
                for i in range(size):
                    if need == 0:
                        break
                    if size - i == need:
                        for j in range(i, size):
                            out_slot[base + count] = s0 + j
                            out_basket[base + count] = basket_id
                            count += 1
                        need = 0
                        break
                    log_include = cw[s0 + i] + suffix[i + 1, need - 1] - suffix[i, need]
                    if log_include > 0.0:
                        log_include = 0.0
                    if _uniform(state) < math.exp(log_include):
                        out_slot[base + count] = s0 + i
                        out_basket[base + count] = basket_id
                        count += 1
                        need -= 1
            if left != 0:
                failed[b] = 1
        filled[b] = count
    total = 0
    for b in range(B):
        total += filled[b]
    slot = np.empty(total, dtype=np.int64)
    basket = np.empty(total, dtype=np.int64)
    at = 0
    for b in range(B):
        base = b * per_trip
        for i in range(filled[b]):
            slot[at] = out_slot[base + i]
            basket[at] = out_basket[base + i]
            at += 1
    return slot, basket, -1 if failed.sum() > 0 else total


def compiled_draws(log_g, centred, log_size, ix, draws: int, generator: torch.Generator,
                   parallel: bool = True):
    """Exact reverse draws from one forward table (level 0): returns flat (slot, basket) arrays,
    basket = draw * B + trip."""
    lg = log_g[0].detach().cpu().numpy().astype(np.float64)
    cw = centred[0].detach().cpu().numpy().astype(np.float64)
    ls = log_size[0].detach().cpu().numpy().astype(np.float64)
    row_trip = ix.row_trip.detach().cpu().numpy()
    row_size = ix.row_size.detach().cpu().numpy()
    row_end = np.cumsum(row_size).astype(np.int64)
    row_start = row_end - row_size
    order = np.argsort(row_trip, kind="stable").astype(np.int64)
    counts = np.bincount(row_trip, minlength=ix.B)
    ptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    seed = int(torch.randint(2**31 - 1, (), generator=generator))
    kernel = _backtrack_parallel if parallel else _backtrack
    slot, basket, status = kernel(lg, cw, ls, ptr, order, row_start, row_end, int(draws), seed)
    if status < 0:
        raise RuntimeError("compiled reverse sampler left slots unfilled")
    return torch.as_tensor(slot), torch.as_tensor(basket)
