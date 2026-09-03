"""Tempered blocked-Gibbs negative phase for the unchanged version-4 model.

This is an estimator prototype, not a training entry point.  It uses the exact
Hubbard--Stratonovich augmentation

    z | S, beta ~ N(sqrt(beta) * sum_{j in S} phi_j, I)

and samples ``S | z, beta`` exactly with the same category/cardinality dynamic
program used by the likelihood.  Parallel tempering acts only on the marginal
basket states.  At beta=0 the basket is an exact independent refresh from the
model with the Gram interaction removed; no local chain is needed there.

The beta=1 replica therefore targets the original version-4 basket law.  Beta is
an algorithmic bridge, not a parameter or alteration of that law.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import torch

from ragged import esp_log_bucketed, seg_max
from poly_degree_native import log_poly_tree_degree_native


@dataclass
class TemperedStep:
    states: List[List[torch.Tensor]]
    # Number of accepted exchanges on each edge/trip.  Dividing by
    # ``exchange_sweeps`` gives the usual acceptance rate.
    swap_accept: torch.Tensor
    sizes: torch.Tensor
    interaction: torch.Tensor


def _numpy_log_prefix(log_polys: Sequence[np.ndarray], degree: int
                      ) -> List[np.ndarray]:
    """Category-polynomial prefixes in log coordinates.

    The optimized reverse samplers normally use scaled probability-space convolutions.
    For a very sharp conditional law, a coefficient at the sampled total degree can
    underflow even though its log coefficient is finite.  This exact log-domain path is
    constructed lazily only for such a reverse step.
    """
    degree = int(degree)
    prefix = [np.array([0.0], dtype=np.float64)]
    for values in log_polys:
        previous = prefix[-1]
        width = min(degree, previous.size + values.size - 2) + 1
        nxt = np.full(width, -np.inf, dtype=np.float64)
        for take in range(min(values.size, width)):
            keep = min(previous.size, width - take)
            if keep and np.isfinite(values[take]):
                segment = previous[:keep] + float(values[take])
                nxt[take:take + keep] = np.logaddexp(
                    nxt[take:take + keep], segment)
        prefix.append(nxt)
    return prefix


def _numpy_log_reverse_weights(log_poly: np.ndarray, log_prefix: np.ndarray,
                               left: int) -> np.ndarray:
    """Exact log weights for one category-count reverse decision."""
    hi = min(int(left), log_poly.size - 1)
    take = np.arange(hi + 1)
    keep = int(left) - take
    answer = np.full(hi + 1, -np.inf, dtype=np.float64)
    valid = keep < log_prefix.size
    answer[valid] = log_poly[take[valid]] + log_prefix[keep[valid]]
    return answer


def _numpy_backtrack(log_g: torch.Tensor, centred: torch.Tensor,
                     log_size: torch.Tensor, ix, generator
                     ) -> List[List[torch.Tensor]]:
    """Fast detached reverse sampler for the already-computed exact DP.

    Category prefix convolutions run in optimized NumPy, and an item subset of
    fixed size is drawn by exact conditional-Bernoulli rejection.  The latter is
    valid because conditioning independent Bernoullis on their sum cancels the
    common odds shift and leaves probability proportional to the product of item
    weights.
    """
    dev = log_g.device
    seed = int(torch.randint(2**63 - 1, (), generator=generator, device=dev))
    rng = np.random.default_rng(seed)
    lg = log_g.detach().cpu().numpy()
    cw = centred.detach().cpu().numpy()
    ls = log_size.detach().cpu().numpy()
    row_trip = ix.row_trip.detach().cpu().numpy()
    row_size = ix.row_size.detach().cpu().numpy()
    row_end = np.cumsum(row_size, dtype=np.int64)
    row_start = row_end - row_size
    trip_order = np.argsort(row_trip, kind="stable")
    trip_count = np.bincount(row_trip, minlength=ix.B)
    trip_end = np.cumsum(trip_count, dtype=np.int64)
    trip_start = trip_end - trip_count
    L = lg.shape[0]

    def categorical_log(v):
        finite = np.isfinite(v)
        if not finite.any():
            raise RuntimeError("reverse sampler received an empty categorical law")
        p = np.zeros_like(v, dtype=np.float64)
        top = float(np.max(v[finite]))
        p[finite] = np.exp(v[finite] - top)
        p /= p.sum()
        return int(rng.choice(len(p), p=p))

    result: List[List[torch.Tensor]] = []
    for level in range(L):
        level_out = []
        for b in range(ix.B):
            n = categorical_log(ls[level, b]) + 1
            rows = trip_order[trip_start[b]:trip_end[b]]
            polys = []
            log_polys = []
            pref = [np.ones(1, dtype=np.float64)]
            for row in rows:
                x = lg[level, row, :n + 1]
                log_polys.append(np.asarray(x, dtype=np.float64))
                finite = np.isfinite(x)
                g = np.zeros_like(x, dtype=np.float64)
                if finite.any():
                    g[finite] = np.exp(x[finite] - float(np.max(x[finite])))
                polys.append(g)
                nxt = np.convolve(pref[-1], g)[:n + 1]
                mx = float(nxt.max(initial=0.0))
                if mx > 0:
                    nxt /= mx
                pref.append(nxt)

            chosen = []
            left = n
            log_pref = None
            for c in range(len(rows) - 1, -1, -1):
                if left == 0:
                    break
                g, pfx = polys[c], pref[c]
                hi = min(left, len(g) - 1)
                take_axis = np.arange(hi + 1)
                valid = left - take_axis < len(pfx)
                prob = np.zeros(hi + 1, dtype=np.float64)
                prob[valid] = g[take_axis[valid]] * pfx[left - take_axis[valid]]
                total = float(prob.sum())
                if total <= np.sqrt(np.finfo(np.float64).tiny) or not np.isfinite(total):
                    if log_pref is None:
                        log_pref = _numpy_log_prefix(log_polys, n)
                    log_probability = _numpy_log_reverse_weights(
                        log_polys[c], log_pref[c], left)
                    take = categorical_log(log_probability)
                else:
                    take = int(rng.choice(hi + 1, p=prob / total))
                if take:
                    row = rows[c]
                    slots = np.arange(row_start[row], row_end[row], dtype=np.int64)
                    logits = cw[level, slots]
                    if take == len(slots):
                        selected = np.ones(len(slots), dtype=bool)
                    else:
                        lo, hi_shift = -80.0, 80.0
                        for _ in range(50):
                            mid = 0.5 * (lo + hi_shift)
                            q = 1.0 / (1.0 + np.exp(-np.clip(logits + mid, -745, 709)))
                            if float(q.sum()) < take:
                                lo = mid
                            else:
                                hi_shift = mid
                        q = 1.0 / (1.0 + np.exp(
                            -np.clip(logits + 0.5 * (lo + hi_shift), -745, 709)))
                        # At the mean-matched shift the probability of the requested
                        # count is O(1/sqrt(take)); rejection is short in this use case.
                        for _ in range(10000):
                            selected = rng.random(len(q)) < q
                            if int(selected.sum()) == take:
                                break
                        else:
                            raise RuntimeError("conditional-Bernoulli rejection did not mix")
                    chosen.extend(slots[selected].tolist())
                left -= take
            if left:
                raise RuntimeError("category reverse sampler left slots unfilled")
            level_out.append(torch.as_tensor(sorted(chosen), dtype=torch.long, device=dev))
        result.append(level_out)
    return result


def _numpy_backtrack_repeated(log_g: torch.Tensor, centred: torch.Tensor,
                              log_size: torch.Tensor, ix, draws: int, generator
                              ) -> List[List[torch.Tensor]]:
    """Many independent reverse draws from one already-computed conditional DP.

    Unlike ``_numpy_backtrack``, this path constructs category prefix polynomials once per
    trip and solves each within-category conditional-Bernoulli law once per requested
    count.  Draws sharing a ``(category,count)`` pair are generated in one NumPy block.
    This removes the dominant repeated work in exact-base importance sampling.
    """
    if log_g.shape[0] != 1 or centred.shape[0] != 1 or log_size.shape[0] != 1:
        raise ValueError("repeated backtracking expects one forward conditional table")
    draws = int(draws)
    if draws < 1:
        raise ValueError("draws must be positive")
    dev = log_g.device
    seed = int(torch.randint(2**63 - 1, (), generator=generator, device=dev))
    rng = np.random.default_rng(seed)
    lg = log_g[0].detach().cpu().numpy()
    cw = centred[0].detach().cpu().numpy()
    ls = log_size[0].detach().cpu().numpy()
    row_trip = ix.row_trip.detach().cpu().numpy()
    row_size = ix.row_size.detach().cpu().numpy()
    row_end = np.cumsum(row_size, dtype=np.int64)
    row_start = row_end - row_size
    trip_order = np.argsort(row_trip, kind="stable")
    trip_count = np.bincount(row_trip, minlength=ix.B)
    trip_end = np.cumsum(trip_count, dtype=np.int64)
    trip_start = trip_end - trip_count
    result: List[List[torch.Tensor | None]] = [
        [None for _ in range(ix.B)] for _ in range(draws)]

    def probabilities_from_log(values):
        finite = np.isfinite(values)
        if not finite.any():
            raise RuntimeError("reverse sampler received an empty categorical law")
        p = np.zeros_like(values, dtype=np.float64)
        top = float(np.max(values[finite]))
        p[finite] = np.exp(values[finite] - top)
        total = float(p.sum())
        if total <= 0 or not np.isfinite(total):
            raise RuntimeError("reverse sampler received non-finite probabilities")
        return p / total

    for b in range(ix.B):
        size_probability = probabilities_from_log(ls[b])
        sizes = rng.choice(len(size_probability), size=draws,
                           p=size_probability).astype(np.int64) + 1
        rows = trip_order[trip_start[b]:trip_end[b]]
        max_n = int(sizes.max())
        polys = []
        log_polys = []
        pref = [np.ones(max_n + 1, dtype=np.float64)]
        pref[0][1:] = 0.0
        row_slots = []
        for row in rows:
            values = lg[row, :max_n + 1]
            log_polys.append(np.asarray(values, dtype=np.float64))
            finite = np.isfinite(values)
            g = np.zeros(max_n + 1, dtype=np.float64)
            if finite.any():
                g[finite] = np.exp(values[finite] - float(np.max(values[finite])))
            polys.append(g)
            nxt = np.convolve(pref[-1], g)[:max_n + 1]
            mx = float(nxt.max(initial=0.0))
            if mx <= 0 or not np.isfinite(mx):
                raise RuntimeError("non-finite category prefix polynomial")
            pref.append(nxt / mx)
            row_slots.append(np.arange(row_start[row], row_end[row], dtype=np.int64))

        # Reverse-sample category counts first.  They are then grouped so item subsets with
        # the same requested count share one Bernoulli shift and one vectorized draw loop.
        allocation = np.zeros((draws, len(rows)), dtype=np.int64)
        left = sizes.copy()
        log_pref = None
        for c in range(len(rows) - 1, -1, -1):
            active = np.flatnonzero(left > 0)
            for draw in active:
                hi = min(int(left[draw]), len(row_slots[c]), len(polys[c]) - 1)
                take_axis = np.arange(hi + 1)
                keep = left[draw] - take_axis
                valid = keep < len(pref[c])
                probability = np.zeros(hi + 1, dtype=np.float64)
                probability[valid] = polys[c][take_axis[valid]] * pref[c][keep[valid]]
                total = float(probability.sum())
                if total <= np.sqrt(np.finfo(np.float64).tiny) or not np.isfinite(total):
                    if log_pref is None:
                        log_pref = _numpy_log_prefix(log_polys, max_n)
                    log_probability = _numpy_log_reverse_weights(
                        log_polys[c], log_pref[c], int(left[draw]))
                    take = probabilities_from_log(log_probability)
                    take = int(rng.choice(len(take), p=take))
                else:
                    take = int(rng.choice(hi + 1, p=probability / total))
                allocation[draw, c] = take
                left[draw] -= take
        if np.any(left):
            raise RuntimeError("category reverse sampler left slots unfilled")

        chosen = [[] for _ in range(draws)]
        for c, slots in enumerate(row_slots):
            requested = allocation[:, c]
            for take in np.unique(requested[requested > 0]):
                group = np.flatnonzero(requested == take)
                take = int(take)
                if take == len(slots):
                    for draw in group:
                        chosen[draw].extend(slots.tolist())
                    continue
                logits = cw[slots]
                lo, hi_shift = -80.0, 80.0
                for _ in range(50):
                    mid = 0.5 * (lo + hi_shift)
                    q = 1.0 / (1.0 + np.exp(-np.clip(logits + mid, -745, 709)))
                    if float(q.sum()) < take:
                        lo = mid
                    else:
                        hi_shift = mid
                q = 1.0 / (1.0 + np.exp(
                    -np.clip(logits + 0.5 * (lo + hi_shift), -745, 709)))
                pending = np.arange(len(group))
                selected = np.zeros((len(slots), len(group)), dtype=bool)
                attempts = 0
                while len(pending):
                    attempts += 1
                    if attempts > 10000:
                        raise RuntimeError("conditional-Bernoulli rejection did not mix")
                    candidate = rng.random((len(slots), len(pending))) < q[:, None]
                    good = candidate.sum(0) == take
                    if good.any():
                        accepted_columns = pending[good]
                        selected[:, accepted_columns] = candidate[:, good]
                        pending = pending[~good]
                for column, draw in enumerate(group):
                    chosen[draw].extend(slots[selected[:, column]].tolist())

        for draw in range(draws):
            result[draw][b] = torch.as_tensor(
                sorted(chosen[draw]), dtype=torch.long, device=dev)

    # Every entry is filled above; the annotation uses Optional only while constructing.
    return result  # type: ignore[return-value]


@torch.no_grad()
def conditional_log_tables_levels(model, ix, z: torch.Tensor,
                                  betas: Sequence[float]):
    """Stable exact log tables for every ``S | z,beta`` law.

    The returned size table excludes the empty basket.  Its log-sum-exp is the
    conditional non-empty normalizer.  Keeping this calculation shared with the
    sampler prevents scalar evaluation from taking a less stable polynomial path.
    """
    beta = torch.as_tensor(betas, dtype=model.phi.dtype, device=model.phi.device)
    L = beta.numel()
    if z.shape != (L, ix.B, model.Kz):
        raise ValueError(f"z must have shape {(L, ix.B, model.Kz)}")
    if bool((beta < 0).any()) or bool((beta > 1).any()):
        raise ValueError("betas must lie in [0,1]")
    phi = model.phi[ix.item]
    projection = torch.einsum("tlk,tk->tl", z[:, ix.item_trip, :].permute(1, 0, 2), phi)
    slot_logw = (model.b_flat(ix).unsqueeze(1)
                 - 0.5 * beta.unsqueeze(0) * phi.square().sum(-1, keepdim=True)
                 + beta.sqrt().unsqueeze(0) * projection)             # [T,L]
    scale = seg_max(slot_logw.transpose(0, 1), ix.item_trip, ix.B)    # [L,B]
    centred = slot_logw.transpose(0, 1) - scale[:, ix.item_trip]      # [L,T]
    log_e = esp_log_bucketed(
        centred, ix.row_of, ix.n_rows, model.R, ix.row_size, ix.item_pos)
    r = torch.arange(model.R + 1, dtype=phi.dtype, device=phi.device)
    log_g = (log_e - model.rho_c[ix.row_cat].unsqueeze(0).unsqueeze(-1)
             * model.pair_feature(r))                                 # [L,row,R+1]

    gp = torch.full((L, ix.B * ix.Cpad, model.R + 1), -float("inf"),
                    dtype=phi.dtype, device=phi.device)
    gp[..., 0] = 0.0
    gp = gp.index_copy(1, ix.flat_slot, log_g).view(
        L, ix.B, ix.Cpad, model.R + 1)
    degree = torch.zeros(ix.B * ix.Cpad, dtype=torch.long, device=phi.device)
    degree[ix.flat_slot] = ix.row_size.clamp(max=model.R)
    degree = degree.view(ix.B, ix.Cpad)
    degree_axis = torch.arange(model.R + 1, dtype=phi.dtype, device=phi.device)
    slopes = (gp[..., 1:] / degree_axis[1:]).amax(dim=(2, 3)).clamp_min(0.0)
    tilted = gp - slopes[:, :, None, None] * degree_axis
    log_a = log_poly_tree_degree_native(
        tilted.contiguous(), degree.contiguous(), model.nmax)
    log_a = log_a + slopes.unsqueeze(-1) * degree_axis
    n_axis = torch.arange(log_a.shape[-1], dtype=phi.dtype, device=phi.device)
    log_size = (log_a + n_axis * scale.unsqueeze(-1)
                - model.rho_0()[:log_a.shape[-1]])[..., 1:]

    return log_g, centred, log_size


@torch.no_grad()
def conditional_slots_levels(model, ix, z: torch.Tensor, betas: Sequence[float],
                             generator: Optional[torch.Generator] = None
                             ) -> List[List[torch.Tensor]]:
    """Vectorized exact ``S | z,beta`` draws for every tempering level.

    All levels share one ESP and one category-polynomial call.  Only the short
    reverse sampling pass remains over level/trip pairs.
    """
    log_g, centred, log_size = conditional_log_tables_levels(model, ix, z, betas)

    return _numpy_backtrack(log_g, centred, log_size, ix, generator)


@torch.no_grad()
def conditional_slots_repeated(model, ix, z: torch.Tensor, beta: float, draws: int,
                               generator: Optional[torch.Generator] = None
                               ) -> List[List[torch.Tensor]]:
    """Draw repeatedly from one exact ``S | z,beta`` law after one forward DP.

    ``conditional_slots_levels`` intentionally vectorizes *different* latent states and
    temperatures.  At beta zero, however, every interaction-tempered particle has the same
    conditional law.  Recomputing the complete catalogue/category polynomial once per
    particle defeats the main computational advantage of the exact base distribution.

    This helper evaluates the forward tables once and reuses them for independent reverse
    draws.  The repeated views are read-only; ``_numpy_backtrack`` owns the random choices.
    """
    draws = int(draws)
    if draws < 1:
        raise ValueError("draws must be positive")
    if z.shape != (ix.B, model.Kz):
        raise ValueError(f"z must have shape {(ix.B, model.Kz)}")
    if not 0.0 <= float(beta) <= 1.0:
        raise ValueError("beta must lie in [0,1]")
    log_g, centred, log_size = conditional_log_tables_levels(
        model, ix, z.unsqueeze(0), [float(beta)])
    return _numpy_backtrack_repeated(log_g, centred, log_size, ix, draws, generator)


def _log_poly_prefix(polys: Sequence[torch.Tensor], degree: int) -> List[torch.Tensor]:
    """Prefix products in log coordinates, truncated to ``degree``."""
    if not polys:
        raise ValueError("a trip must contain at least one assortment category")
    dt, dev = polys[0].dtype, polys[0].device
    pref = [torch.zeros(1, dtype=dt, device=dev)]
    for log_g in polys:
        previous = pref[-1]
        width = min(degree, previous.numel() + log_g.numel() - 2) + 1
        nxt = torch.full((width,), -float("inf"), dtype=dt, device=dev)
        for take in range(min(log_g.numel(), width)):
            keep = min(previous.numel(), width - take)
            if keep:
                nxt[take:take + keep] = torch.logaddexp(
                    nxt[take:take + keep], previous[:keep] + log_g[take])
        pref.append(nxt)
    return pref


@torch.no_grad()
def conditional_slots(model, ix, z: torch.Tensor, beta: float,
                      generator: Optional[torch.Generator] = None) -> List[torch.Tensor]:
    """Draw exact assortment-slot baskets from ``p_beta(S | z, context)``.

    ``z`` is [B,Kz].  Returned entries contain positions in ``ix.item`` rather
    than global product ids, so duplicate store assortments remain unambiguous.
    """
    if z.shape != (ix.B, model.Kz):
        raise ValueError(f"z must have shape {(ix.B, model.Kz)}, got {tuple(z.shape)}")
    if not 0.0 <= float(beta) <= 1.0:
        raise ValueError("beta must lie in [0,1]")
    sb = math.sqrt(float(beta))
    phi = model.phi[ix.item]
    slot_logw = (model.b_flat(ix) - 0.5 * float(beta) * phi.square().sum(-1)
                 + sb * (z[ix.item_trip] * phi).sum(-1))

    # A common per-trip shift makes every item weight <= 1.  The degree-n
    # coefficient is restored by n*scale after the category convolution.
    scale = torch.full((ix.B,), -float("inf"), dtype=slot_logw.dtype,
                       device=slot_logw.device)
    scale.index_reduce_(0, ix.item_trip, slot_logw, "amax", include_self=True)
    centred = slot_logw - scale[ix.item_trip]
    log_e = esp_log_bucketed(
        centred.unsqueeze(0), ix.row_of, ix.n_rows, model.R,
        ix.row_size, ix.item_pos)[0]
    r = torch.arange(model.R + 1, dtype=slot_logw.dtype, device=slot_logw.device)
    log_g = (log_e - model.rho_c[ix.row_cat].unsqueeze(-1)
             * model.pair_feature(r))

    # Native degree-aware convolution supplies the complete size law.  Its
    # degree tilt is an exact polynomial identity, not an approximation.
    gp = torch.full((1, ix.B * ix.Cpad, model.R + 1), -float("inf"),
                    dtype=slot_logw.dtype, device=slot_logw.device)
    gp[..., 0] = 0.0
    gp = gp.index_copy(1, ix.flat_slot, log_g.unsqueeze(0)).view(
        1, ix.B, ix.Cpad, model.R + 1)
    degree = torch.zeros(ix.B * ix.Cpad, dtype=torch.long, device=slot_logw.device)
    degree[ix.flat_slot] = ix.row_size.clamp(max=model.R)
    degree = degree.view(ix.B, ix.Cpad)
    degree_axis = torch.arange(model.R + 1, dtype=slot_logw.dtype,
                               device=slot_logw.device)
    slopes = (gp[..., 1:] / degree_axis[1:]).amax(dim=(2, 3)).clamp_min(0.0)
    tilted = gp - slopes[:, :, None, None] * degree_axis
    log_a = log_poly_tree_degree_native(
        tilted.contiguous(), degree.contiguous(), model.nmax)
    log_a = log_a + slopes.unsqueeze(-1) * degree_axis
    n_axis = torch.arange(log_a.shape[-1], dtype=slot_logw.dtype,
                          device=slot_logw.device)
    log_size = (log_a[0] + n_axis * scale.unsqueeze(-1)
                - model.rho_0()[:log_a.shape[-1]])[:, 1:]

    out: List[torch.Tensor] = []
    for b in range(ix.B):
        if not bool(torch.isfinite(torch.logsumexp(log_size[b], 0))):
            raise RuntimeError(f"non-finite conditional size law for trip {b}")
        n = int(torch.multinomial(torch.softmax(log_size[b], 0), 1,
                                  generator=generator)) + 1
        rows = torch.nonzero(ix.row_trip == b, as_tuple=True)[0].tolist()
        polys = [log_g[row, :n + 1] for row in rows]
        pref = _log_poly_prefix(polys, n)

        chosen: List[int] = []
        left = n
        for c in range(len(rows) - 1, -1, -1):
            if left == 0:
                break
            log_gc, log_p = polys[c], pref[c]
            hi = min(left, log_gc.numel() - 1)
            choices = []
            for take in range(hi + 1):
                if left - take < log_p.numel():
                    choices.append(log_gc[take] + log_p[left - take])
                else:
                    choices.append(torch.full((), -float("inf"), dtype=log_p.dtype,
                                              device=log_p.device))
            take = int(torch.multinomial(torch.softmax(torch.stack(choices), 0), 1,
                                         generator=generator))
            if take:
                slots = torch.nonzero(ix.row_of == rows[c], as_tuple=True)[0]
                lw = centred[slots]
                table = torch.full((lw.numel() + 1, take + 1), -float("inf"),
                                   dtype=lw.dtype, device=lw.device)
                table[0, 0] = 0.0
                for k in range(1, lw.numel() + 1):
                    table[k] = table[k - 1]
                    table[k, 1:] = torch.logaddexp(
                        table[k - 1, 1:], lw[k - 1] + table[k - 1, :-1])
                need = take
                for k in range(lw.numel(), 0, -1):
                    if need == 0:
                        break
                    lp = lw[k - 1] + table[k - 1, need - 1] - table[k, need]
                    if float(torch.rand((), dtype=lw.dtype, device=lw.device,
                                        generator=generator)) < float(lp.exp().clamp(0, 1)):
                        chosen.append(int(slots[k - 1]))
                        need -= 1
                if need:
                    raise RuntimeError("conditional item backtrack left slots unfilled")
            left -= take
        if left:
            raise RuntimeError("conditional category backtrack left slots unfilled")
        out.append(torch.as_tensor(sorted(chosen), dtype=torch.long,
                                   device=slot_logw.device))
    return out


@torch.no_grad()
def basket_interaction(model, ix, states: Sequence[torch.Tensor]) -> torch.Tensor:
    """Version-4 Gram pair statistic for one basket per trip."""
    ans = torch.zeros(ix.B, dtype=model.phi.dtype, device=model.phi.device)
    for b, slots in enumerate(states):
        p = model.phi[ix.item[slots]]
        v = p.sum(0)
        ans[b] = 0.5 * (v.square().sum() - p.square().sum())
    return ans


@torch.no_grad()
def tempered_step(model, ix, states: List[List[torch.Tensor]],
                  betas: Sequence[float], generator: Optional[torch.Generator] = None,
                  parity: int = 0, exchange_sweeps: int = 1) -> TemperedStep:
    """One blocked update per replica followed by adjacent replica exchanges.

    A sweep visits every adjacent edge sequentially.  Alternating the direction is a
    composition of individually reversible Metropolis kernels and remains invariant for
    the joint replica law.  Unlike an even/odd half-sweep, one full sweep can transport an
    exact beta=0 refresh all the way to beta=1 when the bridges overlap.  This matters at
    fresh initialization, where exchange acceptance is deliberately close to one.
    """
    beta = torch.as_tensor(betas, dtype=model.phi.dtype, device=model.phi.device)
    if beta.ndim != 1 or beta.numel() < 2 or float(beta[0]) != 0.0 or float(beta[-1]) != 1.0:
        raise ValueError("betas must be an increasing ladder from 0 to 1")
    if not bool((beta[1:] > beta[:-1]).all()):
        raise ValueError("betas must be strictly increasing")
    if len(states) != beta.numel() or any(len(s) != ix.B for s in states):
        raise ValueError("states must be [n_temperatures][n_trips]")
    if int(exchange_sweeps) < 1:
        raise ValueError("exchange_sweeps must be positive")

    latent = []
    for level, bt in enumerate(beta.tolist()):
        means = torch.zeros(ix.B, model.Kz, dtype=model.phi.dtype,
                            device=model.phi.device)
        if bt:
            for b, slots in enumerate(states[level]):
                means[b] = math.sqrt(bt) * model.phi[ix.item[slots]].sum(0)
        latent.append(means + torch.randn(means.shape, dtype=means.dtype,
                                          device=means.device, generator=generator))
    updated = conditional_slots_levels(model, ix, torch.stack(latent), beta, generator)

    interaction = torch.stack([basket_interaction(model, ix, s) for s in updated])
    accepted = torch.zeros(beta.numel() - 1, ix.B, dtype=torch.long,
                           device=model.phi.device)
    for sweep in range(int(exchange_sweeps)):
        if (sweep + int(parity)) & 1:
            edges = range(beta.numel() - 2, -1, -1)
        else:
            edges = range(beta.numel() - 1)
        for low in edges:
            high = low + 1
            log_alpha = ((beta[high] - beta[low])
                         * (interaction[low] - interaction[high]))
            take = torch.rand(ix.B, dtype=beta.dtype, device=beta.device,
                              generator=generator).log() < log_alpha.clamp_max(0)
            accepted[low] += take.to(torch.long)
            for b in torch.nonzero(take, as_tuple=True)[0].tolist():
                updated[low][b], updated[high][b] = updated[high][b], updated[low][b]
            old_low = interaction[low].clone()
            interaction[low] = torch.where(take, interaction[high], interaction[low])
            interaction[high] = torch.where(take, old_low, interaction[high])

    sizes = torch.as_tensor([[s.numel() for s in level] for level in updated],
                            dtype=torch.long, device=model.phi.device)
    return TemperedStep(updated, accepted, sizes, interaction)
