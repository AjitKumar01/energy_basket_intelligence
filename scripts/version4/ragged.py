"""
The ragged kernel: the same model as core.py, without padding the item axis.

WHY THIS EXISTS.  core.py assumes every trip sees C categories of exactly P products.  A
dunnhumby store carries a median of 18 products in a category and up to 225.  Padding every
category to 225 would waste roughly 12x the arithmetic -- the same mistake that cost an
earlier branch 17.8 hours per fit against a projected 1.

WHAT IS RAGGED AND WHAT IS NOT.  Items are kept in one flat array with a row index, so a
category of 3 products costs 3 slots.  The CATEGORY axis is padded to the batch maximum,
because it is short (at most 185 non-empty categories per store), because the convolution
across categories is a sequential scan either way, and because a missing category pads with
the identity polynomial (1, 0, 0, ...) which the convolution absorbs for free.

HOW THE ELEMENTARY SYMMETRIC POLYNOMIALS ARE FORMED.  Not by the O(N R) recursion -- that
needs a per-row loop over a ragged axis.  Instead by POWER SUMS and Newton's identities,

    p_i = sum_j w_j^i          (a scatter-add: no padding, no recursion, no loop over items)

    e_1 = p1
    e_2 = (p1^2 - p2) / 2
    e_3 = (p1^3 - 3 p1 p2 + 2 p3) / 6
    e_4 = (p1^4 - 6 p1^2 p2 + 3 p2^2 + 8 p1 p3 - 6 p4) / 24

WHERE THAT CAN GO WRONG, AND THE GUARD.  Newton's identities subtract quantities of similar
size, so when one weight dominates its row the difference cancels and precision is lost.
`cancellation` returns (p1^2 - p2)/p1^2 per row; below about 1e-8 roughly half the mantissa
has gone.  validate_ragged.py measures this on the real fitted-scale weights rather than
assuming it is safe.

SCALING.  Weights are divided by the largest weight IN THE TRIP -- not in the row -- so
every category shares one scale and the factor comes back as n*log(M) inside the final
log-sum-exp over n.  A per-row scale would not survive the convolution across categories.
"""
import math

import numpy as np
import torch
from torch.nn.functional import softplus
from torch.autograd.function import once_differentiable

from adaptive_sparse import signed_log_integral


def seg_sum(vals, idx, n):
    """Sum `vals` within each segment.  vals [..., T], idx [T] -> [..., n]."""
    out = torch.zeros(vals.shape[:-1] + (n,), dtype=vals.dtype, device=vals.device)
    return out.index_add_(-1, idx, vals)


def seg_max(vals, idx, n):
    out = torch.full(vals.shape[:-1] + (n,), -float("inf"),
                     dtype=vals.dtype, device=vals.device)
    return out.index_reduce_(-1, idx, vals, "amax", include_self=True)


def _poly_mul_trunc_eager(A, G, nmax):
    """Reference subtraction-free convolution used by the fused autograd operator."""
    lead = torch.broadcast_shapes(A.shape[:-1], G.shape[:-1])
    LA, LG = A.shape[-1], G.shape[-1]
    L = min(LA + LG - 1, nmax + 1)
    out = torch.zeros(lead + (L,), dtype=A.dtype, device=A.device)
    for r in range(min(LG, L)):
        take = min(LA, L - r)
        out[..., r:r + take] = (out[..., r:r + take]
                                  + A[..., :take] * G[..., r:r + 1])
    return out


class _PolyMulTrunc(torch.autograd.Function):
    """Truncated polynomial product without a graph of thousands of CopySlices.

    PyTorch's generic autograd records every sliced update in the direct convolution.
    At the full catalogue shape those CopySlices dominate both memory traffic and backward
    time.  The derivative is another positive convolution, so save only A and G and apply
    that reverse rule explicitly.  Forward uses the reference arithmetic above verbatim.
    """

    @staticmethod
    def forward(ctx, A, G, nmax):
        ctx.save_for_backward(A, G)
        ctx.nmax = int(nmax)
        return _poly_mul_trunc_eager(A, G, ctx.nmax)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out):
        A, G = ctx.saved_tensors
        lead = torch.broadcast_shapes(A.shape[:-1], G.shape[:-1])
        LA, LG, L = A.shape[-1], G.shape[-1], grad_out.shape[-1]
        Ae = A.expand(lead + (LA,))
        Ge = G.expand(lead + (LG,))
        dA = torch.zeros_like(Ae) if ctx.needs_input_grad[0] else None
        dG = torch.zeros_like(Ge) if ctx.needs_input_grad[1] else None
        for r in range(min(LG, L)):
            take = min(LA, L - r)
            go = grad_out[..., r:r + take]
            if dA is not None:
                dA[..., :take].add_(go * Ge[..., r:r + 1])
            if dG is not None:
                dG[..., r] = (go * Ae[..., :take]).sum(-1)
        if dA is not None and dA.shape != A.shape:
            dA = dA.sum_to_size(A.shape)
        if dG is not None and dG.shape != G.shape:
            dG = dG.sum_to_size(G.shape)
        return dA, dG, None


def _poly_mul_trunc(A, G, nmax):
    return _PolyMulTrunc.apply(A, G, int(nmax))


def _esp_product_tree(P, R):
    """Product of (1 + P_i x), truncated at R, with logarithmic item depth.

    The scalar ESP recursion is optimal in arithmetic but serial in the number of items.
    A 1,774-item category therefore launches 1,774 dependent tensor operations per QMC
    block.  Polynomial multiplication is associative: a balanced tree uses about 11
    rounds, exposes parallel work within each round, and remains subtraction-free.
    """
    one = torch.ones(P.shape + (1,), dtype=P.dtype, device=P.device)
    polys = torch.cat([one, P.unsqueeze(-1)], dim=-1)               # [..., rows, item, 2]
    while polys.shape[-2] > 1:
        if polys.shape[-2] % 2:
            ident = torch.zeros(polys.shape[:-2] + (1, polys.shape[-1]),
                                dtype=P.dtype, device=P.device)
            ident[..., 0, 0] = 1.0
            polys = torch.cat([polys, ident], dim=-2)
        A, B = polys[..., 0::2, :], polys[..., 1::2, :]
        polys = _poly_mul_trunc(A, B, R)
    out = polys.squeeze(-2)
    if out.shape[-1] < R + 1:
        out = torch.nn.functional.pad(out, (0, R + 1 - out.shape[-1]))
    return out


def esp_bucketed(w, row_of, n_rows, R, row_size, item_pos, buckets=(8, 32, 96, 256),
                 parallel=False, native=False):
    """e_0..e_R per row by the STABLE O(N R) recursion, without padding everything.

    Newton's identities build e_r from power sums as an alternating sum; at order 12 or 23
    the terms reach 1e30 and cancel catastrophically -- measured log Z errors of 1e62 and
    1e264 against the dense kernel, while the e_2 cancellation diagnostic still read a
    healthy 0.6.  The recursion e_r <- e_r + w_i e_{r-1} has no subtraction at all and is
    unconditionally stable at any R, but it needs a per-row loop over items.

    So rows are BUCKETED by size and padded only to their bucket's maximum.  A padded slot
    carries weight 0, which is invisible to the recursion, so no masking is needed.  The
    historical fixed buckets ended at 256 and therefore SILENTLY returned the identity
    polynomial for larger categories.  Dunnhumby's residual affinity category has 1,774
    products, so that omitted a material part of Z.  Extend the buckets geometrically to
    the actual maximum and assert that every row was covered.
    """
    lead = w.shape[:-1]
    out = torch.zeros(lead + (n_rows, R + 1), dtype=w.dtype, device=w.device)
    out[..., 0] = 1.0
    max_size = int(row_size.max().item()) if row_size.numel() else 0
    limits = sorted({int(x) for x in buckets if int(x) > 0})
    if not limits and max_size:
        limits = [max_size]
    while limits and limits[-1] < max_size:
        limits.append(min(max_size, 2 * limits[-1]))
    covered = torch.zeros(n_rows, dtype=torch.bool, device=w.device)
    lo = 0
    for hi in limits:
        sel_r = (row_size > lo) & (row_size <= hi)
        lo = hi
        if not bool(sel_r.any()):
            continue
        covered |= sel_r
        ridx = torch.nonzero(sel_r, as_tuple=True)[0]
        loc = torch.full((n_rows,), -1, dtype=torch.long, device=w.device)
        loc[ridx] = torch.arange(len(ridx), device=w.device)
        sel_i = sel_r[row_of]
        wi = w[..., sel_i]                                          # [..., T_b]
        flat = loc[row_of[sel_i]] * hi + item_pos[sel_i]            # [T_b]
        P = torch.zeros(lead + (len(ridx) * hi,), dtype=w.dtype, device=w.device)
        P = P.index_copy(-1, flat, wi).view(lead + (len(ridx), hi))
        # The r-loop this replaces counted DOWNWARD, from min(R, i+1) to 1, precisely so
        # that e[r] += x * e[r-1] always read the PRE-update e[r-1].  That is the proof
        # that the R updates within one item are mutually independent: there is no carried
        # dependence to break, only a Python loop that was serialising them.  Held as one
        # tensor E [..., n_rows, R+1] the whole inner loop is a single shifted multiply-add.
        #
        # The item loop stays sequential -- e at item i genuinely depends on item i-1.
        #
        # Launch count, with R = 23 and item i contributing min(R, i+1) updates:
        #     hi=8      36      hi=96    1,955
        #     hi=32    483      hi=256   5,635        total 8,109 -> 392 (20.7x fewer)
        # Each op is small ([16 draws x a few thousand rows]), so this loop was bound by
        # launch overhead rather than arithmetic, which is where fewer-and-larger wins.
        # Measured at the real shape (5,436 rows, 151k slots, D=16, R=23), all three
        # bit-identical to the sequential version and to each other:
        #     sequential 0.147s    cat 0.044s    pad 0.040s    bounded-cat 0.044s
        # so the padded shift wins and is also the shortest to read.  Restoring the
        # min(R, i+1) bound bought nothing: r > i+1 is still exactly zero, so the wider
        # update multiplies zeros rather than doing wrong work, and the extra slicing costs
        # more than the arithmetic it saves.  In-place (E[..., 1:] += ...) is NOT available
        # here -- esp_bucketed sits inside the autograd path and the multiply's saved
        # tensor is the very slice the add would overwrite.
        # A row containing at most ``hi`` products has e_r == 0 for every r > hi.
        # Carrying the global R=120 axis through the 8- and 32-product buckets was pure
        # zero arithmetic and became material once complete support replaced R=23.  Work
        # only through the bucket's attainable degree, then pad the proven zeros back so
        # downstream category convolution sees the unchanged public shape.
        local_R = min(R, hi)
        if native:
            from poly_degree_native import esp_tree_native
            E = esp_tree_native(P.contiguous(), local_R)
        elif parallel and hi >= 64:
            E = _esp_product_tree(P, local_R)
        else:
            E = torch.zeros(lead + (len(ridx), local_R + 1),
                            dtype=w.dtype, device=w.device)
            E[..., 0] = 1.0
            for i in range(hi):
                x = P[..., i].unsqueeze(-1)                         # [..., n_b, 1]
                E = E + x * torch.nn.functional.pad(E[..., :-1], (1, 0))
        if local_R < R:
            E = torch.nn.functional.pad(E, (0, R - local_R))
        out = out.index_copy(-2, ridx, E)
    missing = (row_size > 0) & ~covered
    if bool(missing.any()):
        bad = torch.nonzero(missing, as_tuple=True)[0][:8].tolist()
        raise RuntimeError(f"esp_bucketed failed to cover rows {bad}; max size={max_size}")
    return out


def esp_log_bucketed(logw, row_of, n_rows, R, row_size, item_pos,
                     buckets=(8, 32, 96, 256), blocked=True):
    """Log ESP coefficients using the fused probability-adjoint native tree.

    Padded items have log weight -inf, i.e. exact zero weight.  The public result has the
    same full degree support as :func:`esp_bucketed`; impossible coefficients are -inf.
    """
    from poly_degree_native import esp_blocked_log_native, esp_tree_log_native
    lead = logw.shape[:-1]
    out = torch.full(lead + (n_rows, R + 1), -float("inf"),
                     dtype=logw.dtype, device=logw.device)
    out[..., 0] = 0.0
    max_size = int(row_size.max().item()) if row_size.numel() else 0
    limits = sorted({int(x) for x in buckets if int(x) > 0})
    if not limits and max_size:
        limits = [max_size]
    while limits and limits[-1] < max_size:
        limits.append(min(max_size, 2 * limits[-1]))
    covered = torch.zeros(n_rows, dtype=torch.bool, device=logw.device)
    lo = 0
    for hi in limits:
        sel_r = (row_size > lo) & (row_size <= hi)
        lo = hi
        if not bool(sel_r.any()):
            continue
        covered |= sel_r
        ridx = torch.nonzero(sel_r, as_tuple=True)[0]
        loc = torch.full((n_rows,), -1, dtype=torch.long, device=logw.device)
        loc[ridx] = torch.arange(len(ridx), device=logw.device)
        sel_i = sel_r[row_of]
        wi = logw[..., sel_i]
        flat = loc[row_of[sel_i]] * hi + item_pos[sel_i]
        P = torch.full(lead + (len(ridx) * hi,), -float("inf"),
                       dtype=logw.dtype, device=logw.device)
        P = P.index_copy(-1, flat, wi).view(lead + (len(ridx), hi))
        local_R = min(R, hi)
        if blocked:
            lengths = row_size.index_select(0, ridx).clamp(max=hi).contiguous()
            E = esp_blocked_log_native(P.contiguous(), lengths, local_R, block_size=32)
        else:
            E = esp_tree_log_native(P.contiguous(), local_R)
        if local_R < R:
            E = torch.nn.functional.pad(E, (0, R - local_R), value=-float("inf"))
        out = out.index_copy(-2, ridx, E)
    missing = (row_size > 0) & ~covered
    if bool(missing.any()):
        bad = torch.nonzero(missing, as_tuple=True)[0][:8].tolist()
        raise RuntimeError(f"esp_log_bucketed failed to cover rows {bad}; max size={max_size}")
    return out


def cancellation(w, row_of, n_rows):
    """(p1^2 - p2)/p1^2 per row: how much of the leading term survives the subtraction."""
    p1 = seg_sum(w, row_of, n_rows)
    p2 = seg_sum(w ** 2, row_of, n_rows)
    return (p1 ** 2 - p2) / p1.pow(2).clamp_min(1e-300)


def poly_mul_trunc(A, G, nmax):
    """Multiply polynomials along the last axis, truncating at degree nmax."""
    return _poly_mul_trunc(A, G, nmax)


def poly_tree(P, nmax):
    """Product of the polynomials along axis -2, truncated at degree nmax.

    The categories of a trip contribute independent factors and the basket-size generating
    polynomial is their product.  Multiplying them one at a time takes Cpad - 1 sequential
    steps -- 182 at a dunnhumby store -- each a separate small kernel launch, and that loop
    measured 38% of a log f evaluation.  Multiplying pairwise in a balanced tree needs
    ceil(log2(Cpad)) = 8 rounds instead, at slightly more arithmetic but far fewer launches:
    261 ms -> 96 ms on a batch of 24, agreeing with the sequential result to 4.3e-15.

    Padding to an even count uses the identity polynomial (1, 0, 0, ...), so no masking is
    needed.  This is the same trick as esp_tree in the multinomial baseline, which was
    written first and should have been applied here at the same time.
    """
    while P.shape[-2] > 1:
        C, d = P.shape[-2], P.shape[-1]
        if C % 2:
            pad = torch.zeros(P.shape[:-2] + (1, d), dtype=P.dtype, device=P.device)
            pad[..., 0] = 1.0
            P = torch.cat([P, pad], dim=-2)
            C += 1
        A, B = P[..., 0::2, :], P[..., 1::2, :]
        P = _poly_mul_trunc(A, B, nmax)
    return P[..., 0, :]


class RaggedIndex:
    """The per-batch layout.  Everything here is integer bookkeeping, no parameters.

    item      [T]        global product id of every assortment slot in the batch
    row_of    [T]        which (trip, category) row each slot belongs to
    row_trip  [n_rows]   which trip each row belongs to
    row_cat   [n_rows]   which category
    row_pos   [n_rows]   the row's position within its trip, 0..Cpad-1
    """

    def __init__(self, item, row_of, row_trip, row_cat, n_trips, device=None):
        self.item = torch.as_tensor(item, dtype=torch.long, device=device)
        self.row_of = torch.as_tensor(row_of, dtype=torch.long, device=device)
        self.row_trip = torch.as_tensor(row_trip, dtype=torch.long, device=device)
        self.row_cat = torch.as_tensor(row_cat, dtype=torch.long, device=device)
        self.n_rows = len(self.row_trip)
        self.B = n_trips
        # position of each row inside its trip
        order = torch.argsort(self.row_trip, stable=True)
        pos = torch.zeros(self.n_rows, dtype=torch.long, device=device)
        counts = torch.bincount(self.row_trip, minlength=n_trips)
        self.Cpad = int(counts.max())
        run = torch.cat([torch.zeros(1, dtype=torch.long, device=device),
                         torch.cumsum(counts, 0)[:-1]])
        pos[order] = (torch.arange(self.n_rows, device=device)
                      - run[self.row_trip[order]])
        self.row_pos = pos
        self.row_size = torch.bincount(self.row_of, minlength=self.n_rows)
        # position of each item inside its row, for the bucketed scatter
        starts = torch.cat([torch.zeros(1, dtype=torch.long, device=device),
                            torch.cumsum(self.row_size, 0)[:-1]])
        self.item_pos = (torch.arange(len(self.row_of), device=device)
                         - starts[self.row_of])
        # largest row size appearing at each slot position, for the degree cap
        sd = torch.zeros(self.Cpad, dtype=torch.long, device=device)
        sd.index_reduce_(0, self.row_pos, self.row_size, "amax", include_self=True)
        self.slot_deg = sd
        self.item_trip = self.row_trip[self.row_of]
        self.flat_slot = self.row_trip * self.Cpad + self.row_pos


def smolyak_grid(d, q):
    """Smolyak sparse grid for E_{z~N(0,I_d)}[g(z)], by the combination technique:

        A(q,d) = sum_{q-d+1 <= |i| <= q} (-1)^{q-|i|} C(d-1, q-|i|) (U^{i_1} x ... x U^{i_d})

    with U^i the probabilists' Gauss-Hermite rule at level i (2i-1 nodes).  Verified against
    exact enumeration at ||phi_j|| = 0.96 (phi'phi = 0.92, a grocery complement lift of 2.5):

        Kz=2, q=6:   17 nodes, error -0.005 nats     Kz=4, q=7:  201 nodes, +0.021
        Kz=3, q=7:  105 nodes, error +0.009 nats     Monte Carlo, 4096 draws: 8-36 nats

    Weights are SIGNED; the caller must sum in linear space, not by logsumexp.
    """
    import itertools
    from math import comb

    def _rule(level):
        x, w = np.polynomial.hermite_e.hermegauss(2 * level - 1)
        return x, w / math.sqrt(2 * math.pi)

    def _comps(total, k):
        if k == 1:
            yield (total,)
            return
        for first in range(1, total - k + 2):
            for rest in _comps(total - first, k - 1):
                yield (first,) + rest

    acc = {}
    for total in range(max(d, q - d + 1), q + 1):
        c = (-1) ** (q - total) * comb(d - 1, q - total)
        if c == 0:
            continue
        for idx in _comps(total, d):
            gs = [_rule(k) for k in idx]
            for combo in itertools.product(*[range(len(g[0])) for g in gs]):
                node = tuple(round(float(gs[k][0][combo[k]]), 12) for k in range(d))
                wt = c * float(np.prod([gs[k][1][combo[k]] for k in range(d)]))
                acc[node] = acc.get(node, 0.0) + wt
    nodes = np.array(list(acc.keys()), dtype=np.float64).reshape(-1, d)
    wts = np.array(list(acc.values()), dtype=np.float64)
    keep = np.abs(wts) > 1e-14
    return (torch.as_tensor(nodes[keep], dtype=torch.float64),
            torch.as_tensor(wts[keep], dtype=torch.float64))


def sparse_prepare(model, ix, degree=None, nmax=None):
    """Everything in log f that does not depend on z.

    Only phi-carrying products depend on z: for phi_j = 0 the weight is exp(b_j) whatever
    z is.  With a mask of ~20 products out of 5,455 that is 99.6% of the elementary
    symmetric polynomial work, currently recomputed once per quadrature node.  The scale is
    taken z-INDEPENDENTLY (seg_max over bt alone) so the inactive ESP can be built once;
    that is exact -- shifting the scale is compensated by the n*M term in lg -- and safe
    numerically because |phi_j'z| <= ||phi_j|| * 3.75 on the Smolyak grid.
    """
    R = model.R if degree is None else min(int(degree), model.R)
    nmax = model.nmax if nmax is None else min(int(nmax), model.nmax)
    phi_i = model.phi[ix.item]
    bt = model.b_flat(ix) - 0.5 * (phi_i ** 2).sum(-1)                 # [T]
    act = phi_i.norm(dim=-1) > 1e-12                                   # [T]
    M0 = seg_max(bt.unsqueeze(0), ix.item_trip, ix.B)                  # [1, B]
    sh = M0[0].index_select(0, ix.item_trip)                           # [T]
    w0 = torch.where(act, torch.zeros_like(bt), torch.exp(bt - sh))
    all_active = not bool((~act).any())
    if not all_active:
        e0 = esp_bucketed(w0.unsqueeze(0), ix.row_of, ix.n_rows, R,
                          ix.row_size, ix.item_pos)                    # [1, n_rows, R+1]
    else:
        # Full-catalogue phi makes every inactive polynomial the identity.  Walking all
        # 5,455 zero weights (including the 1,774-item residual row) twice per call only to
        # rediscover [1,0,...] was a sizeable fixed cost.
        e0 = torch.zeros(1, ix.n_rows, R + 1,
                         dtype=w0.dtype, device=w0.device)
        e0[..., 0] = 1.0
    ai = torch.nonzero(act, as_tuple=True)[0]                          # active slots
    arow = ix.row_of[ai]
    urow, inv = torch.unique(arow, return_inverse=True)                # rows touched
    # Dense-by-trip projection layout.  Dunnhumby assortments contain a mean 5,135 of the
    # 5,455 catalogue products, so padding each trip to its largest assortment wastes only
    # a few percent.  A batched GEMM then computes Phi z without materialising the old
    # [active_slots, nodes, Kz] multiply (multiple GiB at B=24, Kz=32).
    atrip = ix.item_trip[ai]
    atpos = torch.zeros(len(ai), dtype=torch.long, device=ai.device)
    atmax = 0
    if len(ai) > 0:
        ao = torch.argsort(atrip * (len(ai) + 1) + torch.arange(len(ai), device=ai.device))
        ats = atrip[ao]
        atcnt = torch.bincount(ats, minlength=ix.B)
        ast = torch.zeros(ix.B, dtype=torch.long, device=ai.device)
        ast[1:] = torch.cumsum(atcnt, 0)[:-1]
        atpos[ao] = torch.arange(len(ai), device=ai.device) - ast[ats]
        atmax = int(atcnt.max().item())
    dense_active = bool(atmax and len(ai) >= 0.5 * ix.B * atmax)
    if dense_active:
        aflat_trip = atrip * atmax + atpos
        apad = torch.zeros(ix.B * atmax, model.Kz,
                           dtype=phi_i.dtype, device=phi_i.device)
        apad = apad.index_copy(0, aflat_trip, phi_i[ai]).view(ix.B, atmax, model.Kz)
    else:
        apad = None
    # position of each active slot within its row's active list
    ordv = torch.argsort(inv * (len(ai) + 1) + torch.arange(len(ai), device=ai.device))
    inv_s = inv[ordv]
    pos = torch.zeros(len(ai), dtype=torch.long, device=ai.device)
    if len(ai) > 0:
        start = torch.zeros(len(urow), dtype=torch.long, device=ai.device)
        cnt = torch.bincount(inv_s, minlength=len(urow))
        start[1:] = torch.cumsum(cnt, 0)[:-1]
        pos[ordv] = torch.arange(len(ai), device=ai.device) - start[inv_s]
    acnt = torch.bincount(inv, minlength=len(urow))
    kmax = int(pos.max().item()) + 1 if len(ai) > 0 else 0
    # The category convolution is 77% of log f (profiled: poly_tree 1.714s of 2.230s), and
    # a category containing no phi product has a z-INDEPENDENT polynomial.  So split
    #     A(z) = A_const  *  A_active(z)
    # and build A_const once over the ~250 inactive categories, leaving the per-node tree
    # to run over the handful that phi actually touches.
    r_ = torch.arange(R + 1, dtype=e0.dtype, device=e0.device)
    base_cat = getattr(model, "_condition_cat_count", None)
    if base_cat is None:
        # Keep the version-4 category potential in log coordinates.  Forming ``a_`` as
        # exp(-rho_c * choose(r, 2)) and immediately taking log(a_) again in the fused
        # full-catalogue path overflows at rho_c < -log(DBL_MAX)/choose(120,2) ~= -0.0994.
        # The model itself is perfectly finite there because the category coefficients
        # are normalised before convolution.  Run196 crossed that implementation boundary
        # at iteration 1376 and thereafter rejected every rho_c gradient.  ``log_a_`` is
        # the exact original potential, without either a clamp or a change of support.
        log_a_ = (-model.rho_c[ix.row_cat].unsqueeze(-1)
                  * model.pair_feature(r_))
    else:
        # Exact law of the unobserved remainder T conditional on a revealed set R.
        # Its category potential is g(|R_c| + |T_c|) - g(|R_c|), not g(|T_c|).
        # The subtracted constant is optional for probabilities but keeps coefficients
        # near their ordinary scale.  Counts beyond the original declared support remain
        # impossible after conditioning.
        base_row = base_cat[ix.row_trip, ix.row_cat].to(dtype=e0.dtype)
        delta_pair = (model.pair_feature(base_row.unsqueeze(-1) + r_)
                      - model.pair_feature(base_row).unsqueeze(-1))
        log_a_ = -model.rho_c[ix.row_cat].unsqueeze(-1) * delta_pair
        log_a_ = torch.where(
            r_ <= (R - base_row).unsqueeze(-1), log_a_,
            torch.full_like(log_a_, -float("inf")))

    # The all-products/rank-32 production path consumes log_a_ directly below and its
    # inactive polynomial is the identity, so no linear-space category potential is
    # needed.  Sparse/conditional legacy paths still require it; keeping the conversion
    # out of the full path removes the sole float64 range restriction found in run196.
    # A conditioned recommendation catalogue can have all *remaining* products active
    # while some retained category rows contain no remaining product.  An empty row has
    # ESP (1,0,...), and multiplying it by any category potential is still exactly the
    # identity polynomial.  It therefore belongs in neither half of the convolution.  The
    # previous ``len(urow) == ix.n_rows`` check sent these cases to the legacy linear path;
    # with attractive rho_c that path suffered incompatible category maxima and made MRR
    # depend on evaluation batch size.  Require coverage of every NONEMPTY row instead.
    nonempty_rows = int((ix.row_size > 0).sum())
    all_nonempty_rows_active = len(urow) == nonempty_rows
    # The bounded log-coordinate kernels apply to sparse Phi as well as dense Phi.  The
    # former restriction to ``all_active`` sent a row-masked run through the raw
    # linear-coefficient adjoint; on rare tail batches its proposal and likelihood
    # backwards formed 0*inf even though log Z itself was finite.  Split inactive and
    # active item polynomials in log coordinates below, so sparsity remains exact without
    # reintroducing that numerical range restriction.
    native_log_path = (bool(getattr(model, "_esp_native", False))
                       and bool(getattr(model, "_poly_degree_native", False)))
    native_log_blocked = bool(getattr(model, "_esp_log_blocked", True))
    a_ = None if native_log_path else torch.exp(log_a_)
    aflat = ix.flat_slot[urow] if len(ai) > 0 else ix.flat_slot[:0]
    const_identity = all_nonempty_rows_active
    log_e0 = None
    log_A_const = None
    const_degree = None
    if native_log_path:
        from poly_degree_native import log_poly_tree_degree_native
        if all_active:
            # Every product is active, so the entire inactive factor is exactly the
            # identity polynomial.  The former native path nevertheless walked every zero
            # inactive weight and multiplied every identity category once per minibatch.
            log_A_const = torch.full((1, ix.B, nmax + 1), -float("inf"),
                                     dtype=e0.dtype, device=e0.device)
            log_A_const[..., 0] = 0.0
            const_degree = torch.zeros(ix.B, dtype=torch.long, device=e0.device)
        else:
            # Build the inactive ESP once per minibatch with its bounded probability
            # adjoint. Active slots are exact zero weights (-inf in log coordinates).
            logw0 = torch.where(act, torch.full_like(bt, -float("inf")), bt - sh)
            log_e0 = esp_log_bucketed(
                logw0.unsqueeze(0), ix.row_of, ix.n_rows, R,
                ix.row_size, ix.item_pos, blocked=native_log_blocked)

            # Multiply categories untouched by Phi once. A common degree tilt is an exact
            # generating-polynomial identity and prevents incompatible category maxima.
            logG0 = log_e0 + log_a_.unsqueeze(0)
            logGc = torch.full((1, ix.B * ix.Cpad, R + 1), -float("inf"),
                               dtype=e0.dtype, device=e0.device)
            logGc[..., 0] = 0.0
            logGc = logGc.index_copy(1, ix.flat_slot, logG0)
            category_degree = torch.zeros(ix.B * ix.Cpad, dtype=torch.long,
                                          device=e0.device)
            category_degree[ix.flat_slot] = ix.row_size.clamp(max=R)
            if len(aflat) > 0:
                logGc[:, aflat, :] = -float("inf")
                logGc[:, aflat, 0] = 0.0
                category_degree[aflat] = 0
            logGc = logGc.view(1, ix.B, ix.Cpad, R + 1)
            category_degree = category_degree.view(ix.B, ix.Cpad)
            degree_axis = torch.arange(R + 1, dtype=e0.dtype, device=e0.device)
            const_slope = (logGc[..., 1:] / degree_axis[1:]).amax(dim=(2, 3))
            const_slope = const_slope.clamp_min(0.0)
            tilted_const = logGc - const_slope[:, :, None, None] * degree_axis
            log_A_const = log_poly_tree_degree_native(
                tilted_const.contiguous(), category_degree.contiguous(), nmax)
            log_A_const = log_A_const + const_slope.unsqueeze(-1) * degree_axis
            const_degree = category_degree.sum(1).clamp(max=nmax)
        A_const = None
    elif const_identity:
        # Every category is z-dependent at full catalogue coverage, hence the constant
        # half is exactly the identity.  Avoid building and tree-multiplying thousands of
        # row polynomials that are immediately overwritten by identities.
        A_const = torch.zeros(1, ix.B, nmax + 1,
                              dtype=e0.dtype, device=e0.device)
        A_const[..., 0] = 1.0
    else:
        G0 = a_.unsqueeze(0) * e0                                      # [1,n_rows,R+1]
        Gc = torch.zeros(1, ix.B * ix.Cpad, R + 1,
                         dtype=e0.dtype, device=e0.device)
        Gc[:, :, 0] = 1.0
        Gc = Gc.index_copy(1, ix.flat_slot, G0)
        if len(aflat) > 0:                   # active categories -> identity in const half
            Gc[:, aflat, :] = 0.0
            Gc[:, aflat, 0] = 1.0
        A_const = poly_tree(Gc.view(1, ix.B, ix.Cpad, R + 1), nmax)
    ab = (aflat // ix.Cpad) if len(aflat) > 0 else aflat
    if len(aflat) > 0:
        o = torch.argsort(ab * (len(ab) + 1) + torch.arange(len(ab), device=ab.device))
        ab_s = ab[o]
        cnt = torch.bincount(ab_s, minlength=ix.B)
        st = torch.zeros(ix.B, dtype=torch.long, device=ab.device)
        st[1:] = torch.cumsum(cnt, 0)[:-1]
        acol = torch.zeros(len(ab), dtype=torch.long, device=ab.device)
        acol[o] = torch.arange(len(ab), device=ab.device) - st[ab_s]
        cpad_a = int(cnt.max().item())
    else:
        acol, cpad_a = ab, 0
    active_degree = torch.zeros(ix.B, cpad_a, dtype=torch.long, device=e0.device)
    if len(urow) > 0:
        active_degree[ab, acol] = ix.row_size[urow].clamp(max=R)
    base_size = getattr(model, "_condition_size", None)
    if base_size is not None:
        base_size = base_size.to(dtype=torch.long, device=e0.device)
        if base_size.shape != (ix.B,):
            raise ValueError("conditional base size must have one entry per trip")
    return dict(bt=bt, M0=M0, sh=sh, e0=e0, log_e0=log_e0,
                ai=ai, urow=urow, inv=inv, pos=pos,
                acnt=acnt, kmax=kmax, a_row=a_, log_a_row=log_a_,
                A_const=A_const, log_A_const=log_A_const,
                const_degree=const_degree, native_log_path=native_log_path,
                native_log_blocked=native_log_blocked,
                ab=ab, acol=acol,
                cpad_a=cpad_a, atrip=atrip, atpos=atpos, atmax=atmax, apad=apad,
                active_degree=active_degree,
                inactive_identity=all_active, const_identity=const_identity, R=R,
                condition_size=base_size)


def log_f_sparse(model, z, ix, C, drop_empty=False, return_terms=False,
                 detach_params=False, nmax=None):
    """log f(z) using the z-independent cache from sparse_prepare.  Same value as
    log_f_ragged, with the ESP over non-phi products lifted out of the node loop."""
    D = z.shape[1]
    nmax = model.nmax if nmax is None else int(nmax)
    R, dt, dev = C.get("R", model.R), C["e0"].dtype, C["e0"].device
    ai, urow, inv, pos = C["ai"], C["urow"], C["inv"], C["pos"]
    node_M = C["M0"]
    logA = None
    degree_tilt = None
    # The fused log-coordinate path currently covers the production full-catalogue
    # normaliser.  Conditional complete-the-basket scoring deliberately removes revealed
    # items and therefore has a non-identity inactive polynomial; retain the exact legacy
    # algebra for that smaller evaluation-only graph.
    stable_log_native = bool(C.get("native_log_path", False))
    if len(ai) > 0:
        if C["apad"] is not None:
            apad = C["apad"].detach() if detach_params else C["apad"]
            proj_pad = torch.bmm(apad, z.transpose(1, 2))               # [B, Tmax, D]
            proj = proj_pad[C["atrip"], C["atpos"]]                    # [Ta, D]
        else:
            phi_a = model.phi[ix.item[ai]]                              # [Ta, Kz]
            if detach_params:
                phi_a = phi_a.detach()
            proj = (z[ix.item_trip[ai]] * phi_a.unsqueeze(1)).sum(-1)   # [Ta, D]
        if C["inactive_identity"]:
            # With every product active there is no cached inactive polynomial whose scale
            # must remain fixed.  Re-centre the weights separately at every (node, trip),
            # exactly as log_f_ragged does.  The former z-independent scale was safe on the
            # old radius-4 Smolyak grid but overflows at a legitimate remote mode: degree-120
            # coefficients can exceed float64 even when log f itself is only O(100).
            #
            # Scaling every weight in a trip by exp(-M) scales the degree-n coefficient by
            # exp(-nM), so adding n*M below is an exact algebraic inverse, not a clamp.
            logwa = C["bt"][ai].unsqueeze(1) + proj                    # [Ta, D]
            node_M = seg_max(logwa.transpose(0, 1), C["atrip"], ix.B) # [D, B]
            centred_logwa = logwa - node_M[:, C["atrip"]].transpose(0, 1)
            wa = None if stable_log_native else torch.exp(centred_logwa)
        else:
            centred_logwa = (C["bt"][ai].unsqueeze(1)
                              - C["sh"][ai].unsqueeze(1) + proj)
            wa = None if stable_log_native else torch.exp(centred_logwa)
        # Bucket the ACTIVE slots themselves.  The old rectangular allocation was
        # [nodes, active_rows, max_active_row_size]; with all products active, a single
        # 1,774-product residual category forced every other row to that width (34+ GiB at
        # the real batch/node shape).  This representation is proportional to the actual
        # active slots plus modest bucket padding.
        if stable_log_native:
            logEa = esp_log_bucketed(
                centred_logwa.transpose(0, 1), inv, len(urow), R, C["acnt"], pos,
                blocked=C.get("native_log_blocked", True))
        else:
            Ea = esp_bucketed(
                wa.transpose(0, 1), inv, len(urow), R, C["acnt"], pos,
                parallel=True, native=bool(getattr(model, "_esp_native", False)))
        # combined row polynomial = (inactive ESP) * (active ESP), truncated at R
        if stable_log_native:
            conv = None
        elif C["inactive_identity"]:
            # At full-catalogue coverage e0 is exactly (1,0,...), so convolution with it
            # is the identity.  The generic loop performed R+1 sliced multiply-adds for
            # every mode, curvature and QMC node block to reproduce Ea unchanged.
            conv = Ea
        else:
            base = C["e0"][0, urow]                                     # [n_arow, R+1]
            conv = torch.zeros(D, len(urow), R + 1, dtype=dt, device=dev)
            for r in range(R + 1):
                conv[..., r:] = (conv[..., r:]
                                  + base[:, r].unsqueeze(0).unsqueeze(-1)
                                  * Ea[..., : R + 1 - r])
    if stable_log_native and C["cpad_a"] > 0:
        from poly_degree_native import log_poly_tree_degree_native
        degree_axis = torch.arange(R + 1, dtype=dt, device=dev)

        if C["inactive_identity"]:
            log_combined = logEa
        else:
            # Exact within-category product
            #   ESP(inactive weights) * ESP(active weights at z).
            # Both operands and the adjoint remain in log/probability coordinates.  A
            # shared linear degree tilt is removed before convolution and restored after;
            # this changes neither coefficients nor derivatives.
            logbase = C["log_e0"][0, C["urow"]]
            pair = torch.stack([
                logbase.unsqueeze(0).expand(D, -1, -1), logEa], dim=2)
            inactive_degree = (ix.row_size[C["urow"]] - C["acnt"]).clamp(min=0, max=R)
            pair_degree = torch.stack([inactive_degree, C["acnt"].clamp(max=R)], dim=1)
            pair_slope = (pair[..., 1:] / degree_axis[1:]).amax(dim=(2, 3))
            pair_slope = pair_slope.clamp_min(0.0)
            pair_tilted = pair - pair_slope[:, :, None, None] * degree_axis
            log_combined = log_poly_tree_degree_native(
                pair_tilted.contiguous(), pair_degree.contiguous(), nmax)
            log_combined = log_combined + pair_slope.unsqueeze(-1) * degree_axis

        # sparse_prepare already has the exact category potential in log coordinates.
        # Do not round-trip it through exp/log: complete support at R=120 legitimately
        # exceeds the dynamic range of a raw coefficient while its log remains finite.
        loga = C["log_a_row"][C["urow"]]
        logGa = log_combined + loga.unsqueeze(0)
        # One constant scale per category is numerically wrong for strongly attractive
        # complete-support interactions: different categories attain their maxima at
        # mutually incompatible high counts, so multiplying their normalized degree-zero
        # coefficients underflows although the requested global degree is ordinary.
        #
        # Apply instead one LINEAR degree tilt per (node,trip):
        #   G'_c(r) = G_c(r) exp(-t r),
        #   A'_n    = A_n exp(-t n).
        # The same t for every category makes this an exact generating-polynomial identity;
        # adding n*t back below recovers the original version-4 coefficient.  Choosing t as
        # the largest finite logG_c(r)/r makes every transformed coefficient <= 1 while
        # retaining G'_c(0)=1, so neither overflow nor incompatible-max underflow occurs.
        # Include the z-independent product of inactive-only categories as one extra
        # polynomial, then multiply it with the active categories.  This retains the
        # sparse O(1) cache across Sobol nodes while giving every route the bounded native
        # adjoint.
        n_factor = C["cpad_a"] + (0 if C["const_identity"] else 1)
        offset = 0 if C["const_identity"] else 1
        logGz = torch.full((D, ix.B, n_factor, R + 1), -float("inf"),
                          dtype=dt, device=dev)
        logGz[..., 0] = 0.0
        factor_degree = torch.zeros(ix.B, n_factor, dtype=torch.long, device=dev)
        if offset:
            logGz[:, :, 0] = C["log_A_const"].expand(D, -1, -1)
            factor_degree[:, 0] = C["const_degree"]
        logGz[:, C["ab"], C["acol"] + offset] = logGa
        factor_degree[C["ab"], C["acol"] + offset] = C["active_degree"][
            C["ab"], C["acol"]]
        row_slope = (logGz[..., 1:] / degree_axis[1:]).amax(-1)
        degree_tilt = row_slope.amax(2).clamp_min(0.0)
        logGz = logGz - degree_tilt[:, :, None, None] * degree_axis
        logA = log_poly_tree_degree_native(
            logGz.contiguous(), factor_degree.contiguous(), nmax)
    elif stable_log_native and C["cpad_a"] == 0:
        logA = C["log_A_const"].expand(D, -1, -1)
    elif C["cpad_a"] == 0:
        A = C["A_const"][..., :nmax + 1].expand(D, -1, -1)
    else:
        Ga = C["a_row"][C["urow"]].unsqueeze(0) * conv          # [D, n_arow, R+1]
        Gz = torch.zeros(D, ix.B, C["cpad_a"], R + 1, dtype=dt, device=dev)
        Gz[:, :, :, 0] = 1.0
        Gz[:, C["ab"], C["acol"]] = Ga
        if bool(getattr(model, "_poly_degree_native", False)):
            # Audit-gated exact replacement for the dense category product.  Import is
            # deliberately lazy so ordinary checkpoints and environments do not depend on
            # a compiled extension until the complete forward/backward gate has passed.
            from poly_degree_native import poly_tree_degree_native
            if Gz.shape[-1] < nmax + 1:
                raise RuntimeError(
                    f"native category product received degree axis {Gz.shape[-1]} "
                    f"for requested global degree {nmax}; cache R={R}, model R={model.R}")
            A_active = poly_tree_degree_native(
                Gz.contiguous(), C["active_degree"].contiguous(), nmax)
        else:
            A_active = poly_tree(Gz, nmax)
        # Likewise A_const is exactly the identity when every category is active.  This
        # skips a 121-term polynomial convolution in the common all-5,455-products path.
        A = (A_active if C["const_identity"] else
             poly_mul_trunc(A_active, C["A_const"], nmax))
    degree_width = logA.shape[-1] if logA is not None else A.shape[-1]
    n = torch.arange(degree_width, dtype=dt, device=dev)
    rho0_all = model.rho_0()
    if detach_params:
        rho0_all = rho0_all.detach()
    base_size = C.get("condition_size")
    if base_size is None:
        rho0 = rho0_all[:degree_width]
        valid_size = None
    else:
        total_n = base_size.unsqueeze(-1) + n.to(torch.long).unsqueeze(0)
        valid_size = total_n < rho0_all.numel()
        # Work relative to the revealed basket.  The empty remainder must have score
        # exactly zero, so its contribution to the conditional completion partition is
        # one.  Omitting this constant leaves marginal probabilities unchanged but makes
        # the returned conditional log normalizer (and absolute completion log scores)
        # wrong by -rho_0(|A|).
        rho0 = (rho0_all[total_n.clamp(max=rho0_all.numel() - 1)]
                - rho0_all[base_size].unsqueeze(-1))
    coefficient_log = logA if logA is not None else torch.log(A.clamp_min(1e-300))
    global_scale = node_M if degree_tilt is None else node_M + degree_tilt
    lg = coefficient_log - rho0 + n * global_scale.unsqueeze(-1)
    if valid_size is not None:
        lg = lg.masked_fill(~valid_size.unsqueeze(0), -float("inf"))
    if drop_empty:
        lg = lg[..., 1:]
    if return_terms:
        return lg
    return torch.logsumexp(lg, dim=-1).transpose(0, 1)


def log_f_ragged(model, z, ix, drop_empty=False, return_terms=False,
                 return_parts=False):
    """log f(z) for a batch.  z [B, D, Kz] -> [B, D].

    drop_empty excludes the n = 0 term, giving f(z) - 1 directly.  That matters: the model
    conditions on a non-empty basket, so the quantity actually needed is Z - 1, and forming
    it as exp(log Z) - 1 subtracts two nearly equal numbers whenever Z is close to 1.  The
    empty basket contributes exactly 1 to f for every z (A_0 = 1, rho_0(0) = 0), so it can
    be dropped from the sum instead of subtracted afterwards -- exact, and stable however
    small Z - 1 becomes."""
    D = z.shape[1]
    phi_i = model.phi[ix.item]                                     # [T, Kz]
    bt = model.b_flat(ix) - 0.5 * (phi_i ** 2).sum(-1)             # [T]
    proj = (z[ix.item_trip] * phi_i.unsqueeze(1)).sum(-1)          # [T, D]
    logw = (bt.unsqueeze(1) + proj).transpose(0, 1)                # [D, T]
    # ONE scale per (trip, draw): a per-row scale would not survive the convolution
    M = seg_max(logw, ix.item_trip, ix.B)                          # [D, B]
    w = torch.exp(logw - M.index_select(-1, ix.item_trip))         # [D, T]
    e = esp_bucketed(w, ix.row_of, ix.n_rows, model.R, ix.row_size, ix.item_pos)
    r = torch.arange(model.R + 1, dtype=w.dtype, device=w.device)
    a = torch.exp(-model.rho_c[ix.row_cat].unsqueeze(-1) * model.pair_feature(r))
    G = a.unsqueeze(0) * e                                          # [D, n_rows, R+1]
    # scatter rows into [D, B, Cpad, R+1]; missing rows are the identity polynomial
    Gp = torch.zeros(D, ix.B * ix.Cpad, model.R + 1, dtype=w.dtype, device=w.device)
    Gp[:, :, 0] = 1.0
    Gp = Gp.index_copy(1, ix.flat_slot, G).view(D, ix.B, ix.Cpad, model.R + 1)
    # The per-slot degree cap that used to sit here claimed a 4x cut on the grounds that
    # "the median category never exceeds 3 items in any observed basket".  That reasoning
    # was wrong -- the normaliser sums over every POSSIBLE basket, so the bound is the
    # assortment's category size (up to 225 products), not the observed composition.
    # Measured, it cut 4,392 coefficients to 3,476: 1.26x, not 4x, capping 63 of 183 slots.
    # The tree supersedes it and is worth 2.7x on its own.
    A = poly_tree(Gp, model.nmax)
    n = torch.arange(A.shape[-1], dtype=w.dtype, device=w.device)
    lg = (torch.log(A.clamp_min(1e-300)) - model.rho_0()[: A.shape[-1]]
          + n * M.unsqueeze(-1))
    if return_parts:
        # Hand back the objects the sampler needs so it consumes exactly what this function
        # builds.  The sampler used to rebuild w, G and the convolution itself, and the two
        # constructions disagreed: it returned baskets of 25 where no single draw had
        # E[n|z] above 13.1.  Patching only its size stage made that worse -- size became
        # right while allocation still came from the local rebuild, and generated item
        # rates fell to 0.08x with none of the 200 commonest pairs ever appearing.  One
        # construction, consumed by every stage, is the only version that cannot drift.
        lg_out = lg[..., 1:] if drop_empty else lg
        return dict(logw=logw, M=M, w=w, G=G, Gp=Gp, A=A, lg=lg_out)
    if drop_empty:
        lg = lg[..., 1:]
    if return_terms:
        # the per-size terms, before they are summed away.  log f is a logsumexp over n, so
        # the size law is already sitting inside every normaliser evaluation; returning the
        # summands costs nothing and is what size_dist needs.
        return lg                                                   # [D, B, n]
    return torch.logsumexp(lg, dim=-1).transpose(0, 1)              # [B, D]


class RaggedModel(torch.nn.Module):
    """Same parameters and the same three quantities as core.Model, ragged over items."""

    def __init__(self, J, N, C, K=8, Kz=3, nmax=24, R=4, seed=0,
                 S=1, Kp=8, Kt=8, Ks=4, n_week=53, phi_init=0.03,
                 taste_init=0.3, household_size_rank1=False):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.J, self.N, self.C, self.S = J, N, C, S
        self.K, self.Kz, self.nmax, self.R = K, Kz, nmax, R
        if household_size_rank1 and K < 2:
            raise ValueError("household-size rank-one decomposition requires K >= 2")
        # This is a constrained coordinate system for the existing theta_h'alpha_j term,
        # not an added energy term.  The last taste coordinate is kappa_h * 1_j and the
        # remaining product loadings are catalogue-centred:
        #
        #   theta_h'alpha_j = kappa_h + theta_tilde_h'alpha_tilde_j.
        #
        # Hence kappa is rank one across the household-by-product utility surface and adds
        # n*kappa_h to a basket of size n.  It is exactly a household-specific linear
        # direction inside the original rho_0/utility gauge.
        self.household_size_rank1 = bool(household_size_rank1)
        self.lam = torch.nn.Parameter(torch.zeros(J))
        self.alpha = torch.nn.Parameter(torch.randn(J, K, generator=g) * taste_init)
        self.theta = torch.nn.Parameter(torch.randn(N, K, generator=g) * taste_init)
        # ||phi_j|| ~ phi_init * sqrt(Kz).  At the old 0.15 with Kz = 12 that was 0.52,
        # against a cap of 0.20 -- the model began 2.6x outside the region section 14
        # requires it to stay in, and the first projection yanked it back.  Since
        # lambda_max <~ ||phi||^2 E[n], starting at 0.52 with E[n] ~ 7.8 means lambda_max
        # ~ 2.1 at step zero: no unique mode, and a fixed-point iteration that diverges
        # rather than converges.  0.03 puts ||phi|| at 0.10, half the cap.
        self.phi = torch.nn.Parameter(torch.randn(J, Kz, generator=g) * phi_init)
        self.rho_c = torch.nn.Parameter(torch.zeros(C))
        # Version-4's foundational energy uses rho_c*k(k-1)/2 on the ENTIRE declared
        # support.  Saturating at the historical R=23 implementation limit changes the
        # joint law and its conditional logits.  Keep the original quadratic through
        # nmax; numerical safety is enforced by the rho_c floor.  At rho_c=-0.92 and
        # nmax=120 the largest exponent is 0.92*C(120,2)=656.9, within float64.
        self.rho_pair_cap = nmax
        self.rho_0_free = torch.nn.Parameter(torch.zeros(nmax))
        # Optional exact factorisation P(S|x)=P(S||S|,x)P_size(|S||x).  The composition
        # normaliser Z_n(x) is still supplied by this model; only the unstable coupling of
        # all n through one context-free rho_0 is replaced.  Buffers make the statistical
        # model self-describing in a checkpoint and add no fitted parameters.
        self.register_buffer("factored_size_enabled", torch.tensor(False))
        self.register_buffer("factored_size_log_p", torch.full((nmax,), -math.log(nmax)))
        # softplus(0.5413) = 1.0: starts as the un-split model, so a resumed run is unchanged
        self.price_kappa = torch.nn.Parameter(torch.tensor(0.5413))
        self.quad = None        # (nodes, weights) -> deterministic log Z
        # --- conditioning blocks (Eq. 7) -------------------------------------------------
        # softplus(-2.8) = 0.060, so gamma.beta starts at about 8 * 0.060^2 = 0.029 --
        # the same magnitude the unconstrained product had at initialisation
        # softplus(-3.3) = 0.036, so gamma.beta starts near 8 * 0.036^2 = 0.010 -- the
        # value the data's aggregate elasticity of -0.121 implies, given Var(n) ~ 83.
        # -2.8 started it at 0.029 and training pushed the median to 1.81.
        self.gamma = torch.nn.Parameter(torch.randn(N, Kp, generator=g) * 0.1 - 3.3)
        self.beta = torch.nn.Parameter(torch.randn(J, Kp, generator=g) * 0.1 - 3.3)
        self.w_dsp = torch.nn.Parameter(torch.zeros(J))
        self.w_mlr = torch.nn.Parameter(torch.zeros(J))
        self.mu = torch.nn.Parameter(torch.randn(J, Kt, generator=g) * 0.1)
        self.delta = torch.nn.Parameter(torch.randn(n_week, Kt, generator=g) * 0.1)
        self.psi = torch.nn.Parameter(torch.zeros(J, 4))       # recency loading
        self.zeta = torch.nn.Parameter(torch.randn(J, Ks, generator=g) * 0.1)
        self.xi = torch.nn.Parameter(torch.randn(S, Ks, generator=g) * 0.1)
        # --- units, Eq. 19: P(q_j | j in S) = NB(q_j - 1 ; Lambda_ij, r_j) ------------
        # A separate factor, and the factorisation is DERIVED rather than assumed: if the
        # energy does not depend on q, summing q out of P(S, q) leaves exactly P(S), so
        # P(S, q) = P(S) prod_j P(q_j).  Without it the model cannot say how many of
        # anything, which makes revenue -- price times units -- uncomputable, so a
        # simulator that omits it cannot price a promotion.
        self.a_q = torch.nn.Parameter(torch.zeros(J))
        self.gamma_q = torch.nn.Parameter(torch.randn(N, Kp, generator=g) * 0.1 - 2.8)
        self.beta_q = torch.nn.Parameter(torch.randn(J, Kp, generator=g) * 0.1 - 2.8)
        self.log_r = torch.nn.Parameter(torch.zeros(1))
        self.register_buffer("cat_of", torch.zeros(J, dtype=torch.long))
        self.house = None      # [B] set per batch, with .ctx below
        self.ctx = None        # dict of per-slot features, set per batch

    def rho_0(self):
        z = torch.zeros(1, dtype=self.rho_0_free.dtype, device=self.rho_0_free.device)
        return torch.cat([z, self.rho_0_free])

    def pair_feature(self, count):
        k = count.clamp(max=self.rho_pair_cap)
        return k * (k - 1) / 2.0

    def price_coefficients(self, it, trip):
        """Nonnegative product/household slopes shared by utility and price audits."""
        return (softplus(self.gamma[self.house[trip]])
                * softplus(self.beta[it])).sum(-1)

    @torch.no_grad()
    def set_global_price_sensitivity(self, sensitivity, relative_multiplier=1.0):
        """Set an exactly constant price slope without changing the energy formula.

        ``gamma`` and ``beta`` are a nonnegative factorization of the household-product
        coefficient.  Equal factor entries make every dot product exactly ``sensitivity``.
        The caller decides whether to freeze these parameters during fitting.
        """
        sensitivity = float(sensitivity)
        values = torch.full(
            (self.J,), sensitivity, dtype=self.beta.dtype, device=self.beta.device)
        self.set_product_price_sensitivity(values, relative_multiplier)

    @torch.no_grad()
    def set_product_price_sensitivity(self, sensitivity, relative_multiplier=1.0):
        """Set household-invariant product slopes in the existing factorization."""
        values = torch.as_tensor(
            sensitivity, dtype=self.beta.dtype, device=self.beta.device)
        relative_multiplier = float(relative_multiplier)
        if values.shape != (self.J,) or not bool(torch.isfinite(values).all()) \
                or not bool((values >= 0).all()):
            raise ValueError(
                "product price sensitivity must have one finite nonnegative value per product")
        if not math.isfinite(relative_multiplier) or relative_multiplier <= 0.0:
            raise ValueError("relative-price multiplier must be finite and positive")
        rank = int(self.gamma.shape[1])
        household_factor = 1.0 / math.sqrt(rank)
        product_factor = (values / math.sqrt(rank)).clamp_min(
            torch.finfo(values.dtype).tiny)
        household_raw = math.log(math.expm1(household_factor))
        product_raw = torch.log(torch.expm1(product_factor))
        multiplier_raw = math.log(math.expm1(relative_multiplier))
        self.gamma.fill_(household_raw)
        self.beta.copy_(product_raw[:, None].expand_as(self.beta))
        self.price_kappa.fill_(multiplier_raw)

    def b_at(self, it, trip, c):
        """Eq. 7 at an arbitrary set of (product, trip) pairs.

        ONE function for both the normaliser and the energy.  They previously had separate
        code paths and drifted: b_flat applied price, promotion, seasonality and store while
        energy() applied only taste, so E(S) and log Z scored the same product differently
        and the difference was free reward for the optimiser.  Nothing may compute an item
        value except through here.
        """
        hh = self.house[trip]
        theta = self.theta_c()
        if self.household_size_rank1:
            # Centring alpha_tilde makes every composition coordinate have zero catalogue
            # mean, so it cannot silently act as another household size intercept.
            alpha = self.alpha[:, :-1]
            alpha = alpha - alpha.mean(0, keepdim=True).detach()
            b = (self.lam[it] + theta[hh, -1]
                 + (theta[hh, :-1] * alpha[it]).sum(-1))
        else:
            b = self.lam[it] + (theta[hh] * self.alpha[it]).sum(-1)
        if c is None:
            return b
        # g_hj >= 0. With m equal to the offered mean price deviation, the exact
        # Jacobian is -g_hj [kappa*1(j=k) + (1-kappa)/J]. A common price move therefore
        # shifts utilities by -g_hj, not by a single catalogue-average coefficient.
        # Its size response is -Cov(N, sum_j g_hj Y_j). A one-price action also changes
        # other utilities through m, so coefficient positivity alone does not guarantee
        # marginal own-incidence monotonicity for kappa != 1.
        _gb = self.price_coefficients(it, trip)
        if "dlp_bar" in c:
            _m = c["dlp_bar"][trip]
            b = b - _gb * (_m + softplus(self.price_kappa) * (c["dlp"] - _m))
        else:
            b = b - _gb * c["dlp"]
        b = b + self.w_dsp[it] * c["disp"] + self.w_mlr[it] * c["mail"]
        b = b + (self.mu[it] * self.delta_c()[c["week"]]).sum(-1)
        b = b + (self.zeta[it] * self.xi_c()[c["store"]]).sum(-1)
        if "rec" in c:
            b = b + (self.psi[it] * c["rec"]).sum(-1)
        if "log_avail" in c:
            # Fixed data, not a parameter: log a_jst scales the product's weight in both the
            # normaliser and the energy, so an unconfirmed product keeps epsilon of its mass.
            b = b + c["log_avail"]
        return b

    # ---- GAUGE FIXING on the three bilinear terms -------------------------------------
    #
    # b contains lam_j + theta_h'alpha_j + mu_j'delta_w + zeta_j'xi_s.  For any fixed a,
    #
    #     lam_j -> lam_j + mu_j'a ,   delta_w -> delta_w - a
    #
    # leaves b EXACTLY invariant, for every product and every week.  So the likelihood is
    # flat along an 8-dimensional family and cannot decide how much per-product intercept
    # sits in lam versus in the seasonal term.  Only the penalties decide, and they are of
    # different degree.  Writing mu_j = (v_j/s).unit(delta_bar) to store an intercept v:
    #
    #     pool_ctx on mu   2.0 * ||v-vbar||^2 / (J*Kp*s^2)      falls with s
    #     wd on delta      (wd/2) * W * s^2                     rises with s
    #     minimised over s at 2*sqrt(AB) = 2.20e-4 * ||v-vbar|| -- LINEAR in ||v||
    #
    # while the same intercept in lam costs (wd/2)||v||^2 -- QUADRATIC.  Two factors of one
    # product turn squared penalties on the factors into a nuclear-norm penalty on the
    # product, which is degree 1.  Crossover at ||v|| = 44; the measured net intercept is
    # ||c|| = 244, so essentially all of it belongs in the bilinear term at the optimum, and
    # lam saturates at ||lam|| = 2.20e-4/wd = 22.0, i.e. std 0.298 REGARDLESS of ||c||.
    # run73 measured lam std 0.173 -> 0.311 flattening, against 0.298 predicted.  That is
    # why raising --pool-ctx never worked: it changes the coefficient, not the exponent, and
    # the optimiser evades it outright by shrinking mu and growing delta.
    #
    # Centring the CONTEXT side removes the flat direction and makes lam the unique owner of
    # the per-product constant.  It is gauge fixing, not regularisation: no signal is
    # deleted, and expressiveness strictly INCREASES, because the constant channel was
    # mu_j'delta_bar, confined to a <=Kp-dimensional subspace of R^J, and is now lam_j, free
    # in all J.  Applied to a running model it would need lam += mu'delta_bar at the same
    # instant to keep b invariant; from scratch both start at zero and no transfer is needed.
    #
    # theta'alpha needs this too.  It is the same shape -- a product of TWO trained tensors
    # -- so centring only delta and xi would let the intercept migrate into taste instead of
    # lam, and the fix would silently fail.  psi'rec does NOT need it: rec is fixed data, so
    # psi enters linearly, is penalised quadratically, and has no escape route.
    def theta_c(self):
        # Keep the forward value in the same zero-mean gauge, but do not route the
        # derivative of that gauge operation through Adam.  With the ordinary expression
        # every minibatch gives *all* households a tiny dense gradient through ``mean``.
        # Adam is coordinatewise scale invariant, so those 1/N gradients become ordinary
        # sized updates to households that were not in the minibatch.  The mathematically
        # equivalent constrained update is: differentiate in the ambient coordinates,
        # take the optimizer step, then project back to zero mean (project_context_gauges).
        return self.theta - self.theta.mean(0, keepdim=True).detach()

    def delta_c(self):
        return self.delta - self.delta.mean(0, keepdim=True).detach()

    def xi_c(self):
        return self.xi - self.xi.mean(0, keepdim=True).detach()

    @torch.no_grad()
    def project_context_gauges(self):
        """Project context factors to the exact gauge used by :meth:`b_at`.

        This changes no item utility: ``theta_c``, ``delta_c`` and ``xi_c`` subtract the
        same means in the forward pass.  Its purpose is optimizer geometry.  Projection
        after Adam keeps inactive rows inactive during differentiation while preventing
        the raw, unidentified common components from drifting.
        """
        self.theta.sub_(self.theta.mean(0, keepdim=True))
        self.delta.sub_(self.delta.mean(0, keepdim=True))
        self.xi.sub_(self.xi.mean(0, keepdim=True))
        if self.household_size_rank1:
            # These changes are already made in the forward gauge above and therefore do
            # not change any utility.  Keeping the stored tensors in that gauge prevents
            # optimizer drift and makes theta[:, -1] directly auditable as kappa_h.
            self.alpha[:, :-1].sub_(self.alpha[:, :-1].mean(0, keepdim=True))
            self.alpha[:, -1].fill_(1.0)

    def b_flat(self, ix):
        """b at every assortment slot, [T] -- the normaliser's view."""
        if getattr(self, "_b_override", None) is not None:
            return self._b_override
        return self.b_at(ix.item, ix.item_trip, self.ctx)

    def _log_Z_quad(self, ix, drop_empty, return_ess, return_size, return_mode):
        """log Z by deterministic Smolyak quadrature, with finite-rule integration error.

        f(z) is already exact (the ESP/poly-tree recursion is a closed form), so the only
        approximation in the whole pipeline was the outer E_z over a Kz-dimensional
        Gaussian.  Importance sampling cannot do that integral here: verified against exact
        enumeration, 4096 draws are wrong by 8-36 nats, because f(z) ~ exp(c||z||) keeps its
        mass in a shell the mode-centred proposal never reaches.  A sparse grid does it
        deterministically -- at the strength grocery needs (phi'phi = 0.92) the error is
        0.005 nats at Kz=2 with 17 points, fewer points than the 16 draws it replaces.

        Smolyak weights are signed, so the sum is formed in linear space after factoring out
        the row max; logsumexp cannot be used.
        """
        nodes, wts = self.quad                     # [P, Kz], [P]
        B, P = ix.B, nodes.shape[0]
        zs = nodes.to(self.lam.dtype).unsqueeze(0).expand(B, P, self.Kz)
        w = wts.to(self.lam.dtype)
        # Sparse path: only phi-carrying products depend on z, so the ESP over the rest
        # and the category convolution over untouched categories are lifted out of the node
        # loop.  Verified bit-equal to log_f_ragged (max rel 2.3e-16) and gradient-equal on
        # masked-in phi (6.9e-14); the only difference is that phi_j = 0 products get no
        # gradient, and fit.py re-applies the mask every step so those are discarded anyway.
        # 83.7x at a 24-product mask, 7.0x at 400.
        _C = sparse_prepare(self, ix)
        if return_size:
            lg = log_f_sparse(self, zs, ix, _C, drop_empty, return_terms=True)   # [P, B, n]
            lf = torch.logsumexp(lg, dim=-1).transpose(0, 1)                     # [B, P]
        else:
            lf = log_f_sparse(self, zs, ix, _C, drop_empty)                      # [B, P]
        lz, sign, log_condition = signed_log_integral(lf, w, dim=1)
        self._last_quad_log_condition = log_condition.detach()
        if bool((sign != 1).any()) or not bool(torch.isfinite(lz).all()):
            raise FloatingPointError(
                "signed sparse quadrature produced a non-positive partition estimate; "
                f"minimum sign={float(sign.min().detach()):.0f}, "
                f"maximum log cancellation={float(log_condition.max().detach()):.6g}")
        out = [lz]
        if return_ess:
            # Deterministic: every node contributes by construction.  Reported as 1 so the
            # ESS gate -- which exists to catch a collapsed sampler -- never fires on a rule
            # that has no sampler to collapse.
            out.append(torch.ones(B, dtype=lz.dtype, device=lz.device))
        if return_size:
            size_log_mass, size_sign, size_condition = signed_log_integral(
                lg.permute(1, 2, 0), w, dim=2)
            invalid_size = (size_sign < 0) | ((size_sign == 0)
                                              & torch.isfinite(size_log_mass))
            if bool(invalid_size.any()):
                raise FloatingPointError(
                    "signed sparse quadrature produced a negative size mass; "
                    f"maximum live log cancellation="
                    f"{float(size_condition[torch.isfinite(size_log_mass)].max().detach()):.6g}")
            pn = torch.softmax(size_log_mass, dim=-1)
            out.append(pn)
        if return_mode:
            out.append(torch.zeros(B, self.Kz, dtype=lz.dtype, device=lz.device))
        return out[0] if len(out) == 1 else tuple(out)

    def log_Z(self, ix, n_draws=32, mode_steps=1, generator=None, return_ess=False,
              drop_empty=False, return_size=False, return_mode=False):
        """log Z (or log Z_+ with ``drop_empty``) at every trip in ``ix``.

        With ``self.quad`` installed this is the deterministic signed Smolyak rule of
        :meth:`_log_Z_quad`, which every certified stage uses.  Otherwise it is importance
        sampling from an identity-covariance Gaussian centred at a ``mode_steps``
        fixed-point estimate of the z-mode.  In the regime Section 14 requires,
        lambda_max(Lambda) is about 0.1-0.3, so the exact posterior covariance
        (I - Lambda)^{-1} lies between I and about 1.4 I; one mode step triples accuracy
        over none, while further steps change little.
        """
        if getattr(self, "quad", None) is not None:
            return self._log_Z_quad(ix, drop_empty, return_ess, return_size, return_mode)
        B = ix.B
        with torch.no_grad():
            z = torch.zeros(B, 1, self.Kz, dtype=self.lam.dtype, device=self.lam.device)
            for _ in range(mode_steps):
                zz = z.detach().requires_grad_(True)
                with torch.enable_grad():
                    lf = log_f_ragged(self, zz, ix, drop_empty).sum()
                z = torch.autograd.grad(lf, zz)[0]
            zh = z.detach()
            noise = torch.randn(B, n_draws, self.Kz, dtype=zh.dtype, device=zh.device,
                                generator=generator)
            zs = zh + noise
            L2P = float(math.log(2 * math.pi))
            log_q = -0.5 * (noise ** 2).sum(-1) - 0.5 * self.Kz * L2P
        if return_size:
            # The per-size terms are already formed inside log f; taking them here shares
            # the draws with the normaliser, so the size law costs nothing extra.
            lg = log_f_ragged(self, zs, ix, drop_empty, return_terms=True)   # [D, B, n]
            lf = torch.logsumexp(lg, dim=-1).transpose(0, 1)                 # [B, D]
        else:
            lf = log_f_ragged(self, zs, ix, drop_empty)
        base = -0.5 * self.Kz * L2P - 0.5 * (zs ** 2).sum(-1)
        lw = base + lf - log_q
        lz = torch.logsumexp(lw, dim=1) - math.log(n_draws)
        out = [lz]
        if return_ess:
            with torch.no_grad():
                ww = torch.softmax(lw, dim=1)
                out.append(1.0 / (ww ** 2).sum(1) / n_draws)   # PER TRIP, never a batch mean
        if return_size:
            tot = (base - log_q).unsqueeze(-1) + lg.permute(1, 0, 2)         # [B, D, n]
            out.append(torch.softmax(tot.reshape(B, -1), dim=1).view(tot.shape).sum(1))
        if return_mode:
            out.append(zh[:, 0, :].detach())
        return out[0] if len(out) == 1 else tuple(out)

    def energy(self, line_item, line_trip, line_cat, B, line_ctx=None):
        """E(S) from the observed lines, using the SAME item values as the normaliser."""
        dt, dev = self.lam.dtype, self.lam.device
        lin = torch.zeros(B, dtype=dt, device=dev).index_add_(
            0, line_trip, self.b_at(line_item, line_trip, line_ctx))
        v = torch.zeros(B, self.Kz, dtype=dt, device=dev).index_add_(
            0, line_trip, self.phi[line_item])
        sq = torch.zeros(B, dtype=dt, device=dev).index_add_(
            0, line_trip, (self.phi[line_item] ** 2).sum(-1))
        pair = 0.5 * ((v * v).sum(-1) - sq)
        key = line_trip * self.C + line_cat
        nc = torch.bincount(key, minlength=B * self.C).view(B, self.C).to(dt)
        pen_c = (self.rho_c.unsqueeze(0) * self.pair_feature(nc)).sum(-1)
        n = torch.bincount(line_trip, minlength=B)
        if B and int(n.max()) > self.nmax:
            # The law is declared on 1 <= |S| <= nmax.  Clamping would silently score an
            # unsupported basket with the wrong size potential.
            raise ValueError(
                f"observed basket of size {int(n.max())} lies outside support 1..{self.nmax}")
        return lin + pair - pen_c - self.rho_0()[n]

    def units_loglik(self, line_item, line_trip, units, line_ctx, B):
        """log P(q | S), summed per trip.  Shifted negative binomial: q - 1 ~ NB(mu, r).

        Version 2 measures the shifted POISSON as the wrong law here -- q is under-dispersed
        (0.62) while q - 1 is over-dispersed (2.35), which a one-parameter Poisson cannot do
        -- and the negative binomial cuts units-per-line total variation from 0.045 to
        0.016 for one extra parameter."""
        hh = self.house[line_trip]
        # same constraint on the units model: quantity must not rise with price
        z = self.a_q[line_item] - (softplus(self.gamma_q[hh])
                                   * softplus(self.beta_q[line_item])).sum(-1) \
            * line_ctx["dlp"]
        mu = torch.exp(z.clamp(-6.0, 4.0))
        r = torch.nn.functional.softplus(self.log_r) + 1e-6
        k = (units - 1).to(mu.dtype).clamp_min(0.0)
        ll = (torch.lgamma(k + r) - torch.lgamma(r) - torch.lgamma(k + 1.0)
              + r * (torch.log(r) - torch.log(r + mu))
              + k * (torch.log(mu.clamp_min(1e-12)) - torch.log(r + mu)))
        return torch.zeros(B, dtype=mu.dtype, device=mu.device).index_add_(0, line_trip, ll)

    def loglik(self, ix, line_item, line_trip, line_cat, n_draws=32,
               generator=None, return_ess=False, line_ctx=None, units=None,
               return_size=False):
        """log P(S | S non-empty) = E(S) - log Z_+, plus log P(q | S) when ``units``.

        log Z_+ omits the degree-zero term directly rather than subtracting one from Z.
        """
        factored = bool(self.factored_size_enabled)
        need_size = return_size or factored
        out = self.log_Z(ix, n_draws=n_draws, generator=generator,
                         return_ess=return_ess, drop_empty=True, return_size=need_size)
        out = list(out) if isinstance(out, tuple) else [out]
        lz1 = out.pop(0)
        ess = out.pop(0) if return_ess else None
        pn_internal = out.pop(0) if need_size else None
        ll = self.energy(line_item, line_trip, line_cat, ix.B, line_ctx) - lz1
        if factored:
            n = torch.bincount(line_trip, minlength=ix.B)
            if bool((n <= 0).any()) or int(n.max()) > pn_internal.shape[1]:
                raise ValueError("observed basket lies outside factored size support")
            row = torch.arange(ix.B, device=line_trip.device)
            ll = (ll - pn_internal[row, n - 1].clamp_min(1e-300).log()
                  + self.factored_size_log_p[n - 1])
        if units is not None:
            ll = ll + self.units_loglik(line_item, line_trip, units, line_ctx, ix.B)
        res = [ll]
        if return_ess:
            res.append(ess)
        if return_size:
            res.append(self.factored_size_log_p.exp().unsqueeze(0).expand(ix.B, -1)
                       if factored else pn_internal)
        return res[0] if len(res) == 1 else tuple(res)

    @torch.no_grad()
    def project_rho_c(self, floor=-1.5):
        """Floor the within-category ATTRACTION.

        rho_c enters as exp(-rho_c n_c(n_c-1)/2), so repulsion (rho_c > 0) shrinks the term
        and attraction grows it QUADRATICALLY in the category count.  With R = 23 and
        rho_c = -2.809 that is exp(711); float64 overflows at exp(709), and run35c went NaN.
        The spec puts no floor on rho_c, which was harmless while the partition was
        dunnhumby's commodity groups -- they rarely group true complements, so rho_c stayed
        near zero.  Affinity groups drive it hard negative and the term detonates.

        The data bounds the sensible value: a 2.5x pair lift is rho_c = -0.92, so -1.5
        (4.5x) leaves room while keeping exp(1.5 * 253) at 1e165.
        """
        self.rho_c.clamp_(min=floor)
