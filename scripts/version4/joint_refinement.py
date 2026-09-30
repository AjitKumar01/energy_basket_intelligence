"""Joint refinement of a staged Version-4 fit by damped, iterated Monte Carlo maximum likelihood.

The staged pipeline is a two-step estimator: the additive block (utilities, household taste,
rho_c, rho_0) is fitted with Phi = 0 and then frozen while the interaction is solved.  The
additive block absorbs part of the interaction and cannot give it back.  This module moves the
blocks together, starting from the staged optimum, without differentiating the quadrature:

  1. Bank.  For every context, draw M baskets from the CURRENT law with Model A's blocked Gibbs
     sampler on (S, z) at beta = 1: z | S ~ N(sum_{j in S} phi_j, I) and S | z exactly by the
     category/size dynamic program (tempered_block_gibbs.conditional_slots).
  2. Fixed-bank objective (Geyer and Thompson 1992).  With dE = E_new - E_parent,
         sum_t dE(S_t) - sum_t log mean_m exp dE(S_t^m).
     Every refined block enters the energy linearly -- the interaction as tr(C F_U(S)) in the fixed
     basis U, item intercepts, rho_c, the size potential, household taste theta (product taste
     alpha fixed) and alpha (theta fixed) -- so each block's objective is concave.
  3. Trust region.  Each round solves the objective plus lam * |change|^2 / N and takes the
     smallest lam whose solution keeps the bank's effective sample size above target; the step
     therefore never leaves the region where the bank represents the law.  The solution becomes
     the next parent, the bank is redrawn, and the rounds stop when the bank gain vanishes.

The certified price component, seasonality, store and promotion blocks, and Phi's basis U stay
fixed.  Acceptance is decided outside this module, on held-out likelihood.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import torch

from tempered_block_gibbs import (conditional_log_tables_levels, conditional_slots, conditional_slots_levels,
                                  conditional_slots_repeated)


# ---------------------------------------------------------------------------------------------
# Baskets as slot lists
# ---------------------------------------------------------------------------------------------
@dataclass
class SlotBaskets:
    """Baskets as positions in a RaggedIndex's slot list.

    slots [L]     assortment-slot position of every basket member
    basket [L]    basket id of every member
    context [N]   index-trip (context) of every basket
    """
    slots: torch.Tensor
    basket: torch.Tensor
    context: torch.Tensor

    @property
    def n(self) -> int:
        return int(self.context.numel())


def slot_table(ix, n_item: int) -> torch.Tensor:
    """[B, J] slot position of each product in each trip's assortment, -1 when not offered."""
    table = torch.full((ix.B, n_item), -1, dtype=torch.long)
    table[ix.item_trip, ix.item] = torch.arange(ix.item.numel())
    return table


def baskets_from_items(table: torch.Tensor, contexts, item_lists) -> SlotBaskets:
    slots, basket = [], []
    for b, (context, items) in enumerate(zip(contexts, item_lists)):
        s = table[int(context), torch.as_tensor(items, dtype=torch.long)]
        if bool((s < 0).any()):
            raise ValueError("a basket contains a product outside its context's assortment")
        slots.append(s)
        basket.append(torch.full((len(s),), b, dtype=torch.long))
    return SlotBaskets(torch.cat(slots), torch.cat(basket), torch.as_tensor(np.asarray(contexts), dtype=torch.long))


# ---------------------------------------------------------------------------------------------
# Energy
# ---------------------------------------------------------------------------------------------
def taste_utility(model, ix) -> torch.Tensor:
    """The refined part of b_at at every slot: lam_j + theta_c[h]'alpha_j (Model A's formula).

    b_flat = (frozen part: price, promotion, season, store, availability) + this term, so the
    frozen part can be computed once per round and only this term re-evaluated.
    """
    hh = model.house[ix.item_trip]
    it = ix.item
    theta = model.theta_c()
    if model.household_size_rank1:
        alpha = model.alpha[:, :-1]
        alpha = alpha - alpha.mean(0, keepdim=True).detach()
        return model.lam[it] + theta[hh, -1] + (theta[hh, :-1] * alpha[it]).sum(-1)
    return model.lam[it] + (theta[hh] * model.alpha[it]).sum(-1)


def basket_energy(model, ix, baskets: SlotBaskets, basis=None, C=None, slot_b=None) -> torch.Tensor:
    """E(S) for every basket with Model A's utilities and penalties.

    With (basis, C) the Gram energy is sum_{i<j} u_i'C u_j (linear in C); otherwise it is phi's.
    slot_b, when given, replaces model.b_flat(ix).
    """
    N = baskets.n
    b = model.b_flat(ix) if slot_b is None else slot_b
    linear = torch.zeros(N, dtype=b.dtype).index_add(0, baskets.basket, b[baskets.slots])
    item = ix.item[baskets.slots]
    if basis is None:
        rows = model.phi[item]
        v = torch.zeros(N, rows.shape[1], dtype=b.dtype).index_add(0, baskets.basket, rows)
        sq = torch.zeros(N, dtype=b.dtype).index_add(0, baskets.basket, rows.square().sum(-1))
        pair = 0.5 * (v.square().sum(-1) - sq)
    else:
        rows = basis[item]
        v = torch.zeros(N, rows.shape[1], dtype=b.dtype).index_add(0, baskets.basket, rows)
        self_term = torch.zeros(N, dtype=b.dtype).index_add(
            0, baskets.basket, ((rows @ C) * rows).sum(-1))
        pair = 0.5 * (((v @ C) * v).sum(-1) - self_term)
    category = ix.row_cat[ix.row_of[baskets.slots]]
    key = baskets.basket * model.C + category
    counts = torch.bincount(key, minlength=N * model.C).view(N, model.C).to(b.dtype)
    penalty = (model.rho_c.unsqueeze(0) * model.pair_feature(counts)).sum(-1)
    size = torch.bincount(baskets.basket, minlength=N)
    return linear + pair - penalty - model.rho_0()[size]


@dataclass
class BankDesign:
    """Per-basket sufficient statistics of a fixed set of baskets for the refined terms of E(S).

    With the frozen utility part and the Phi basis U held fixed, every refined term is linear in
    these statistics (the taste term bilinear in alpha and theta), so an objective evaluation is a
    few sparse products over N baskets instead of a gather over every basket member:

      E(S) = frozen(S) + A_S lam + sum_k theta_h,k (A_S alpha)_k + 0.5 <G_S, C> - P_S rho_c - rho_0[|S|]

    with A the basket-by-product incidence, G_S = s s' - sum_j u_j u_j' (s = sum_j u_j) stored as
    its upper triangle (off-diagonals doubled), and P_S the capped category pair counts.
    """
    frozen: torch.Tensor      # [N]
    incidence: torch.Tensor   # sparse [N, J]
    household: torch.Tensor   # [N]
    gram: torch.Tensor        # [N, r(r+1)/2]
    pairs: torch.Tensor       # sparse [N, n_cat]
    size: torch.Tensor        # [N]


def _upper(rank: int):
    iu = torch.triu_indices(rank, rank)
    weight = torch.where(iu[0] == iu[1], 1.0, 2.0).to(torch.get_default_dtype())
    return iu, weight


@torch.no_grad()
def bank_design(model, ix, baskets: SlotBaskets, basis: torch.Tensor, frozen_slot: torch.Tensor,
                chunk: int = 1 << 20) -> BankDesign:
    N, J = baskets.n, model.lam.numel()
    dtype = frozen_slot.dtype
    item = ix.item[baskets.slots]
    frozen = torch.zeros(N, dtype=dtype).index_add(0, baskets.basket, frozen_slot[baskets.slots])
    incidence = torch.sparse_coo_tensor(torch.stack([baskets.basket, item]),
                                        torch.ones(item.numel(), dtype=dtype), (N, J)).coalesce()
    iu, _ = _upper(basis.shape[1])
    s = torch.zeros(N, basis.shape[1], dtype=dtype)
    self_term = torch.zeros(N, iu.shape[1], dtype=dtype)
    for a in range(0, item.numel(), chunk):
        rows = basis[item[a:a + chunk]]
        s.index_add_(0, baskets.basket[a:a + chunk], rows)
        self_term.index_add_(0, baskets.basket[a:a + chunk], rows[:, iu[0]] * rows[:, iu[1]])
    gram = s[:, iu[0]] * s[:, iu[1]] - self_term
    category = ix.row_cat[ix.row_of[baskets.slots]]
    key = baskets.basket * model.C + category
    unique, counts = torch.unique(key, return_counts=True)
    feature = model.pair_feature(counts.to(dtype))
    keep = feature != 0
    pairs = torch.sparse_coo_tensor(torch.stack([unique[keep] // model.C, unique[keep] % model.C]),
                                    feature[keep], (N, model.C)).coalesce()
    size = torch.bincount(baskets.basket, minlength=N)
    return BankDesign(frozen, incidence, model.house[baskets.context], gram, pairs, size)


def design_energy(model, design: BankDesign, C: torch.Tensor) -> torch.Tensor:
    """basket_energy(model, ix, baskets, basis, C, frozen + taste_utility) from the design."""
    A = design.incidence
    theta = model.theta_c()[design.household]
    if model.household_size_rank1:
        alpha = model.alpha[:, :-1]
        alpha = alpha - alpha.mean(0, keepdim=True).detach()
        taste = theta[:, -1] * design.size.to(theta.dtype) + (torch.sparse.mm(A, alpha) * theta[:, :-1]).sum(-1)
    else:
        taste = (torch.sparse.mm(A, model.alpha) * theta).sum(-1)
    linear = torch.sparse.mm(A, model.lam.unsqueeze(1)).squeeze(1)
    iu, weight = _upper(C.shape[0])
    pair = 0.5 * (design.gram @ (C[iu[0], iu[1]] * weight))
    penalty = torch.sparse.mm(design.pairs, model.rho_c.unsqueeze(1)).squeeze(1)
    return design.frozen + linear + taste + pair - penalty - model.rho_0()[design.size]


# ---------------------------------------------------------------------------------------------
# Bank: blocked Gibbs on (S, z) at beta = 1
# ---------------------------------------------------------------------------------------------
@torch.no_grad()
def draw_bank(model, ix_rep, base_of_rep, table, draws_per_chain: int, burn: int,
              generator: torch.Generator) -> SlotBaskets:
    """Draw baskets from the current law.

    ix_rep indexes the contexts repeated once per chain; base_of_rep maps each of its trips to the
    base context.  Each chain runs `burn` sweeps, then records `draws_per_chain` sweeps.
    """
    Kz = model.Kz
    z = torch.zeros(ix_rep.B, Kz, dtype=model.phi.dtype)
    contexts, items = [], []
    for sweep in range(burn + draws_per_chain):
        state = conditional_slots(model, ix_rep, z, 1.0, generator)
        v = torch.zeros(ix_rep.B, Kz, dtype=model.phi.dtype)
        for trip, slots in enumerate(state):
            chosen = ix_rep.item[slots]
            v[trip] = model.phi[chosen].sum(0)
            if sweep >= burn:
                contexts.append(int(base_of_rep[trip]))
                items.append(chosen.numpy())
        z = v + torch.randn(v.shape, generator=generator, dtype=v.dtype)
    order = np.argsort(np.asarray(contexts), kind="stable")
    contexts = np.asarray(contexts)[order]
    items = [items[i] for i in order]
    return baskets_from_items(table, contexts, items)


def _flatten(state):
    """List of per-trip slot tensors -> (slots, trip) flat arrays."""
    lengths = torch.as_tensor([len(s) for s in state], dtype=torch.long)
    slots = torch.cat([torch.as_tensor(s, dtype=torch.long) for s in state]) if len(state) else torch.zeros(0, dtype=torch.long)
    return slots, torch.repeat_interleave(torch.arange(len(state)), lengths)


@torch.no_grad()
def draw_bank_fast(model, ix_rep, base_of_rep, table, draws_per_chain: int, burn: int,
                   generator: torch.Generator, init_items=None, baskets_per_z: int = 1,
                   compiled: bool = True, parallel: bool = True) -> SlotBaskets:
    """Same law as draw_bank, with Model A's vectorized exact S | z sampler.

    init_items: optional per-rep-trip item lists; the chain starts at z ~ N(sum phi over them, I)
    (a warm start at an observed basket).  After burn-in, each z draw yields `baskets_per_z`
    independent exact draws from S | z (each is marginally a draw from the basket law once z is
    stationary; baskets sharing a z are correlated).
    """
    B, Kz = ix_rep.B, model.Kz
    if init_items is not None:
        v = torch.zeros(B, Kz, dtype=model.phi.dtype)
        for trip, items in enumerate(init_items):
            v[trip] = model.phi[torch.as_tensor(items, dtype=torch.long)].sum(0)
        z = v + torch.randn(v.shape, generator=generator, dtype=v.dtype)
    else:
        z = torch.zeros(B, Kz, dtype=model.phi.dtype)
    contexts, flat_slots, flat_basket = [], [], []
    count = 0
    base_tensor = torch.as_tensor(np.asarray(base_of_rep), dtype=torch.long)
    for sweep in range(burn + draws_per_chain):
        recording = sweep >= burn
        if compiled:
            from compiled_backtrack import compiled_draws
            k = baskets_per_z if recording else 1
            log_g, centred, log_size = conditional_log_tables_levels(model, ix_rep, z.unsqueeze(0), [1.0])
            slots, basket = compiled_draws(log_g, centred, log_size, ix_rep, k, generator, parallel)
            trip = basket % B
            if recording:
                flat_slots.append(table[base_tensor[trip], ix_rep.item[slots]])
                flat_basket.append(basket + count)
                contexts.append(np.tile(np.asarray(base_of_rep), k))
                count += k * B
            first_draw = basket < B                       # the first draw of each trip sets z
            v = torch.zeros(B, Kz, dtype=model.phi.dtype).index_add(
                0, trip[first_draw], model.phi[ix_rep.item[slots[first_draw]]])
            z = v + torch.randn(v.shape, generator=generator, dtype=v.dtype)
            continue
        if recording and baskets_per_z > 1:
            draws = conditional_slots_repeated(model, ix_rep, z, 1.0, baskets_per_z, generator)
            states = [[d[t] for t in range(B)] for d in draws]
        else:
            states = [conditional_slots_levels(model, ix_rep, z.unsqueeze(0), [1.0], generator)[0]]
        last = None
        for state in states:
            slots, trip = _flatten(state)
            last = (slots, trip)
            if recording:
                items = ix_rep.item[slots]
                base_trip = torch.as_tensor(np.asarray(base_of_rep), dtype=torch.long)[trip]
                flat_slots.append(table[base_trip, items])
                flat_basket.append(trip + count)
                contexts.append(np.asarray(base_of_rep))
                count += B
        slots, trip = last
        v = torch.zeros(B, Kz, dtype=model.phi.dtype).index_add(0, trip, model.phi[ix_rep.item[slots]])
        z = v + torch.randn(v.shape, generator=generator, dtype=v.dtype)
    slots = torch.cat(flat_slots); basket = torch.cat(flat_basket)
    context = torch.as_tensor(np.concatenate(contexts), dtype=torch.long)
    order = torch.argsort(context, stable=True)              # context-major basket order
    rank = torch.empty_like(order); rank[order] = torch.arange(order.numel())
    return SlotBaskets(slots, rank[basket], context[order])


# ---------------------------------------------------------------------------------------------
# Damped rounds
# ---------------------------------------------------------------------------------------------
def orthonormal_basis(phi: torch.Tensor, rank: int):
    U, S, _ = torch.linalg.svd(phi[:, :rank], full_matrices=False)
    C = torch.diag(S.square())
    return U, C


def set_phi_from(model, U, C, rank: int, cap: float):
    """Write phi = U C^{1/2} after projecting C to 0 <= C <= cap*I (Model A's interaction contract)."""
    with torch.no_grad():
        C = (C + C.T) / 2
        values, vectors = torch.linalg.eigh(C)
        values = values.clamp(0.0, cap)
        root = vectors * values.sqrt()
        model.phi.zero_()
        model.phi[:, :rank] = U @ root
    return (vectors * values) @ vectors.T


def pooling_penalty(model, weight: float) -> torch.Tensor:
    """The additive stage's pooling term on the bilinear blocks (fit_exact_additive --pool-prod)."""
    total = torch.zeros((), dtype=model.lam.dtype)
    for product, context in ((model.mu, model.delta_c()), (model.zeta, model.xi_c()),
                             (model.alpha, model.theta_c())):
        total = total + torch.trace((product.T @ product) @ (context.T @ context)) / (
            product.shape[0] * context.shape[0])
    return weight * total


def lbfgs(params, value, max_iter: int = 100, rounds: int = 3):
    opt = torch.optim.LBFGS(params, lr=1, max_iter=max_iter, history_size=20,
                            tolerance_grad=1e-9, tolerance_change=1e-12, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        v = value()
        v.backward()
        return v

    for _ in range(rounds):
        opt.step(closure)


def bank_statistics(dE_bank: torch.Tensor):
    """Per-context ESS fraction of the bank weights exp(dE)."""
    w = torch.softmax(dE_bank, 1)
    return (1.0 / w.square().sum(1)) / dE_bank.shape[1]


def concat_designs(parts) -> BankDesign:
    """Stack per-chunk designs (baskets in chunk order) into one design."""
    rows = 0
    incidence, pairs = [], []
    for d in parts:
        i = d.incidence.indices().clone(); i[0] += rows
        incidence.append((i, d.incidence.values()))
        i = d.pairs.indices().clone(); i[0] += rows
        pairs.append((i, d.pairs.values()))
        rows += d.frozen.numel()
    J, n_cat = parts[0].incidence.shape[1], parts[0].pairs.shape[1]

    def sparse(chunks, width):
        return torch.sparse_coo_tensor(torch.cat([c[0] for c in chunks], 1), torch.cat([c[1] for c in chunks]),
                                       (rows, width)).coalesce()

    return BankDesign(torch.cat([d.frozen for d in parts]), sparse(incidence, J),
                      torch.cat([d.household for d in parts]), torch.cat([d.gram for d in parts]),
                      sparse(pairs, n_cat), torch.cat([d.size for d in parts]))


def round_designs(model, ix, observed: SlotBaskets, bank: SlotBaskets, U):
    """Designs of the observed and bank baskets over one assortment index (all contexts at once)."""
    with torch.no_grad():
        frozen = model.b_flat(ix) - taste_utility(model, ix)
        return bank_design(model, ix, observed, U, frozen), bank_design(model, ix, bank, U, frozen)


def damped_round(model, obs_design: BankDesign, bank_design_: BankDesign, draws: int, U, C,
                 rank: int, trust_ladder, ess_rule, cycles: int, pool_prod: float,
                 cap: float, log=print, frozen_names=(), trust_scale=None):
    """One round: parent = current model; returns (accepted, record).

    The observed and bank baskets enter only through their designs (per-basket sufficient
    statistics), so the caller may build them chunk by chunk and memory does not grow with
    contexts x assortment.

    frozen_names: refined parameters to hold at their current values this round (e.g. "theta").
    trust_scale: per-parameter multipliers on the trust penalty (e.g. {"theta": 10} shrinks the
    household taste update ten times harder than the population parameters)."""
    trust_scale = trust_scale or {}
    names = tuple(n for n in ("lam", "theta", "rho_c", "rho_0_free") if n not in frozen_names)
    N = obs_design.frozen.numel()
    with torch.no_grad():
        E_obs0 = design_energy(model, obs_design, C)
        E_bank0 = design_energy(model, bank_design_, C)
    start = {name: getattr(model, name).detach().clone() for name in names + ("alpha",)}
    C_start = C.detach().clone()
    timings = []

    for lam_trust in trust_ladder:
        t_level = time.time()
        with torch.no_grad():
            for name, value in start.items():
                getattr(model, name).copy_(value)
        Cv = C_start.clone().requires_grad_(True)

        def objective():
            Cs = (Cv + Cv.T) / 2
            dE_obs = design_energy(model, obs_design, Cs) - E_obs0
            dE_bank = (design_energy(model, bank_design_, Cs) - E_bank0).view(N, draws)
            fit = (dE_obs.sum() - (torch.logsumexp(dE_bank, 1) - math.log(draws)).sum()) / N
            change = sum(trust_scale.get(k, 1.0) * (getattr(model, k) - start[k]).square().sum() for k in start) \
                + (Cv - C_start).square().sum()
            return -fit + pooling_penalty(model, pool_prod) + lam_trust * change / N

        block_a = [Cv] + [getattr(model, k) for k in names]
        blocks = [block_a] + ([] if "alpha" in frozen_names else [[model.alpha]])
        for _ in range(cycles):
            for block in blocks:
                others = [p for p in model.parameters() if all(p is not q for q in block)]
                saved = [p.requires_grad for p in others]
                for p in others:
                    p.requires_grad_(False)
                for p in block:
                    p.requires_grad_(True)
                lbfgs(block, objective)
                for p, flag in zip(others, saved):
                    p.requires_grad_(flag)
        with torch.no_grad():
            Cs = (Cv + Cv.T) / 2
            dE_bank = (design_energy(model, bank_design_, Cs) - E_bank0).view(N, draws)
            dE_obs = design_energy(model, obs_design, Cs) - E_obs0
            ess = bank_statistics(dE_bank)
            gain = float((dE_obs.sum() - (torch.logsumexp(dE_bank, 1) - math.log(draws)).sum()) / N)
            finite = bool(torch.isfinite(dE_bank).all() and torch.isfinite(dE_obs).all())
        ok, summary = ess_rule(ess)
        timings.append({"trust": lam_trust, "seconds": round(time.time() - t_level, 2)})
        log(f"[joint-refinement]   trust {lam_trust:g}: bank gain {gain:+.5f}, ESS {summary}, "
            f"{timings[-1]['seconds']:.1f}s" + ("" if finite else ", non-finite"))
        if finite and ok:
            return True, {"trust": lam_trust, "bank_gain": gain, "ess": summary, "C": Cs.detach(),
                          "trust_seconds": timings}
    with torch.no_grad():
        for name, value in start.items():
            getattr(model, name).copy_(value)
    return False, {"status": "no trust level kept the bank valid", "trust_seconds": timings}
