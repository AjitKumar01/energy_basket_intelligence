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
from dataclasses import dataclass

import numpy as np
import torch

from tempered_block_gibbs import conditional_slots, conditional_slots_levels, conditional_slots_repeated


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
                   generator: torch.Generator, init_items=None, baskets_per_z: int = 1) -> SlotBaskets:
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
    for sweep in range(burn + draws_per_chain):
        recording = sweep >= burn
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


def damped_round(model, ix, observed: SlotBaskets, bank: SlotBaskets, draws: int, U, C,
                 rank: int, trust_ladder, ess_rule, cycles: int, pool_prod: float,
                 cap: float, log=print):
    """One round: parent = current model; returns (accepted, record)."""
    names = ("lam", "theta", "rho_c", "rho_0_free")
    with torch.no_grad():
        frozen = model.b_flat(ix) - taste_utility(model, ix)
        E_obs0 = basket_energy(model, ix, observed, U, C)
        E_bank0 = basket_energy(model, ix, bank, U, C)
    N = observed.n
    start = {name: getattr(model, name).detach().clone() for name in names + ("alpha",)}
    C_start = C.detach().clone()

    for lam_trust in trust_ladder:
        with torch.no_grad():
            for name, value in start.items():
                getattr(model, name).copy_(value)
        Cv = C_start.clone().requires_grad_(True)

        def objective():
            slot_b = frozen + taste_utility(model, ix)
            Cs = (Cv + Cv.T) / 2
            dE_obs = basket_energy(model, ix, observed, U, Cs, slot_b) - E_obs0
            dE_bank = (basket_energy(model, ix, bank, U, Cs, slot_b) - E_bank0).view(N, draws)
            fit = (dE_obs.sum() - (torch.logsumexp(dE_bank, 1) - math.log(draws)).sum()) / N
            change = sum((getattr(model, k) - start[k]).square().sum() for k in start) \
                + (Cv - C_start).square().sum()
            return -fit + pooling_penalty(model, pool_prod) + lam_trust * change / N

        block_a = [Cv] + [getattr(model, k) for k in names]
        block_b = [model.alpha]
        for _ in range(cycles):
            for block in (block_a, block_b):
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
            dE_bank = (basket_energy(model, ix, bank, U, Cs) - E_bank0).view(N, draws)
            dE_obs = basket_energy(model, ix, observed, U, Cs) - E_obs0
            ess = bank_statistics(dE_bank)
            gain = float((dE_obs.sum() - (torch.logsumexp(dE_bank, 1) - math.log(draws)).sum()) / N)
            finite = bool(torch.isfinite(dE_bank).all() and torch.isfinite(dE_obs).all())
        ok, summary = ess_rule(ess)
        log(f"[joint-refinement]   trust {lam_trust:g}: bank gain {gain:+.5f}, ESS {summary}"
            + ("" if finite else ", non-finite"))
        if finite and ok:
            return True, {"trust": lam_trust, "bank_gain": gain, "ess": summary, "C": Cs.detach()}
    with torch.no_grad():
        for name, value in start.items():
            getattr(model, name).copy_(value)
    return False, {"status": "no trust level kept the bank valid"}
