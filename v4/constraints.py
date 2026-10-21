"""Bounds that keep the within-category term safe on the whole basket support.

The energy has -rho_c C(n_c, 2).  A negative rho_c (attraction) grows quadratically in the
category count, so a broad category with a modestly negative rho_c could reward huge
baskets.  After every update rho_c is projected so that each category's largest possible
attractive reward, over every count a store can offer, is at most ``max_reward`` nats (and
never below the floor -1.5).  The energy itself is unchanged.
"""
from __future__ import annotations

import numpy as np
import torch


def category_capacities(data, nmax: int) -> np.ndarray:
    """Largest number of products any store offers in each category, capped at nmax."""
    C, S = int(data["n_cat"]), int(data["n_store"])
    ptr = np.asarray(data["store_cat_ptr"])
    out = np.zeros(C, dtype=np.int64)
    for s in range(S):
        widths = ptr[s * C + 1:s * C + C + 1] - ptr[s * C:s * C + C]
        out = np.maximum(out, np.minimum(widths, nmax))
    return out


def rho_c_lower_bounds(capacities, max_reward: float, floor: float = -1.5) -> np.ndarray:
    pairs = np.asarray(capacities, dtype=np.float64)
    pairs = pairs * np.maximum(pairs - 1.0, 0.0) * 0.5
    bound = np.full(len(pairs), float(floor))
    some = pairs > 0
    bound[some] = np.maximum(bound[some], -float(max_reward) / pairs[some])
    return bound


@torch.no_grad()
def project_category_reward_(model, capacities, max_reward: float, floor: float = -1.5,
                             optimizer=None) -> float:
    """Clamp rho_c to its bounds in place; returns the largest remaining reward.

    With an Adam optimizer, first moments that keep pushing a coefficient through its
    bound are cleared, so stale momentum does not repeat the same blocked move."""
    lower = torch.as_tensor(rho_c_lower_bounds(capacities, max_reward, floor), dtype=model.rho_c.dtype)
    model.rho_c.copy_(torch.maximum(model.rho_c, lower))
    if optimizer is not None:
        first = optimizer.state.get(model.rho_c, {}).get("exp_avg")
        if torch.is_tensor(first):
            first.masked_fill_((model.rho_c <= lower + 1e-14) & (first > 0), 0.0)
    pairs = torch.as_tensor(np.asarray(capacities, dtype=np.float64))
    pairs = pairs * (pairs - 1.0).clamp_min(0.0) * 0.5
    return float((torch.clamp(-model.rho_c, min=0.0) * pairs).max())
