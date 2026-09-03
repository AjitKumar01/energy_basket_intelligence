"""Complete-support safety constraints for the existing Version-4 category term.

The model energy contains ``-rho_c[c] * choose(n_c, 2)``.  The affinity partition can
contain broad groups, so a modest negative coefficient can otherwise imply a very large
attractive reward at the remote edge of the complete basket support.  This module does
not alter that energy.  It restricts optimization to parameters for which each category's
largest possible attractive contribution is bounded by a declared number of nats.
"""
from __future__ import annotations

import numpy as np
import torch


def category_capacities(data, categories: int, nmax: int) -> np.ndarray:
    """Largest available category count over every store, truncated to support."""
    pointer = np.asarray(data["store_cat_ptr"])
    stores = int(data["n_store"])
    answer = np.zeros(categories, dtype=np.int64)
    for store in range(stores):
        start = store * categories
        widths = (pointer[start + 1:start + categories + 1]
                  - pointer[start:start + categories])
        answer = np.maximum(answer, np.minimum(widths, nmax))
    return answer


def category_pair_counts(capacities: np.ndarray) -> np.ndarray:
    capacities = np.asarray(capacities, dtype=np.float64)
    if np.any(capacities < 0):
        raise ValueError("category capacities must be nonnegative")
    return capacities * np.maximum(capacities - 1.0, 0.0) * 0.5


def support_lower_bounds(capacities: np.ndarray, max_category_reward: float,
                         floor: float = -1.5) -> np.ndarray:
    """Lower rho_c bounds that cap ``(-rho_c)+ choose(m_c,2)`` on full support."""
    if max_category_reward <= 0:
        raise ValueError("max_category_reward must be positive")
    if floor > 0:
        raise ValueError("rho_c floor must be nonpositive")
    pairs = category_pair_counts(capacities)
    answer = np.full(len(pairs), float(floor), dtype=np.float64)
    nonzero = pairs > 0
    answer[nonzero] = np.maximum(
        answer[nonzero], -float(max_category_reward) / pairs[nonzero])
    return answer


def attractive_category_rewards(rho_c: torch.Tensor,
                                capacities: np.ndarray) -> torch.Tensor:
    """Maximum attractive reward of each category over the supported basket sizes."""
    pairs = torch.as_tensor(category_pair_counts(capacities),
                            dtype=rho_c.dtype, device=rho_c.device)
    if pairs.shape != rho_c.shape:
        raise ValueError("one capacity is required per rho_c coefficient")
    return torch.clamp(-rho_c, min=0.0) * pairs


@torch.no_grad()
def project_category_reward_(model, capacities: np.ndarray,
                             max_category_reward: float, floor: float = -1.5,
                             optimizer=None) -> dict:
    """Project rho_c onto the complete-support admissible set in-place.

    If an Adam/AdamW optimizer is supplied, outward first moments at an active lower
    bound are cleared.  This prevents stale momentum from repeatedly proposing the same
    inadmissible move while retaining second-moment preconditioning.
    """
    lower = torch.as_tensor(
        support_lower_bounds(capacities, max_category_reward, floor),
        dtype=model.rho_c.dtype, device=model.rho_c.device)
    before = attractive_category_rewards(model.rho_c, capacities)
    below = model.rho_c < lower
    model.rho_c.copy_(torch.maximum(model.rho_c, lower))
    cleared = 0
    if optimizer is not None:
        state = optimizer.state.get(model.rho_c, {})
        first = state.get("exp_avg")
        if torch.is_tensor(first):
            # Adam subtracts exp_avg.  A positive moment therefore moves rho downward.
            active_outward = (model.rho_c <= lower + 1e-14) & (first > 0)
            cleared = int(active_outward.sum().item())
            first.masked_fill_(active_outward, 0.0)
    after = attractive_category_rewards(model.rho_c, capacities)
    return {
        "projected_coefficients": int(below.sum().item()),
        "cleared_outward_moments": cleared,
        "maximum_reward_before": float(before.max().item()),
        "maximum_reward_after": float(after.max().item()),
        "active_lower_bounds": int((model.rho_c <= lower + 1e-14).sum().item()),
        "minimum_rho_lower_bound": float(lower.min().item()),
    }
