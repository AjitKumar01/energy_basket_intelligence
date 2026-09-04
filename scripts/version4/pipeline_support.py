"""Small shared helpers used by the selected Version-4 stages."""
from __future__ import annotations

import numpy as np
import torch

from ragged import smolyak_grid


def supported_trips(data, split: int, nmax: int) -> np.ndarray:
    return np.flatnonzero(
        (data["trip_split"] == int(split))
        & (data["trip_nlines"] >= 1)
        & (data["trip_nlines"] <= int(nmax)))


def smolyak_rule(model, rank: int, level: int):
    active, weights = smolyak_grid(rank, level)
    nodes = torch.zeros(len(weights), model.Kz, dtype=model.phi.dtype,
                        device=model.phi.device)
    nodes[:, :rank] = active.to(device=model.phi.device)
    return nodes, weights.to(device=model.phi.device)


def install_quadrature(model, quadrature) -> None:
    model.quad = quadrature
    model.quad_a = None


def copied_context(context):
    return {key: value.clone() if torch.is_tensor(value) else value
            for key, value in context.items()}


def particle_delta(states, delta_slot, batches: int) -> torch.Tensor:
    answer = torch.zeros(len(states), batches, dtype=delta_slot.dtype,
                         device=delta_slot.device)
    for particle_index, particle in enumerate(states):
        for batch_index, slots in enumerate(particle):
            answer[particle_index, batch_index] = delta_slot[slots].sum()
    return answer


def named_basket(items, metadata, limit: int = 8) -> list[dict[str, object]]:
    names = []
    for item in items[:limit]:
        text = str(metadata.SUB_COMMODITY_DESC.iloc[int(item)]).strip()
        names.append({"item": int(item), "description": text})
    return names


@torch.no_grad()
def collect_size_law(model, batcher, trips, quadrature, chunk: int, label: str):
    """Evaluate normalized size laws for a deterministic trip panel."""
    install_quadrature(model, quadrature)
    probability, observed = [], []
    for start in range(0, len(trips), chunk):
        sub = trips[start:start + chunk]
        ix, ctx, _line_ctx, house, _li, lt, _lc, _lq = batcher.make(sub)
        model.house, model.ctx = house, ctx
        _logz, size_probability = model.log_Z(
            ix, drop_empty=True, return_size=True)
        probability.append(size_probability.cpu().numpy())
        observed.append(torch.bincount(lt, minlength=ix.B).cpu().numpy())
        if ((start // chunk + 1) % 20 == 0 or start + chunk >= len(trips)):
            print(f"[size-law] {label} {min(start + chunk, len(trips))}/"
                  f"{len(trips)}", flush=True)
    probability = np.concatenate(probability)
    log_probability = np.log(np.clip(probability, 1e-300, None))
    log_probability -= np.logaddexp.reduce(log_probability, axis=1)[:, None]
    return np.concatenate(observed), log_probability
