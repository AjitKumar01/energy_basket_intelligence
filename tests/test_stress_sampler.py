"""The vectorized exact sampler matches the native Version-4 dynamic program on stress-world trips."""
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "data/stress_world/model_input_category"
pytestmark = pytest.mark.skipif(not BUNDLE.exists(), reason="stress bundle not present")
os.environ.setdefault("ENERGY_MODEL_DATA_ROOT", str(BUNDLE))
sys.path.insert(0, str(ROOT / "scripts/v4_training"))


def test_sampler_matches_exact_marginals_and_size_law():
    import stress
    from interaction_particles import differentiable_log_size_beta0
    A = ROOT.parent / "energy_basket_intelligence/artifacts/stress_clean_confirm"
    data, model, meta, batcher = stress.setup(A / "artifacts/initialization.pt",
                                              A / "out/v3_pipeline_additive_best.pt", 4)
    J, r = int(data["n_item"]), 8
    with torch.no_grad():
        model.phi[:, :r] = 0.4 * torch.randn(J, r, generator=torch.Generator().manual_seed(0))
    trips = np.repeat(np.flatnonzero(data["trip_split"] == 2)[:3], 3000)
    bt = stress.Batch(batcher, trips, J)
    bt.bind(model)
    z = torch.randn(3, r, generator=torch.Generator().manual_seed(1)).repeat_interleave(3000, 0)
    u = stress.slot_utilities(model, bt.ix, z, r).detach()
    X = stress.sample_slots(model, bt.ix, u, torch.Generator().manual_seed(2))
    uu = u.clone().requires_grad_(True)
    lz = stress.log_zc(model, bt.ix, uu)
    p, = torch.autograd.grad(lz.sum(), uu)
    for b in range(3):
        sel = (bt.ix.item_trip >= 3000 * b) & (bt.ix.item_trip < 3000 * (b + 1))
        first = bt.ix.item_trip == 3000 * b
        freq = torch.zeros(J).index_add_(0, bt.ix.item[sel], X[sel].double()) / 3000
        exact = torch.zeros(J).index_add_(0, bt.ix.item[first], p[first])
        assert float((freq - exact).abs().max()) < 0.03
        sizes = torch.zeros(3000 * 3).index_add_(0, bt.ix.item_trip, X.double())[3000 * b:3000 * (b + 1)]
        log_size = differentiable_log_size_beta0(model, bt.ix, u)[3000 * b]
        assert abs(float(sizes.mean()) - float((torch.softmax(log_size, 0) * torch.arange(1, len(log_size) + 1)).sum())) < 0.1
