"""The vectorized exact sampler and C++ sampler match the native Version-4 dynamic program on stress-world trips."""
import os
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch

try:
    import pytest
except ModuleNotFoundError:
    pytest = types.ModuleType("pytest")

    class _Mark:
        @staticmethod
        def skipif(cond, reason=""):
            return lambda f: f

        @staticmethod
        def parametrize(argnames, argvalues, ids=None):
            def deco(f):
                f._parametrize = (argnames, argvalues, ids or [str(v) for v in argvalues])
                return f
            return deco

    pytest.mark = _Mark()

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "data/stress_world/model_input_category"
pytestmark = pytest.mark.skipif(not BUNDLE.exists(), reason="stress bundle not present")
os.environ.setdefault("ENERGY_MODEL_DATA_ROOT", str(BUNDLE))
sys.path.insert(0, str(ROOT / "scripts/v4_training"))


@pytest.mark.parametrize("use_cpp", [False, True], ids=["python", "cpp"])
def test_sampler_matches_exact_marginals_and_size_law(use_cpp):
    import stress
    from interaction_particles import differentiable_log_size_beta0
    A = ROOT.parent / "energy_basket_intelligence/artifacts/stress_clean_confirm"
    data, model, meta, batcher = stress.setup(A / "artifacts/initialization.pt",
                                              A / "out/v3_pipeline_additive_best.pt", 1)
    J, r = int(data["n_item"]), 8
    with torch.no_grad():
        model.phi[:, :r] = 0.4 * torch.randn(J, r, generator=torch.Generator().manual_seed(0))
    trips = np.repeat(np.flatnonzero(data["trip_split"] == 2)[:3], 3000)
    bt = stress.Batch(batcher, trips, J)
    bt.bind(model)
    z = torch.randn(3, r, generator=torch.Generator().manual_seed(1)).repeat_interleave(3000, 0)
    u = stress.slot_utilities(model, bt.ix, z, r).detach()
    X = stress.sample_slots(model, bt.ix, u, torch.Generator().manual_seed(2), use_cpp=use_cpp)
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
        assert abs(float(sizes.mean()) - float((torch.softmax(log_size, 0) * torch.arange(1, len(log_size) + 1)).sum()))< 4 * float(sizes.std()) / 3000 ** 0.5   # 4 standard errors
    if use_cpp:
        Xg = stress.gibbs(model, bt.ix, X, r, torch.Generator().manual_seed(3), sweeps=2, beta=0.0, use_cpp=True)
        u0 = stress.slot_utilities(model, bt.ix, None, r, beta=0.0).detach()
        uu0 = u0.clone().requires_grad_(True)
        lz0 = stress.log_zc(model, bt.ix, uu0)
        p0, = torch.autograd.grad(lz0.sum(), uu0)
        for b in range(3):
            sel = (bt.ix.item_trip >= 3000 * b) & (bt.ix.item_trip < 3000 * (b + 1))
            first = bt.ix.item_trip == 3000 * b
            freq0 = torch.zeros(J).index_add_(0, bt.ix.item[sel], Xg[sel].double()) / 3000
            exact0 = torch.zeros(J).index_add_(0, bt.ix.item[first], p0[first])
            assert float((freq0 - exact0).abs().max()) < 0.03
            sizes0 = torch.zeros(3000 * 3).index_add_(0, bt.ix.item_trip, Xg.double())[3000 * b:3000 * (b + 1)]
            log_size0 = differentiable_log_size_beta0(model, bt.ix, u0)[3000 * b]
            assert abs(float(sizes0.mean()) - float((torch.softmax(log_size0, 0) * torch.arange(1, len(log_size0) + 1)).sum()))< 4 * float(sizes0.std()) / 3000 ** 0.5   # 4 standard errors


if __name__ == "__main__":
    t0 = time.time()
    for use_cpp, label in ((False, "python"), (True, "cpp")):
        ts = time.time()
        test_sampler_matches_exact_marginals_and_size_law(use_cpp=use_cpp)
        print(f"tests/test_stress_sampler.py::test_sampler_matches_exact_marginals_and_size_law[{label}] PASSED ({time.time() - ts:.2f}s)", flush=True)
    print(f"================== 2 passed in {time.time() - t0:.2f}s ==================", flush=True)
