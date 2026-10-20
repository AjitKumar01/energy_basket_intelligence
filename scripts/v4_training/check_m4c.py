#!/usr/bin/env python3
"""M4c checks on real stress trips (hs_rec weights, random lambda in (0, 1), price-only IV shift):
  1. the shift is exactly 0 when the context is set to regular price, and nonzero otherwise;
  2. the exact objective (energy - log Z_c at fixed z) plus the pseudo-likelihood: autograd
     directional derivative against central finite differences, per parameter.  alpha, theta,
     delta and xi are excluded: their gauge centring uses a detached mean (ragged.py), so their
     training gradient differs from the finite difference by design, with or without M4c;
  3. the C++ and Python exact samplers against the exact marginals d log Z / d u."""
import json, os, sys
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(k, "4")
from pathlib import Path
import numpy as np, torch
import stress

R = Path(__file__).resolve().parents[2]
A = R.parent / "energy_basket_intelligence/artifacts/stress_clean_confirm"
torch.set_num_threads(4)
blob = torch.load(R / "runs/stress/hs_rec.pt", map_location="cpu", weights_only=False)
data, model, meta, batcher = stress.setup(A / "artifacts/initialization.pt", A / "out/v3_pipeline_additive_best.pt", 4, recency=True)
stress.configure_model_from_blob(model, blob)
model.eval()
r = int(blob["rank"]); C = int(model.rho_c.numel()); J = int(data["n_item"])
g = torch.Generator().manual_seed(7)
model.iv_slope_raw = torch.nn.Parameter(-0.5 + torch.randn(C, generator=g, dtype=torch.float64))
model.has_iv_slope = True; model.iv_price_only = True
trips = np.load(A / "reports/likelihood_test_paired_per_trip.npz")["trips"][:48].astype(np.int64)
bt = stress.Batch(batcher, trips, J); bt.bind(model); ix = bt.ix
out = {"trips": len(trips)}

# 1. zero at regular price
with torch.no_grad():
    out["mean_abs_shift_actual"] = float(stress.row_iv_shift(model, ix).abs().mean())
    bt.bind(model, {k: (torch.zeros_like(v) if k in stress.PRICE_KEYS else v) for k, v in bt.ctx.items()})
    out["max_abs_shift_regular_price"] = float(stress.row_iv_shift(model, ix).abs().max())
    bt.bind(model)

# 2. gradients
z = torch.randn(ix.B, r, generator=g, dtype=torch.float64)
params = [p for p in model.parameters() if p.dtype == torch.float64]
for p in params: p.requires_grad_(True)


def objective():
    u = stress.slot_utilities(model, ix, z, r)
    return (stress.obs_energy(model, bt, r) - stress.log_zc(model, ix, u, z=z)).sum() + stress.pseudo_loglik(model, ix, bt.obs, r)


GAUGE = {"alpha", "theta", "delta", "xi"}
rows = []
for name, p in model.named_parameters():
    if p.dtype != torch.float64 or name in GAUGE:
        continue
    d = torch.randn(p.shape, generator=g, dtype=torch.float64)
    gr, = torch.autograd.grad(objective(), [p], allow_unused=True)
    ad = 0.0 if gr is None else float((gr * d).sum())
    with torch.no_grad():
        p.add_(1e-5 * d); fp = float(objective()); p.sub_(2e-5 * d); fm = float(objective()); p.add_(1e-5 * d)
    fd = (fp - fm) / 2e-5
    rows.append({"param": name, "autograd": ad, "finite_diff": fd, "rel_err": abs(ad - fd) / max(1.0, abs(fd))})
out["gradient_checks"] = rows
for p in params: p.requires_grad_(False)

# 3. samplers
u = stress.slot_utilities(model, ix, z, r).detach().requires_grad_(True)
iv = stress.row_iv_shift(model, ix).detach()
with torch.enable_grad():
    p_exact, = torch.autograd.grad(stress.log_zc(model, ix, u, z=z, iv_shift=iv).sum(), u)
u = u.detach()
for name, use_cpp, N in (("cpp", True, 20000), ("python", False, 2000)):
    acc = torch.zeros(len(ix.item), dtype=torch.float64)
    for _ in range(N):
        acc += stress.sample_slots(model, ix, u, g, use_cpp=use_cpp, z=z).double()
    p_hat = acc / N
    m = p_exact * N > 20                    # enough expected draws for the normal approximation
    zs = (p_hat - p_exact)[m] / (p_exact * (1 - p_exact) / N)[m].sqrt()
    out[f"sampler_{name}"] = {"draws": N, "slots": int(m.sum()), "mean_z2": float(zs.square().mean()), "max_abs_z": float(zs.abs().max())}
out["pass"] = bool(out["max_abs_shift_regular_price"] == 0.0 and out["mean_abs_shift_actual"] > 0 and all(r_["rel_err"] < 1e-6 for r_ in rows)
                   and all(0.9 < out[f"sampler_{n}"]["mean_z2"] < 1.1 for n in ("cpp", "python")))
print(json.dumps(out, indent=1))
(R / "runs/stress/check_m4c.json").write_text(json.dumps(out, indent=1) + "\n")
