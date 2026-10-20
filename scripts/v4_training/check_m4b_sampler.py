"""Do the C++ and Python exact samplers draw from the M4b law (lambda != 1)? Compare empirical item
marginals on real stress trips against exact marginals d log Z / d u."""
import os, sys, json
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"): os.environ[k] = "4"
R = "/Users/ajit/Projects/nf_dunnhumby/ebi_v4t"
sys.path.insert(0, R + "/scripts/v4_training")
import numpy as np, torch
import stress
A = "/Users/ajit/Projects/nf_dunnhumby/energy_basket_intelligence/artifacts/stress_clean_confirm"
blob = torch.load(R + "/runs/stress/hs_rec.pt", map_location="cpu", weights_only=False)
data, model, meta, batcher = stress.setup(A + "/artifacts/initialization.pt", A + "/out/v3_pipeline_additive_best.pt", 4, recency=True)
stress.configure_model_from_blob(model, blob); model.eval()
for q in model.parameters(): q.requires_grad_(False)
r = int(blob["rank"]); C = int(model.rho_c.numel())
g = torch.Generator().manual_seed(5)
model.iv_slope_raw = torch.nn.Parameter(-0.5 + 1.0 * torch.randn(C, generator=g, dtype=torch.float64), requires_grad=False)
model.has_iv_slope = True
lam = torch.sigmoid(model.iv_slope_raw)
J = int(data["n_item"])
trips = np.load(A + "/reports/likelihood_test_paired_per_trip.npz")["trips"][:64].astype(np.int64)
bt = stress.Batch(batcher, trips, J); bt.bind(model); ix = bt.ix
z = torch.randn(ix.B, r, generator=g, dtype=torch.float64)
u = stress.slot_utilities(model, ix, z, r).detach().requires_grad_(True)
iv = stress.row_iv_shift(model, ix).detach()
with torch.enable_grad():
    lz = stress.log_zc(model, ix, u, z=z, iv_shift=iv)
    p_exact, = torch.autograd.grad(lz.sum(), u)
u = u.detach()
out = {"trips": int(ix.B), "slots": int(len(ix.item)), "lambda_mean": float(lam.mean()), "lambda_min": float(lam.min()),
       "iv_shift_abs_mean": float(iv.abs().mean())}
for name, use_cpp, N in (("cpp", True, 20000), ("python", False, 2000)):
    acc = torch.zeros(len(ix.item), dtype=torch.float64)
    for _ in range(N):
        acc += stress.sample_slots(model, ix, u, g, use_cpp=use_cpp, z=z).double()
    p_hat = acc / N
    se = (p_exact * (1 - p_exact) / N).clamp_min(1e-12).sqrt()
    zs = (p_hat - p_exact) / se
    m = p_exact > 1e-4
    out[name] = {"draws": N, "max_abs_diff": float((p_hat - p_exact).abs().max()), "max_abs_z": float(zs[m].abs().max()),
                 "mean_z2": float(zs[m].square().mean()), "slots_tested": int(m.sum())}
    print(name, out[name], flush=True)
# control: the same check ignoring the shift in the exact marginals must fail, or the test has no power
with torch.enable_grad():
    u2 = u.clone().requires_grad_(True)
    model.has_iv_slope = False
    p_noshift, = torch.autograd.grad(stress.log_zc(model, ix, u2, z=z).sum(), u2)
    model.has_iv_slope = True
out["exact_marginal_shift_effect_max"] = float((p_noshift - p_exact).abs().max())
print(json.dumps(out, indent=1))
json.dump(out, open(R + "/runs/stress/gate_e_m4b_sampler_check.json", "w"), indent=1)
