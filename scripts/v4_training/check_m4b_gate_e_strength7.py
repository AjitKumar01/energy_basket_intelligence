"""Gate E strength 7 (M4b): is the one failing context IS noise? Same world, more burn / draws / other IS seeds."""
import os, sys, json, time
os.environ["OMP_NUM_THREADS"] = "4"; os.environ["MKL_NUM_THREADS"] = "4"; os.environ["OPENBLAS_NUM_THREADS"] = "4"; os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
sys.path.insert(0, "/Users/ajit/Projects/nf_dunnhumby/ebi_v4t/scripts/v4_training")
import numpy as np, torch
torch.set_num_threads(4)
import validate_gate_e as G
from world import make_truth, exact_log_z
seed = 101; J, H, W, nmax = 18, 100, 8, 5
sup = G.Support(J, nmax); ch, cw = G.contexts(H, W)
panel = np.random.default_rng(seed + 4).choice(len(ch), size=200, replace=False)
hp = torch.as_tensor(ch[panel]); cwp = cw[panel]
truth, wp = make_truth(seed, 7.0, J=J, H=H, W=W, K=3, r=3, nmax=nmax, use_f_extra=True, presence_phi=True, cat_rho0=True, u_cat=True, iv_slope=True)
pp = wp[torch.as_tensor(cwp)]
exact = exact_log_z(truth, sup, hp, pp)
out = []
for name, burn, draws, s in [("default", 200, 2048, 12), ("burn4x", 800, 2048, 12), ("draws2x", 200, 4096, 12), ("burn4x_draws2x", 800, 4096, 12),
                             ("default_seedB", 200, 2048, 77), ("default_seedC", 200, 2048, 991), ("burn4x_draws2x_seedB", 800, 4096, 77)]:
    t = time.time()
    est, ess = G.is_log_z(truth, hp, pp, torch.Generator().manual_seed(seed + s), burn=burn, components=256, thin=4, draws=draws)
    err = (est - exact).abs()
    i = int(err.argmax())
    row = {"run": name, "mean_abs_err": round(float(err.mean()), 5), "max_abs_err": round(float(err.max()), 5), "worst_ctx": i,
           "ctx91_err": round(float(err[91]), 5), "ctx91_ess": round(float(ess[91]), 4), "min_ess": round(float(ess.min()), 4),
           "n_err_gt_0.05": int((err > 0.05).sum()), "sec": round(time.time() - t, 1)}
    print(json.dumps(row), flush=True); out.append(row)
json.dump(out, open("/Users/ajit/Projects/nf_dunnhumby/ebi_v4t/runs/stress/gate_e_m4b_strength7_check.json", "w"), indent=1)
