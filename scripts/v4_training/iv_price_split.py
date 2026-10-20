#!/usr/bin/env python3
"""How much of a category's inclusive value IV_c = log sum_j exp(b_j) varies with price, and does
category presence respond to that part as much as to the rest?

For training-split trips, each (trip, category) row gets IV at the actual context and IV at the
regular-price context (price deviation, display and mailer set to 0).  price part = IV_actual -
IV_regular; rest = IV_regular (taste, recency, season, store).  Reports their spread (overall and
within category) and a logistic regression of observed category presence on both parts with
category intercepts.  Uses training trips only.
"""
import argparse, json, os
from pathlib import Path
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(k, "4")
import numpy as np, torch
import stress
from ragged import seg_max, seg_sum

A = Path(__file__).resolve().parents[3] / "energy_basket_intelligence/artifacts/stress_clean_confirm"


def row_iv(model, ix):
    b = model.b_flat(ix).detach()
    m = seg_max(b.unsqueeze(0), ix.row_of, ix.n_rows)[0]
    return m + torch.log(seg_sum(torch.exp(b - m[ix.row_of]).unsqueeze(0), ix.row_of, ix.n_rows)[0])


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--trips", type=int, default=4096)
    p.add_argument("--initialization", type=Path, default=A / "artifacts/initialization.pt")
    p.add_argument("--additive", type=Path, default=A / "out/v3_pipeline_additive_best.pt")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    blob = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    data, model, meta, batcher = stress.setup(a.initialization, a.additive, 4,
                                              recency=bool(blob.get("recency", False)))
    stress.configure_model_from_blob(model, blob); model.eval()
    cache = np.load(Path(os.environ["ENERGY_MODEL_DATA_ROOT"]) / "basket_input/v3_index_affinity.npz")
    tr = np.flatnonzero((cache["trip_split"] == 0) & (np.diff(cache["line_ptr"]) > 0))
    trips = np.sort(np.random.default_rng(0).choice(tr, a.trips, replace=False)).astype(np.int64)
    J = int(data["n_item"])
    act, reg, pres, cat = [], [], [], []
    with torch.no_grad():
        for s in range(0, len(trips), 256):
            bt = stress.Batch(batcher, trips[s:s + 256], J); ix = bt.ix
            bt.bind(model); iv_a = row_iv(model, ix)
            ctx = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in bt.ctx.items()}
            for k in ("dlp", "dlp_bar", "disp", "mail"):
                if k in ctx: ctx[k] = torch.zeros_like(ctx[k])
            bt.bind(model, ctx); iv_r = row_iv(model, ix)
            n_row = torch.zeros(ix.n_rows).index_add_(0, ix.row_of, bt.obs.double())
            act.append(iv_a.numpy()); reg.append(iv_r.numpy()); pres.append((n_row > 0).numpy()); cat.append(ix.row_cat.numpy())
    act, reg, pres, cat = map(np.concatenate, (act, reg, pres, cat))
    price = act - reg

    def within(x):
        mu = np.bincount(cat, x) / np.bincount(cat)
        return x - mu[cat]
    out = {"checkpoint": str(a.checkpoint), "trips": len(trips), "rows": int(len(act)), "presence_rate": float(pres.mean()),
           "sd_price_part": float(price.std()), "sd_rest": float(reg.std()),
           "sd_price_part_within_category": float(within(price).std()), "sd_rest_within_category": float(within(reg).std()),
           "corr_price_rest_within_category": float(np.corrcoef(within(price), within(reg))[0, 1])}
    out["price_share_of_within_category_variance"] = out["sd_price_part_within_category"] ** 2 / (
        out["sd_price_part_within_category"] ** 2 + out["sd_rest_within_category"] ** 2)
    from sklearn.linear_model import LogisticRegression
    C = int(cat.max()) + 1
    X = np.c_[price, reg, np.eye(C)[cat]]
    lr = LogisticRegression(C=1e4, max_iter=5000, fit_intercept=False).fit(X, pres)
    out["presence_logit_slope_price_part"] = float(lr.coef_[0, 0])
    out["presence_logit_slope_rest"] = float(lr.coef_[0, 1])
    lam = torch.sigmoid(blob["model"]["iv_slope_raw"]).numpy() if "iv_slope_raw" in blob["model"] else None
    if lam is not None:
        out["lambda_mean"] = float(lam.mean())
    a.output.write_text(json.dumps(out, indent=1) + "\n"); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
