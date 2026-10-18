#!/usr/bin/env python3
"""Score Category Chain, Certified Version-4, Exact Phi=0 Parent, and a trained HS checkpoint
on the 2,048 held-out split=0 validation trips, 4,096 split=1 validation trips, and 4,096 split=2 test trips."""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
from pathlib import Path

sys.modules.setdefault("pathlib._local", pathlib)
os.environ.setdefault("V3_AFFINITY", "1")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/version4"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT.parent / "basket_chain"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import stress  # noqa: E402
from stress import Batch, is_log_z, log_zc, setup  # noqa: E402
from pipeline_support import supported_trips  # noqa: E402
from basket_chain.data import load_bundle  # noqa: E402
from scripts.evaluate_stress import load_model as load_chain_model, loglik as chain_loglik  # noqa: E402

torch.set_default_dtype(torch.float64)


def main():
    A = ROOT.parent / "energy_basket_intelligence/artifacts/stress_clean_confirm"
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--initialization", type=Path, default=A / "artifacts/initialization.pt")
    p.add_argument("--additive", type=Path, default=A / "out/v3_pipeline_additive_best.pt")
    p.add_argument("--certified", type=Path, default=A / "artifacts/candidate_refined.pt")
    p.add_argument("--chain-run", type=Path, default=ROOT.parent / "basket_chain/runs/stress/fit_h64_p10")
    p.add_argument("--hs-checkpoint", type=Path, default=ROOT / "runs/stress/hs_conv.pt")
    p.add_argument("--hs-test-per-trip", type=Path, default=ROOT / "runs/stress/hs_conv.per_trip.npz")
    p.add_argument("--ref-test", type=Path, default=A / "reports/likelihood_test_paired_per_trip.npz")
    p.add_argument("--ref-val", type=Path, default=A / "reports/likelihood_validation_per_trip.npz")
    p.add_argument("--bundle", type=Path, default=ROOT / "data/stress_world/model_input_category")
    p.add_argument("--seed", type=int, default=41001)
    p.add_argument("--val-trips", type=int, default=2048)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--output", type=Path, default=ROOT / "runs/stress/validation_vs_test_table.json")
    args = p.parse_args()

    stress.USE_CPP_SAMPLER = True
    t0 = time.time()
    data, model, meta, batcher = setup(args.initialization, args.additive, args.threads)
    J, nmax = int(data["n_item"]), int(meta["nmax"])

    all_train_trips = supported_trips(data, 0, nmax)
    v_perm = np.random.default_rng(args.seed + 99).permutation(all_train_trips)
    val_trips_2048 = np.sort(v_perm[:args.val_trips])
    val_batches = [Batch(batcher, val_trips_2048[s:s + 256], J) for s in range(0, len(val_trips_2048), 256)]

    ref_test = np.load(args.ref_test)
    ref_val_split1 = np.load(args.ref_val)
    test_trips_4096 = ref_test["trips"].astype(np.int64)
    val_split1_trips = ref_val_split1["trips"].astype(np.int64)

    # 1. Category Chain on all 3 trip sets
    bundle = load_bundle(args.bundle)
    chain_model, _ = load_chain_model(args.chain_run, bundle)
    chain_val_2048 = chain_loglik(chain_model, bundle, val_trips_2048)
    chain_val_split1 = chain_loglik(chain_model, bundle, val_split1_trips)
    chain_test_4096 = chain_loglik(chain_model, bundle, test_trips_4096)

    # 2. Exact Phi=0 Additive Parent on val_trips_2048
    parent_lls = []
    with torch.no_grad():
        for bt in val_batches:
            bt.bind(model)
            e = model.energy(bt.li, bt.lt, bt.lc, bt.ix.B, bt.line_ctx)
            lz = log_zc(model, bt.ix, model.b_flat(bt.ix))
            parent_lls.append((e - lz).numpy())
    parent_val_2048 = np.concatenate(parent_lls)

    # 3. Certified Version-4 on val_trips_2048
    ck_cert = torch.load(args.certified, map_location="cpu", weights_only=False)
    model.load_state_dict(ck_cert["model"])
    model.eval()
    vgen = torch.Generator().manual_seed(args.seed + 777)
    cert_lls, cert_ess = [], []
    with torch.no_grad():
        for bt in val_batches:
            bt.bind(model)
            est, _, ess, _ = is_log_z(model, bt, 8, vgen, burn=80, components=32, thin=2, draws=256)
            e = model.energy(bt.li, bt.lt, bt.lc, bt.ix.B, bt.line_ctx)
            cert_lls.append((e - est).numpy())
            cert_ess.append(ess.numpy())
    cert_val_2048 = np.concatenate(cert_lls)
    cert_ess_2048 = np.concatenate(cert_ess)

    # 4. HS checkpoint on val_trips_2048
    ck_hs = torch.load(args.hs_checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ck_hs["model"])
    model.eval()
    r_hs = int(ck_hs["rank"])
    vgen_hs = torch.Generator().manual_seed(args.seed + 777)
    hs_lls, hs_ess = [], []
    with torch.no_grad():
        for bt in val_batches:
            bt.bind(model)
            est, _, ess, _ = is_log_z(model, bt, r_hs, vgen_hs, burn=80, components=32, thin=2, draws=256)
            e = model.energy(bt.li, bt.lt, bt.lc, bt.ix.B, bt.line_ctx)
            hs_lls.append((e - est).numpy())
            hs_ess.append(ess.numpy())
    hs_val_2048 = np.concatenate(hs_lls)
    hs_test_4096 = np.load(args.hs_test_per_trip)["ll"]

    out = {
        "val_2048_split0": {
            "category_chain": round(float(chain_val_2048.mean()), 4),
            "hs_conv": round(float(hs_val_2048.mean()), 4),
            "certified_v4": round(float(cert_val_2048.mean()), 4),
            "certified_v4_ess_mean": round(float(cert_ess_2048.mean()), 4),
            "exact_phi0_parent": round(float(parent_val_2048.mean()), 4),
        },
        "val_4096_split1": {
            "category_chain": round(float(chain_val_split1.mean()), 4),
            "certified_v4": round(float(ref_val_split1["target_child"].mean()), 4),
            "exact_phi0_parent": round(float(ref_val_split1["exact_parent"].mean()), 4),
        },
        "test_4096_split2": {
            "category_chain": round(float(chain_test_4096.mean()), 4),
            "hs_conv": round(float(hs_test_4096.mean()), 4),
            "certified_v4": round(float(ref_test["target_child"].mean()), 4),
            "exact_phi0_parent": round(float(ref_test["exact_parent"].mean()), 4),
        },
        "seconds": round(time.time() - t0, 1),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
