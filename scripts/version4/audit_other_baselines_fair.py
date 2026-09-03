"""Paired same-manifest audit for the repaired basket baselines."""
import argparse
import hashlib
import json
import math
import os
import time

import numpy as np
import torch

from baselines import Batches, Bernoulli, DPP
from baselines2 import Multinomial, NDPP, Shopper, size_law
from data import build
from features import Features


HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "..", "out"))
ALL_MODELS = ("multinomial", "bernoulli", "dpp", "ndpp", "shopper")


def parse_models(raw):
    models = tuple(part.strip().lower() for part in raw.split(",") if part.strip())
    if not models:
        raise argparse.ArgumentTypeError("--models must select at least one baseline")
    unknown = sorted(set(models) - set(ALL_MODELS))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown baseline(s): {', '.join(unknown)}")
    if len(models) != len(set(models)):
        raise argparse.ArgumentTypeError("--models contains a duplicate baseline")
    return models


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_record(path, blob):
    stat = os.stat(path)
    return dict(path=os.path.basename(path), sha256=sha256(path), bytes=stat.st_size,
                mtime_ns=stat.st_mtime_ns, format=blob.get("format"),
                iteration=blob.get("iteration"), training_provenance="embedded")


def summarize(values, lines):
    values = np.asarray(values, dtype=np.float64)
    return dict(n=len(values), per_basket=float(values.mean()),
                se_per_basket=float(values.std(ddof=1) / np.sqrt(len(values))),
                per_line=float(values.sum() / np.asarray(lines).sum()))


def paired(main, baseline):
    difference = np.asarray(main) - np.asarray(baseline)
    return dict(main_minus_baseline=float(difference.mean()),
                paired_se=float(difference.std(ddof=1) / np.sqrt(len(difference))))


def hash_ids(ids):
    return hashlib.sha256(np.ascontiguousarray(ids, dtype=np.int64).tobytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def model_from_checkpoint(name, path, data, iteration, require_converged):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    require(blob.get("format") == 3 and blob.get("kind") == "verified-basket-baseline",
            f"{name}: checkpoint has no verified provenance")
    require(blob.get("model_name") == name,
            f"{name}: checkpoint contains a different model")
    if iteration:
        require(int(blob.get("iteration", -1)) == iteration,
                f"{name}: checkpoint is not the requested iteration-{iteration} model")
    cfg, md = blob["config"], blob["data"]
    fresh = (blob.get("lineage", {}).get("fresh_initialization")
             or not cfg.get("resume"))
    require(fresh and int(cfg["R"]) == int(cfg["nmax"]) == 120,
            f"{name}: not a fresh-lineage complete-support run")
    if require_converged:
        certificate = blob.get("convergence_certificate", {})
        require(certificate.get("required") and certificate.get("passed"),
                f"{name}: selected checkpoint has no passed convergence certificate")
        require(int(certificate.get("selected_iteration", -1)) ==
                int(blob.get("iteration", -2)),
                f"{name}: certificate does not select this checkpoint")
    require(md.get("affinity") == "1" and int(md["n_item"]) == 5455,
            f"{name}: wrong data universe")
    J, N, S = (int(data[k]) for k in ("n_item", "n_user", "n_store"))
    common = dict(K=cfg["K"], Kp=cfg["Kp"], seed=cfg["seed"],
                  taste_init=cfg["taste_init"])
    if name == "bernoulli":
        model = Bernoulli(J, N, S, **common)
    elif name == "multinomial":
        model = Multinomial(
            J, N, S, size_law(data, cfg["nmax"], cfg["R"]), **common)
    elif name == "dpp":
        model = DPP(J, N, S, rank=cfg["rank"],
                    interaction_init=cfg["interaction_init"], **common)
    elif name == "ndpp":
        model = NDPP(J, N, S, rank=cfg["rank"], srank=cfg["srank"],
                     interaction_init=cfg["interaction_init"], **common)
    elif name == "shopper":
        model = Shopper(J, N, S, Ki=cfg["interaction_rank"],
                        interaction_init=cfg["interaction_init"], **common)
    else:
        raise ValueError(name)
    model.load_state_dict(blob["model"], strict=True)
    return model.double().eval(), blob


@torch.no_grad()
def score(model, name, batcher, trips, chunk, shopper_orders, seed):
    values, lines = [], []
    gen = torch.Generator().manual_seed(seed)
    for k in range(0, len(trips), chunk):
        sub = trips[k:k + chunk]
        d = batcher.make(sub)
        if name == "multinomial":
            ll = model.loglik(d, category_cap=120)
        elif name == "bernoulli":
            ll = model.loglik(d, nmax=120, category_cap=120)
        elif name == "shopper":
            ll = model.loglik(d, n_orders=shopper_orders, gen=gen,
                              exact_max_n=6, max_size=120)
        else:
            ll = model.loglik(d)
        require(len(ll) == len(sub) and bool(torch.isfinite(ll).all()),
                f"{name}: invalid score")
        values.extend(ll.tolist())
        lines.extend(torch.bincount(d["lt"], minlength=len(sub)).tolist())
    return np.asarray(values), np.asarray(lines)


def ndpp_middle(model, dtype):
    k, m = model.rank, 2 * model.srank
    out = torch.zeros(k + m, k + m, dtype=dtype)
    out[:k, :k] = torch.eye(k, dtype=dtype)
    out[k:, k:] = model._C().to(dtype)
    return out


@torch.no_grad()
def dpp_tail_bounds(model, name, batcher, trips, cutoff=120, chunk=8):
    """Chernoff upper bound for P(N>cutoff | N>0) from det(I+zL)."""
    # A coarse log-z grid is sufficient here: accepted bounds must be many orders below
    # the score precision, not merely just below a threshold.
    t_grid = (0.1, 0.2, 0.4, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0)
    bounds = []
    for start in range(0, len(trips), chunk):
        d = batcher.make(trips[start:start + chunk])
        q = model.idx(d["item"], d["st"], d["house"], d["ctx"])
        for b in range(d["B"]):
            mask = d["st"] == b
            dg = torch.exp(q[mask].clamp(-12, 6))
            if name == "dpp":
                W = model.V[d["item"][mask]]
                middle = torch.eye(model.rank, dtype=dg.dtype)
            else:
                W = torch.cat([model.V[d["item"][mask]], model.B[d["item"][mask]]], 1)
                middle = ndpp_middle(model, dg.dtype)

            def logdet(z):
                s = 1.0 + z * dg
                A = torch.eye(W.shape[1], dtype=dg.dtype) \
                    + z * middle @ (W / s[:, None]).T @ W
                sign, ld = torch.linalg.slogdet(A)
                require(float(sign) > 0, f"{name}: scaled determinant is not positive")
                return torch.log(s).sum() + ld

            base = logdet(1.0)
            log_nonempty = torch.log1p(-torch.exp((-base).clamp(max=-1e-12)))
            candidates = []
            for t in t_grid:
                candidates.append(float(logdet(math.exp(t)) - base
                                        - t * (cutoff + 1) - log_nonempty))
            bounds.append(min(0.0, min(candidates)))
    return np.asarray(bounds)


def main(args):
    torch.set_default_dtype(torch.float64)
    torch.set_flush_denormal(True)
    require(os.environ.get("V3_AFFINITY") == "1", "set V3_AFFINITY=1")
    data = build()
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    full_path = (args.full_per_trip if os.path.isabs(args.full_per_trip)
                 else os.path.join(root, args.full_per_trip))
    full = np.load(full_path)
    required_arrays = {"trips", "lines", args.full_key}
    require(required_arrays.issubset(full.files),
            f"stored full score lacks {sorted(required_arrays - set(full.files))}")
    trips = np.asarray(full["trips"], dtype=np.int64)
    full_joint = np.asarray(full[args.full_key], dtype=np.float64)
    lines = np.asarray(full["lines"], dtype=np.int64)
    require(len(trips) == len(full_joint) == len(lines),
            "stored trip manifest and scores have different lengths")
    if args.maximum_trips:
        require(args.maximum_trips > 0, "--maximum-trips must be positive")
        trips = trips[:args.maximum_trips]
        full_joint = full_joint[:args.maximum_trips]
        lines = lines[:args.maximum_trips]
    expected_split = {"validation": 1, "test": 2}[args.split]
    require(bool(np.all(data["trip_split"][trips] == expected_split)),
            f"stored manifest is not entirely from the {args.split} split")
    batcher = Batches(data, Features(int(data["n_item"]), int(data["n_store"]), 712))

    result = dict(schema=2, created_unix=time.time(), iteration=args.iteration,
                  split=args.split, full_score_key=args.full_key,
                  manifest=dict(n=len(trips), sha256=hash_ids(trips), ids=trips.tolist()),
                  checkpoint_kind=args.checkpoint_kind,
                  convergence_required=bool(args.require_converged),
                  full=summarize(full_joint, lines), baselines={})
    arrays = dict(trips=trips, lines=lines, full=full_joint)
    for name in args.models:
        suffix = f"_{args.baseline_tag}" if args.baseline_tag else ""
        best_suffix = "_best" if args.checkpoint_kind == "best" else ""
        path = os.path.join(
            OUT, f"baseline_verified_{name}{suffix}{best_suffix}.pt")
        model, blob = model_from_checkpoint(
            name, path, data, args.iteration, args.require_converged)
        print(f"[other] scoring {name}", flush=True)
        values, got_lines = score(model, name, batcher, trips,
                                  args.bernoulli_chunk if name == "bernoulli" else args.chunk,
                                  args.shopper_orders, args.seed)
        require(np.array_equal(lines, got_lines), f"{name}: line counts differ")
        record = checkpoint_record(path, blob)
        record["iteration"] = int(blob["iteration"])
        if "convergence_certificate" in blob:
            record["convergence_certificate"] = blob["convergence_certificate"]
        block = dict(score=summarize(values, lines),
                     paired_full_minus_baseline=paired(full_joint, values),
                     checkpoint=record, support={})
        if name in ("dpp", "ndpp"):
            print(f"[other] auditing {name} cardinality tail", flush=True)
            log_bounds = dpp_tail_bounds(model, name, batcher, trips)
            block["support"] = dict(
                law="all nonempty cardinalities; compared to n<=120 via rigorous tail bound",
                max_log_chernoff_bound=float(log_bounds.max()),
                max_chernoff_bound=float(np.exp(log_bounds.max())),
                accepted_numerically_equivalent=bool(log_bounds.max() < math.log(1e-8)))
            arrays[name + "_log_tail_bound"] = log_bounds
        elif name == "shopper":
            block["support"] = dict(law="exact forced checkout at 120",
                                    ordering_samples=args.shopper_orders,
                                    exact_order_sum_through_size=6)
        elif name == "bernoulli":
            block["support"] = dict(law="exact 1<=n<=120 ESP normalizer")
        else:
            block["support"] = dict(
                law=("empirical P(n) times exact distinct-item ESP composition; "
                     "1<=n<=120"))
        result["baselines"][name] = block
        arrays[name] = values
        gap = block["paired_full_minus_baseline"]
        print(f"[other] {name:9s} {values.mean():9.4f}; full-gap "
              f"{gap['main_minus_baseline']:+.4f} +/- {gap['paired_se']:.4f}", flush=True)

    stem = args.output if os.path.isabs(args.output) else os.path.join(root, args.output)
    os.makedirs(os.path.dirname(stem), exist_ok=True)
    np.savez_compressed(stem + "_per_trip.npz", **arrays)
    with open(stem + ".json", "w") as stream:
        json.dump(result, stream, indent=2)
    print(f"[other] wrote {stem}.json", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--full-per-trip", default="reports/likelihood_test_per_trip.npz")
    p.add_argument("--full-key", default="target_child")
    p.add_argument("--split", choices=("validation", "test"), default="validation")
    p.add_argument("--iteration", type=int, default=400)
    p.add_argument("--checkpoint-kind", choices=("last", "best"), default="last")
    p.add_argument("--require-converged", action="store_true")
    p.add_argument("--baseline-tag", default="")
    p.add_argument("--models", type=parse_models, default=ALL_MODELS,
                   help="comma-separated subset of verified baselines to score")
    p.add_argument("--shopper-orders", type=int, default=8192)
    p.add_argument("--seed", type=int, default=20260821)
    p.add_argument("--chunk", type=int, default=8)
    p.add_argument("--bernoulli-chunk", type=int, default=4)
    p.add_argument("--maximum-trips", type=int, default=0,
                   help="score only a manifest prefix (integration tests only)")
    p.add_argument("--output", default="reports/baselines")
    main(p.parse_args())
