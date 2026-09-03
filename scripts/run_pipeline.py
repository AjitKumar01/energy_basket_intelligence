#!/usr/bin/env python3
"""One reproducible driver for the selected Version-4 staged pipeline."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
V4 = ROOT / "scripts" / "version4"
ART = ROOT / "artifacts"
REPORT = ROOT / "reports"
RAW_DEFAULT = (ROOT.parent / "dunnhumby_The-Complete-Journey" /
               "dunnhumby_The-Complete-Journey CSV")


def preflight(*, from_raw: bool, stop_after: str) -> None:
    if sys.version_info < (3, 11):
        raise SystemExit("Python 3.11 or newer is required")
    modules = ("numpy", "pandas", "pyarrow", "scipy", "sklearn", "torch", "setuptools")
    missing_modules = [name for name in modules
                       if importlib.util.find_spec(name) is None]
    if missing_modules:
        raise SystemExit(
            "missing Python dependencies: " + ", ".join(missing_modules)
            + "; run python -m pip install -r requirements.txt")

    raw = Path(os.environ.get("NF_RAW_DIR", RAW_DEFAULT)).expanduser()
    missing_raw = [raw / name for name in (
        "transaction_data.csv", "product.csv", "causal_data.csv")
        if not (raw / name).is_file()]
    if missing_raw:
        raise SystemExit(
            "raw dunnhumby input is incomplete; set NF_RAW_DIR to the directory "
            "containing transaction_data.csv, product.csv and causal_data.csv; missing: "
            + ", ".join(map(str, missing_raw)))

    if not from_raw:
        required = (
            ROOT / "data" / "tx.parquet",
            ROOT / "data" / "price_week.parquet",
            ROOT / "data" / "price_store_week.parquet",
            ROOT / "basket_input" / "meta.json",
            ROOT / "basket_input" / "items.parquet",
            ROOT / "basket_input" / "baskets.parquet",
            ROOT / "basket_input" / "promo.npz",
        )
        missing_derived = [path for path in required if not path.is_file()]
        if missing_derived:
            raise SystemExit(
                "derived data are incomplete; rerun with --from-raw; missing: "
                + ", ".join(map(str, missing_derived)))

    if stop_after != "data" and not any(
            shutil.which(name) for name in ("c++", "clang++", "g++")):
        raise SystemExit(
            "no C++ compiler was found on PATH; install a compiler compatible with "
            "the active Python/PyTorch environment")


def command_text(command: list[str]) -> str:
    return shlex.join(command)


class Driver:
    def __init__(self, dry_run: bool):
        self.dry_run = dry_run
        self.commands: list[list[str]] = []
        self.environment = os.environ.copy()
        native = ART / "native" / "lib"
        old = self.environment.get("PYTHONPATH", "")
        self.environment["PYTHONPATH"] = os.pathsep.join(
            [str(native), str(V4)] + ([old] if old else []))
        self.environment["V3_AFFINITY"] = "1"

    def run(self, command: list[str], *, allow_failure: bool = False) -> int:
        self.commands.append(command)
        print(f"[pipeline] {command_text(command)}", flush=True)
        if self.dry_run:
            return 0
        result = subprocess.run(command, cwd=ROOT, env=self.environment,
                                check=False)
        if result.returncode and not allow_failure:
            raise SystemExit(result.returncode)
        return result.returncode


def script(name: str, *arguments: object) -> list[str]:
    return [PY, "-u", str(V4 / name), *map(str, arguments)]


def rank_selection(driver: Driver, parent: Path, contexts: int,
                   *, smoke: bool = False, threads: int = 8) -> tuple[int, Path]:
    if driver.dry_run:
        if smoke:
            output = ART / "interaction_basis_rank4.npz"
            driver.run(script("build_spectral_phi_initialization.py", "--parent", parent,
                              "--trips", contexts, "--draws", 2, "--rank", 4,
                              "--threads", threads,
                              "--minimum-stability", -1.0, "--output", output))
            return 4, output
        output = ART / "interaction_basis_rank8.npz"
        driver.run(script("build_spectral_phi_initialization.py", "--parent", parent,
                          "--trips", contexts, "--draws", 2, "--rank", 8,
                          "--threads", threads,
                          "--output", output))
        return 8, output
    maximum_rank = 4 if smoke else 8
    output = ART / f"interaction_basis_rank{maximum_rank}.npz"
    driver.run(script("build_spectral_phi_initialization.py", "--parent", parent,
                      "--trips", contexts, "--draws", 2, "--rank", maximum_rank,
                      "--threads", threads,
                      "--minimum-stability", -1.0 if smoke else 0.5,
                      "--output", output), allow_failure=True)
    report_path = output.with_suffix(".json")
    if report_path.exists():
        report = json.loads(report_path.read_text())
        profiles = report.get("rank_stability", {})
        for rank in range(maximum_rank, 3, -1):
            if profiles.get(str(rank), {}).get("accepted"):
                print(f"[pipeline] selected independently stable rank {rank}")
                return rank, output
    raise SystemExit("no rank in 4..8 passed the predeclared split-half gate")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-raw", action="store_true",
                        help="rebuild data/ and basket_input/ from dunnhumby CSVs")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the complete command graph without executing it")
    parser.add_argument("--profile", choices=("full", "smoke"), default="full")
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto",
        help=("execution policy; auto preserves the exact CPU backend even when an "
              "accelerator is present"))
    parser.add_argument(
        "--threads", type=int, default=0,
        help="CPU intra-op threads; zero selects a hardware-aware value capped at 8")
    parser.add_argument("--resume-additive", type=Path,
                        help=("continue an exact-additive checkpoint with optimizer and "
                              "minibatch stream intact; downstream stages still rerun"))
    parser.add_argument("--stop-after", choices=(
        "data", "initialize", "additive", "rank", "interaction",
        "evaluation", "certification"), default="certification")
    args = parser.parse_args()
    if args.from_raw and args.resume_additive is not None:
        parser.error("--from-raw cannot be combined with --resume-additive")
    if args.threads < 0:
        parser.error("--threads cannot be negative")
    preflight(from_raw=args.from_raw, stop_after=args.stop_after)
    from runtime_capabilities import (detect_runtime, resolve_backend,
                                      write_runtime_report)
    ART.mkdir(exist_ok=True)
    REPORT.mkdir(exist_ok=True)
    (ROOT / "out").mkdir(exist_ok=True)
    capabilities = detect_runtime(ROOT)
    try:
        selected_device = resolve_backend(args.device, capabilities)
    except RuntimeError as exc:
        parser.error(str(exc))
    cpu_threads = args.threads or capabilities.recommended_cpu_threads
    if cpu_threads > capabilities.logical_cpu_count:
        parser.error(
            f"--threads={cpu_threads} exceeds the detected logical CPU count "
            f"({capabilities.logical_cpu_count})")
    if args.profile == "full":
        if capabilities.memory_gib is not None and capabilities.memory_gib < 12:
            parser.error(
                f"full profile requires about 12 GiB RAM; detected "
                f"{capabilities.memory_gib:g} GiB (use --profile smoke only for a "
                "software-path check)")
        if capabilities.workspace_free_gib < 5:
            parser.error(
                f"full profile requires at least 5 GiB free workspace disk; detected "
                f"{capabilities.workspace_free_gib:g} GiB")
    write_runtime_report(
        ART / "runtime_capabilities.json", capabilities,
        requested_device=args.device, selected_device=selected_device,
        cpu_threads=cpu_threads)
    accelerator = (f", CUDA devices={capabilities.cuda_device_count} (not eligible for "
                   "the exact normalizer)" if capabilities.cuda_device_count else "")
    print(f"[pipeline] backend={selected_device}, CPU threads={cpu_threads}, "
          f"RAM={capabilities.memory_gib or 'unknown'} GiB{accelerator}", flush=True)
    print(f"[pipeline] hardware report: {ART / 'runtime_capabilities.json'}", flush=True)
    driver = Driver(args.dry_run)

    if args.from_raw:
        driver.run([PY, "-u", "scripts/data/01_build_base.py"])
        driver.run([PY, "-u", "scripts/data/22_basket_data.py"])
        driver.run([PY, "-u", "scripts/data/23_promo_data.py"])
    # Always fail closed on data integrity, including when reusing derived files.
    driver.run([PY, "-u", "scripts/data/audit_preprocessing.py"])
    driver.run(script("build_affinity_partition.py"))
    # Build the ragged index only after the training-only partition exists.  The
    # partition builder reads baskets directly and cannot consume a stale model cache.
    driver.run(script("data.py", "--force"))
    if args.stop_after == "data":
        return

    driver.run([PY, str(V4 / "setup_poly_degree_native.py"), "build_ext",
                "--build-lib", str(ART / "native" / "lib"),
                "--build-temp", str(ART / "native" / "temp"), "--force"])
    initialization = ART / "initialization.pt"
    if args.resume_additive is None:
        driver.run(script("initialize_version4.py", "--output", initialization,
                          "--manifest", ART / "initialization.json",
                          "--household-size-rank1", "--threads", cpu_threads))
    elif not initialization.exists() and not driver.dry_run:
        raise SystemExit("resume requested but artifacts/initialization.pt is missing")
    if args.stop_after == "initialize":
        return

    full = args.profile == "full"
    additive_iterations = 30000 if full else 10
    additive_command = script(
        "fit_exact_additive.py", "--artifact", initialization,
        "--label", "pipeline_additive", "--iters", additive_iterations,
        "--batch", 128 if full else 8, "--lr", 0.002,
        "--weight-decay", 1e-5, "--validation-trips", 1024 if full else 16,
        "--validation-chunk", 128 if full else 8,
        "--eval-every", 100 if full else 5, "--size-kl", 1.0,
        "--lr-patience", 4, "--lr-factor", 0.5, "--min-lr", 6.25e-5,
        "--convergence-patience", 8,
        "--convergence-min-updates", 4000 if full else 10,
        "--validation-min-delta", 0.001,
        "--rkl-w", 10.0, "--elast-w", 20.0, "--elast-target", -0.121,
        "--pool-prod", 1.45, "--lam-centre", 1, "--seed", 29001,
        "--threads", cpu_threads,
        "--rho-c-max-category-reward", 1.5,
        *(('--require-convergence',) if full else ()),
        *(('--resume', args.resume_additive)
          if args.resume_additive is not None else ()))
    driver.run(additive_command)
    additive = ROOT / "out" / "v3_pipeline_additive_best.pt"
    if args.stop_after == "additive":
        return

    rank, basis = rank_selection(driver, additive, 50000 if full else 128,
                                  smoke=not full, threads=cpu_threads)
    if args.stop_after == "rank":
        return

    interaction_candidate = ART / "candidate.pt"
    driver.run(script(
        "fit_convex_natural_interactions.py", "--parent", additive,
        "--spectral", basis, "--contexts", 12000 if full else 64,
        "--draws", 64 if full else 4, "--batch", 96 if full else 8,
        "--rank", rank, "--score-mass", 1.0, "--spectral-max", 1.0,
        "--threads", cpu_threads,
        "--minimum-crossfit-gain", 0.005 if full else -1.0,
        "--minimum-half-gain", 0.0 if full else -1e9,
        "--minimum-ess-fraction", 0.20 if full else 0.0,
        "--minimum-ess-p01", 2.0 if full else 0.0,
        "--output", interaction_candidate))
    if not driver.dry_run:
        interaction_report = json.loads(
            interaction_candidate.with_suffix(".json").read_text())
        eigenvalues = interaction_report["candidate_c_eigenvalues"]
        tolerance = max(eigenvalues) * 1e-10 if eigenvalues else 0.0
        fitted_rank = sum(value > max(tolerance, 1e-12) for value in eigenvalues)
        if fitted_rank < 1:
            raise SystemExit("natural-parameter solve produced no positive interaction rank")
        if fitted_rank != rank:
            print(f"[pipeline] convex solve reduced certified basis rank {rank} "
                  f"to active rank {fitted_rank}")
        rank = fitted_rank
    candidate = ART / "candidate_rank1.pt"
    driver.run(script(
        "fit_household_size_rank1.py",
        "--checkpoint", interaction_candidate,
        "--rank", rank, "--screen-level", rank + 1,
        "--contexts", 0 if full else 128,
        "--screen-tail-cap", 0.35,
        "--minimum-crossfit-gain", 0.0 if full else -1e9,
        "--chunk", 48 if full else 8,
        "--threads", cpu_threads,
        "--output", candidate,
        "--report", ART / "candidate_rank1.json",
        "--population-output", REPORT / "population_size.json"))
    if args.stop_after == "interaction":
        return

    # All claims use fixed panels and complete support; recommendation is read-only.
    driver.run(script(
        "compare_rank8_parent_likelihood.py", "--parent", additive,
        "--child", candidate, "--split", "validation", "--trips", 4096 if full else 16,
        "--rank", rank, "--target-level", rank + 2,
        "--audit-trips", 128 if full else 4,
        "--threads", cpu_threads,
        *(('--maximum-audit-error-bound', 0.01, '--require-certified-gain')
          if full else ()),
        "--output", REPORT / "likelihood_validation.json"))
    driver.run(script(
        "compare_rank8_parent_likelihood.py", "--parent", additive,
        "--child", candidate, "--split", "test", "--trips", 4096 if full else 16,
        "--rank", rank, "--target-level", rank + 2,
        "--audit-trips", 128 if full else 4,
        "--threads", cpu_threads,
        *(('--maximum-audit-error-bound', 0.01, '--require-certified-gain')
          if full else ()),
        "--output", REPORT / "likelihood_test.json"))
    driver.run(script(
        "eval_smolyak_rank8_mrr.py", "--ckpt", candidate, "--split", "test",
        "--trips", 2000 if full else 16, "--rank", rank,
        "--level", rank + 2, "--threads", cpu_threads,
        "--output", REPORT / "recommendation.json"))
    driver.run(script(
        "audit_particle_counterfactual_generation.py", "--ckpt", candidate,
        "--trips", 64 if full else 2, "--particles", 64 if full else 4,
        "--threads", cpu_threads,
        "--output", REPORT / "generation_counterfactual.json"))
    driver.run(script(
        "audit_customer_segments.py", "--ckpt", candidate,
        "--candidate-segments", 3, 4, 5, 6,
        "--contexts-per-segment", 48 if full else 2,
        "--particles", 32 if full else 4,
        "--threads", cpu_threads,
        "--output", REPORT / "customer_segments.json",
        "--assignments", ART / "customer_segments.npz"))
    driver.run(script(
        "audit_interaction_embeddings.py", "--checkpoint", candidate,
        "--spectral-report", basis.with_suffix(".json"),
        "--minimum-training-lines", 100 if full else 1,
        "--pairs", 2000 if full else 32,
        "--listed-pairs", 20 if full else 5,
        "--output", REPORT / "interaction_embedding_audit.json"))
    if args.stop_after == "evaluation":
        return

    tail_status = driver.run(script(
        "audit_population_size.py", "--checkpoint", candidate,
        "--rank", rank, "--screen-level", rank + 1,
        "--confirm-level", rank + 2,
        "--contexts", 0 if full else 128,
        "--confirm-contexts", 2048 if full else 8,
        "--calibration-contexts", 2048 if full else 16,
        "--chunk", 48 if full else 8,
        "--threads", cpu_threads,
        "--output", REPORT / "population_size.json"), allow_failure=not full)
    driver.run(script(
        "run_segment_pricing_mdp.py", "--checkpoint", candidate,
        "--assignments", ART / "customer_segments.npz",
        "--segment-report", REPORT / "customer_segments.json",
        "--contexts-per-segment", 64 if full else 2,
        "--particles", 32 if full else 4,
        "--levels", 17 if full else 5,
        "--context-chunk", 8 if full else 2,
        "--threads", cpu_threads,
        "--bundles-per-segment", 3 if full else 1,
        "--products-per-bundle", 5 if full else 3,
        "--horizon-days", 28 if full else 7,
        "--budget-bins", 4000 if full else 200,
        "--minimum-budget-utilization", 0.95 if full else 0.90,
        "--output", REPORT / "segment_promotion_mdp.json"))
    if args.dry_run:
        print("[pipeline] dry run complete; no stage was executed")
    elif full:
        print("[pipeline] certification passed; candidate_rank1.pt is the accepted model")
    else:
        print("[pipeline] smoke integration completed; statistical gates were relaxed "
              f"and tail audit status was {tail_status}; this is not a certified fit")


if __name__ == "__main__":
    main()
