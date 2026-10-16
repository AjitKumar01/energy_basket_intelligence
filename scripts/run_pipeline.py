#!/usr/bin/env python3
"""Run the Version-4 pipeline from initialization to certification in one pass.

Stages, in order (each is one or more scripts in ``scripts/version4``):

    data           check the model-data bundle and the environment
    initialize     write the untrained initialization file
    additive       price-response evidence, then exact maximum likelihood with Phi = 0
    rank           choose the interaction directions and rank
    interaction    fit the interaction strengths, then the household size direction
    refinement     optional (--joint-refinement): refit blocks jointly, keep if accepted
    evaluation     likelihood, recommendation, generation, segments, embeddings
    certification  population basket-size audit, size diagnostic, promotion MDP

Every run starts from scratch in a new ``--run-dir``.  ``--stop-after`` ends the run
early after the named stage.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from version4.io_utils import model_data_root as resolve_model_data_root

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
V4 = ROOT / "scripts" / "version4"
ART = ROOT / "artifacts"
REPORT = ROOT / "reports"
OUT = ROOT / "out"
STAGES = ("data", "initialize", "additive", "rank", "interaction",
          "refinement", "evaluation", "certification")
RUN_MANIFEST = None
RUN_STATUS = None


def stratified_band_draws(nmax: int, *, full: bool) -> list[int]:
    """Match the interaction sampling vector to the data-dependent size support."""
    from version4.stratified_natural import SIZE_BAND_LOWER_BOUNDS
    if int(nmax) < 1:
        raise ValueError("nmax must be positive")
    band_count = sum(lower <= int(nmax) for lower in SIZE_BAND_LOWER_BOUNDS)
    production = [16, 16, 12, 8, 16, 16, 16]
    if len(production) != len(SIZE_BAND_LOWER_BOUNDS):
        raise RuntimeError("production band draws do not match the declared size strata")
    return production[:band_count] if full else [1] * band_count


def model_data_root() -> Path:
    """Resolve at call time so tests and embedded drivers can relocate ROOT safely."""
    return resolve_model_data_root(ROOT)


def report_availability_contract() -> dict:
    """Log the bundle's store availability contract (declared catalogue when absent)."""
    meta_path = model_data_root() / "basket_input" / "meta.json"
    contract = {"enabled": False}
    if meta_path.is_file():
        contract = json.loads(meta_path.read_text()).get("availability_contract") or contract
    if contract.get("enabled"):
        print(f"[pipeline] availability: rule={contract.get('rule')}, "
              f"floor={contract.get('floor'):.4g}, unconfirmed purchase lines "
              f"{contract.get('unconfirmed_lines_by_split')}", flush=True)
    else:
        print("[pipeline] availability: declared catalogue support at every store", flush=True)
    return contract


def price_configuration_from_evidence(status: int, report_path: Path,
                                      coefficients_path: Path) -> tuple:
    """Map the price-evidence stage's exit status and verdict to additive-fit flags."""
    try:
        report = json.loads(report_path.read_text())
    except Exception as exc:
        raise SystemExit(
            f"price-evidence stage exited {status} without a readable verdict "
            f"{report_path}: {exc}") from exc
    passed = report.get("passed") is True
    if status not in (0, 2) or (status == 0) != passed:
        raise SystemExit(
            f"price-evidence stage exit status {status} is inconsistent with its verdict "
            f"passed={report.get('passed')}")
    if passed:
        return ("--supported-price-coefficients", coefficients_path)
    print("[pipeline] price evidence failed its held-out gate; fitting the basket law "
          "with price response fixed to zero", flush=True)
    return ("--disable-price-response",)


def read_json(path: Path, description: str) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        raise SystemExit(f"cannot read {description} {path}: {exc}") from exc


def supplied_price_configuration(path: Path, *, dry_run: bool) -> tuple:
    """Flags for a user-supplied certified price-coefficient file."""
    path = (path if path.is_absolute() else ROOT / path).resolve()
    if not dry_run:
        payload = read_json(path, "supplied price coefficients")
        if payload.get("status") != "certified_observational_predictor":
            raise SystemExit("--price-coefficients is not a certified artifact")
        if payload.get("price_feature_contract") != "chain_product_week_only":
            raise SystemExit(
                "direct current-price coefficients must declare the chain-only feature contract")
    print(f"[pipeline] using supplied direct joint-basket price coefficients: {path}",
          flush=True)
    return ("--supported-price-coefficients", path)


def selected_rank(report_path: Path, *, maximum_rank: int) -> int:
    """The largest rank that passed the rank stage's split-half stability gate."""
    rank = read_json(report_path, "rank report").get("largest_stable_rank")
    if rank is None or not 1 <= int(rank) <= maximum_rank:
        raise SystemExit(f"no rank in 1..{maximum_rank} passed the predeclared split-half gate")
    return int(rank)


def fitted_interaction_rank(report_path: Path) -> int:
    """Number of positive eigenvalues of the fitted interaction matrix C."""
    eigenvalues = read_json(report_path, "interaction report")["candidate_c_eigenvalues"]
    tolerance = max(eigenvalues) * 1e-10 if eigenvalues else 0.0
    rank = sum(value > max(tolerance, 1e-12) for value in eigenvalues)
    if rank < 1:
        raise SystemExit("natural-parameter solve produced no positive interaction rank")
    return rank


def refinement_accepted(report_path: Path) -> bool:
    decision = read_json(report_path, "refinement report").get("decision", {})
    if not decision.get("accepted"):
        print(f"[pipeline] joint refinement not accepted ({decision.get('reason')}); "
              "evaluating the staged candidate", flush=True)
        return False
    return True


def preflight(*, stop_after: str) -> None:
    if sys.version_info < (3, 11):
        raise SystemExit("Python 3.11 or newer is required")
    modules = ("numpy", "pandas", "pyarrow", "scipy", "sklearn", "torch", "numba", "setuptools")
    missing_modules = [name for name in modules
                       if importlib.util.find_spec(name) is None]
    if missing_modules:
        raise SystemExit(
            "missing Python dependencies: " + ", ".join(missing_modules)
            + "; run python -m pip install -r requirements.txt")

    data_root = model_data_root()
    required = (
        data_root / "data" / "price_week.parquet",
        data_root / "data" / "build_meta.json",
        data_root / "basket_input" / "meta.json",
        data_root / "basket_input" / "items.parquet",
        data_root / "basket_input" / "baskets.parquet",
        data_root / "basket_input" / "promo.npz",
    )
    missing_derived = [path for path in required if not path.is_file()]
    if missing_derived:
        raise SystemExit(
            "the model-data bundle is incomplete; prepare it with "
            "scripts/prepare_model_bundle.py; missing: "
            + ", ".join(map(str, missing_derived)))

    if stop_after != "data" and not any(
            shutil.which(name) for name in ("c++", "clang++", "g++")):
        raise SystemExit(
            "no C++ compiler was found on PATH; install a compiler compatible with "
            "the active Python/PyTorch environment")


class Driver:
    """Runs stage commands, logging each one when a run directory is in use."""

    def __init__(self, dry_run: bool, log_dir: Path | None = None):
        self.dry_run = dry_run
        self.log_dir = log_dir
        self.commands: list[list[str]] = []
        self.environment = os.environ.copy()
        native = ART / "native" / "lib"
        old = self.environment.get("PYTHONPATH", "")
        self.environment["PYTHONPATH"] = os.pathsep.join(
            [str(native), str(V4)] + ([old] if old else []))
        self.environment["V3_AFFINITY"] = "1"

    def run(self, command: list[str], *, allow_failure: bool = False) -> int:
        self.commands.append(command)
        print(f"[pipeline] {shlex.join(command)}", flush=True)
        if self.dry_run:
            return 0
        if self.log_dir is None:
            code = subprocess.run(command, cwd=ROOT, env=self.environment, check=False).returncode
        else:
            code = self._run_logged(command)
        if code and not allow_failure:
            raise SystemExit(code)
        return code

    def _run_logged(self, command: list[str]) -> int:
        from run_manifest import write_manifest
        name = f"{len(self.commands):02d}_{Path(next(x for x in command if x.endswith('.py'))).stem}"
        path = self.log_dir / f"{name}.log"
        row = {"name": name, "command": command, "log": str(path), "status": "running"}
        RUN_STATUS["stages"].append(row)
        RUN_STATUS["current_stage"] = name
        write_manifest(RUN_MANIFEST, RUN_STATUS)
        started = time.monotonic()
        with path.open("x", buffering=1) as log:
            with subprocess.Popen(command, cwd=ROOT, env=self.environment, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
                for line in process.stdout:
                    log.write(line)
                    print(line, end="", flush=True)
                code = process.wait()
        row.update(status="completed" if code == 0 else "failed", exit_code=code,
                   runtime_seconds=time.monotonic() - started)
        write_manifest(RUN_MANIFEST, RUN_STATUS)
        return code


def script(name: str, *arguments: object) -> list[str]:
    return [PY, "-u", str(V4 / name), *map(str, arguments)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path,
                        help="new directory for this run's artifacts/, reports/ and out/")
    parser.add_argument(
        "--model-data-root", type=Path,
        help=("model-data bundle prepared by scripts/prepare_model_bundle.py; sets "
              "ENERGY_MODEL_DATA_ROOT for every stage (required unless that variable is set)"))
    parser.add_argument("--dry-run", action="store_true",
                        help="print the complete command graph without executing it")
    parser.add_argument("--profile", choices=("full", "smoke"), default="full")
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto",
        help=("execution policy; auto preserves the exact CPU backend even when an "
              "accelerator is present"))
    parser.add_argument(
        "--nmax", type=int, default=0,
        help="declared maximum basket size for initialization (0 = largest training basket)")
    parser.add_argument(
        "--threads", type=int, default=0,
        help="CPU intra-op threads; zero selects a hardware-aware value capped at 8")
    parser.add_argument(
        "--price-coefficients", type=Path,
        help=("certified direct joint-basket price coefficient artifact; when supplied, "
              "skip the price-change evidence stage and use this input"))
    parser.add_argument("--stop-after", choices=STAGES, default="certification")
    parser.add_argument(
        "--joint-refinement", action="store_true",
        help=("after the staged fit, refine taste, utility, penalty and interaction-strength "
              "parameters jointly (fit_joint_refinement.py); the refined model replaces "
              "the staged one only if its held-out gate accepts it"))
    parser.add_argument(
        "--evaluation-level-offset", type=int, default=2,
        help=("Smolyak level = rank + offset for the likelihood, recommendation and "
              "population-size gates (use 3 where rank + 2 fails the audit bound)"))
    args = parser.parse_args()
    if args.model_data_root is None and not os.environ.get("ENERGY_MODEL_DATA_ROOT"):
        parser.error("--model-data-root is required (a bundle from scripts/prepare_model_bundle.py)")
    if args.model_data_root is not None:
        bundle = args.model_data_root.expanduser().resolve()
        if not (bundle / "basket_input" / "meta.json").is_file():
            parser.error(f"--model-data-root {bundle} is not a prepared model-data bundle")
        configured = os.environ.get("ENERGY_MODEL_DATA_ROOT")
        if configured and Path(configured).expanduser().resolve() != bundle:
            parser.error("--model-data-root conflicts with ENERGY_MODEL_DATA_ROOT")
        os.environ["ENERGY_MODEL_DATA_ROOT"] = str(bundle)
        os.environ.setdefault("V3_AFFINITY", "1")
    if args.threads < 0:
        parser.error("--threads cannot be negative")
    return args


def start_run_log(args: argparse.Namespace) -> Path | None:
    """Point ART/REPORT/OUT at the run directory and start its log and manifest."""
    global ART, REPORT, OUT, RUN_MANIFEST, RUN_STATUS
    if args.run_dir is None:
        return None
    run_dir = args.run_dir.resolve()
    ART, REPORT, OUT = run_dir / "artifacts", run_dir / "reports", run_dir / "out"
    if args.dry_run:
        return None
    from run_manifest import Tee, write_manifest
    run_dir.mkdir(parents=True, exist_ok=False)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_dir = run_dir / f"invocation_{stamp}"
    log_dir.mkdir()
    sys.stdout = Tee(sys.stdout, log_dir / "pipeline.log")
    sys.stderr = sys.stdout
    RUN_MANIFEST = log_dir / "manifest.json"
    RUN_STATUS = {"status": "running", "pid": os.getpid(), "started_utc": stamp,
                  "run_dir": str(run_dir), "profile": args.profile,
                  "model_data_root": str(model_data_root()), "stages": [],
                  "configuration": {k: str(v) if isinstance(v, Path) else v
                                    for k, v in vars(args).items()}}
    write_manifest(RUN_MANIFEST, RUN_STATUS)
    return log_dir


def main() -> None:
    args = parse_args()
    log_dir = start_run_log(args)
    preflight(stop_after=args.stop_after)
    from runtime_capabilities import (detect_runtime, resolve_backend,
                                      write_runtime_report)
    if not args.dry_run:
        ART.mkdir(parents=True, exist_ok=True)
        REPORT.mkdir(parents=True, exist_ok=True)
        OUT.mkdir(parents=True, exist_ok=True)
    capabilities = detect_runtime(ROOT)
    try:
        selected_device = resolve_backend(args.device, capabilities)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    cpu_threads = args.threads or capabilities.recommended_cpu_threads
    if cpu_threads > capabilities.logical_cpu_count:
        raise SystemExit(
            f"--threads={cpu_threads} exceeds the detected logical CPU count "
            f"({capabilities.logical_cpu_count})")
    if args.profile == "full":
        if capabilities.memory_gib is not None and capabilities.memory_gib < 12:
            raise SystemExit(
                f"full profile requires about 12 GiB RAM; detected "
                f"{capabilities.memory_gib:g} GiB (use --profile smoke only for a "
                "software-path check)")
        if capabilities.workspace_free_gib < 5:
            raise SystemExit(
                f"full profile requires at least 5 GiB free workspace disk; detected "
                f"{capabilities.workspace_free_gib:g} GiB")
    if not args.dry_run:
        write_runtime_report(
            ART / "runtime_capabilities.json", capabilities,
            requested_device=args.device, selected_device=selected_device,
            cpu_threads=cpu_threads)
    accelerator = (f", CUDA devices={capabilities.cuda_device_count} (not eligible for "
                   "the exact normalizer)" if capabilities.cuda_device_count else "")
    print(f"[pipeline] backend={selected_device}, CPU threads={cpu_threads}, "
          f"RAM={capabilities.memory_gib or 'unknown'} GiB{accelerator}", flush=True)
    print(f"[pipeline] stages: data -> {args.stop_after}", flush=True)
    print(f"[pipeline] model data root: {model_data_root()}", flush=True)
    print(f"[pipeline] hardware report: {ART / 'runtime_capabilities.json'}", flush=True)
    driver = Driver(args.dry_run, log_dir)
    dry = driver.dry_run
    full = args.profile == "full"
    threads = cpu_threads

    # ---- data -------------------------------------------------------------------------
    report_availability_contract()
    if args.stop_after == "data":
        return

    # ---- initialize -------------------------------------------------------------------
    driver.run([PY, str(V4 / "setup_poly_degree_native.py"), "build_ext",
                "--build-lib", str(ART / "native" / "lib"),
                "--build-temp", str(ART / "native" / "temp"), "--force"])
    initialization = ART / "initialization.pt"
    driver.run(script("initialize_version4.py", "--output", initialization,
                      "--manifest", ART / "initialization.json",
                      "--household-size-rank1", "--threads", threads,
                      *(("--nmax", args.nmax) if args.nmax else ())))
    nmax = 120 if dry else int(read_json(ART / "initialization.json",
                                         "initialization summary")["metadata"]["nmax"])
    if args.stop_after == "initialize":
        return

    # ---- additive ---------------------------------------------------------------------
    additive = OUT / "v3_pipeline_additive_best.pt"
    if args.price_coefficients is not None:
        price_configuration = supplied_price_configuration(args.price_coefficients, dry_run=dry)
    else:
        # The evidence stage exits 2 when no price component is certified.  That is a
        # verdict, not a crash: the basket law is then fitted with price fixed to zero.
        price_report = ART / "supported_price_response" / "report.json"
        price_coefficients = ART / "supported_price_response" / "coefficients.json"
        evidence_status = driver.run(script(
            "fit_supported_price_response.py",
            "--output", price_report,
            "--coefficients-output", price_coefficients,
            "--seed", 68101), allow_failure=True)
        price_configuration = (
            ("--supported-price-coefficients", price_coefficients) if dry else
            price_configuration_from_evidence(evidence_status, price_report, price_coefficients))
    driver.run(script(
        "fit_exact_additive.py", "--artifact", initialization,
        "--output-dir", OUT,
        "--label", "pipeline_additive", "--iters", 30000 if full else 10,
        "--batch", 128 if full else 8, "--lr", 0.002,
        "--weight-decay", 1e-5, "--validation-trips", 1024 if full else 16,
        "--validation-chunk", 128 if full else 8,
        "--eval-every", 100 if full else 5, "--size-kl", 1.0,
        "--lr-patience", 4, "--lr-factor", 0.5, "--min-lr", 6.25e-5,
        "--convergence-patience", 8,
        "--convergence-min-updates", 4000 if full else 10,
        "--validation-min-delta", 0.001,
        "--rkl-w", 10.0, "--elast-w", 0.0,
        *price_configuration,
        "--pool-prod", 1.45, "--lam-centre", 1, "--seed", 29001,
        "--threads", threads,
        "--rho-c-max-category-reward", 1.5,
        *(('--require-convergence',) if full else ())))
    if args.stop_after == "additive":
        return

    # ---- rank -------------------------------------------------------------------------
    maximum_rank = 8 if full else 4
    basis = ART / f"interaction_basis_rank{maximum_rank}.npz"
    status = driver.run(script(
        "build_spectral_phi_initialization.py", "--parent", additive,
        "--trips", 50000 if full else 128, "--draws", 8 if full else 2,
        "--rank", maximum_rank, "--threads", threads,
        "--minimum-stability", 0.5 if full else -1.0,
        "--output", basis), allow_failure=True)
    if dry:
        rank = maximum_rank
    else:
        if status != 0:
            raise SystemExit("spectral rank build failed")
        rank = selected_rank(basis.with_suffix(".json"), maximum_rank=maximum_rank)
        print(f"[pipeline] selected independently stable rank {rank}")
    if args.stop_after == "rank":
        return

    # ---- interaction ------------------------------------------------------------------
    interaction_candidate = ART / "candidate.pt"
    candidate = ART / "candidate_rank1.pt"
    driver.run(script(
        "fit_stratified_natural_interactions.py", "--parent", additive,
        "--spectral", basis, "--contexts", 12000 if full else 64,
        "--band-draws", *stratified_band_draws(nmax, full=full),
        "--batch", 96 if full else 8, "--rank", rank,
        "--score-mass", 1.0, "--spectral-max", 1.0,
        "--category-bound", 0.0,
        "--size-ridge", 1e-3, "--size-smoothness", 1e-1,
        "--threads", threads,
        "--minimum-crossfit-gain", 0.005 if full else -1.0,
        "--minimum-half-gain", 0.0 if full else -1e9,
        "--minimum-within-band-ess-fraction", 0.20 if full else 0.0,
        "--minimum-within-band-ess", 2.0 if full else 0.0,
        "--output", interaction_candidate))
    if not dry:
        fitted_rank = fitted_interaction_rank(interaction_candidate.with_suffix(".json"))
        if fitted_rank != rank:
            print(f"[pipeline] natural-parameter solve reduced certified basis rank "
                  f"{rank} to active rank {fitted_rank}")
        rank = fitted_rank
    print("[pipeline] post-interaction size block: retain the additive kappa_h and "
          "test only a cross-fitted residual household increment", flush=True)
    driver.run(script(
        "fit_household_size_rank1.py",
        "--checkpoint", interaction_candidate,
        "--rank", rank, "--screen-level", rank + 1,
        "--contexts", 0 if full else 128,
        "--screen-tail-cap", 0.35,
        "--minimum-crossfit-gain", 0.0 if full else -1e9,
        "--on-crossfit-failure", "fallback",
        "--chunk", 48 if full else 8,
        "--threads", threads,
        "--output", candidate,
        "--report", ART / "candidate_rank1.json"))
    if args.stop_after == "interaction":
        return

    # ---- refinement (optional) --------------------------------------------------------
    level = rank + args.evaluation_level_offset
    if args.joint_refinement:
        refined = ART / "candidate_refined.pt"
        refinement_report = REPORT / "joint_refinement.json"
        driver.run(script(
            "fit_joint_refinement.py", "--checkpoint", candidate,
            "--output", refined, "--report", refinement_report,
            "--contexts-per-household", 10,
            *(() if full else ("--max-contexts", 256)),
            "--chains", 4, "--draws-per-chain", 4, "--burn", 3, "--baskets-per-z", 2,
            "--rounds", 5 if full else 1,
            "--select-trust", 300, 1000,
            "--validation-trips", 1024 if full else 64,
            "--selection-trips", 1024 if full else 64,
            "--level-offset", args.evaluation_level_offset,
            # keep the household-size stage's contract (same cap and screen) in every round
            "--household-size-cap", 0.35,
            "--household-size-chunk", 48 if full else 8,
            # the certification stage's population-size audit is also an acceptance gate here
            "--size-gate",
            "--size-gate-contexts", 0 if full else 128,
            "--size-gate-confirm-contexts", 2048 if full else 8,
            "--size-gate-calibration-contexts", 2048 if full else 16,
            "--size-gate-chunk", 48 if full else 8,
            "--threads", threads))
        if dry or refinement_accepted(refinement_report):
            candidate = refined
        print(f"[pipeline] evaluation candidate after refinement: {candidate}", flush=True)
    if args.stop_after == "refinement":
        return

    # ---- evaluation -------------------------------------------------------------------
    # All claims use fixed panels and complete support; recommendation is read-only.
    # The likelihood comparisons exit nonzero (stopping the run) when a full-profile
    # numerical or likelihood gate fails.
    for split, extra in (("validation", ('--maximum-audit-error-bound', 0.01,
                                         '--require-certified-gain')),
                         ("test", ('--maximum-audit-error-bound', 0.01))):
        driver.run(script(
            "compare_rank8_parent_likelihood.py", "--parent", additive,
            "--child", candidate, "--split", split,
            "--trips", 4096 if full else 16,
            "--rank", rank, "--target-level", level,
            "--audit-trips", 128 if full else 4,
            "--threads", threads,
            *(extra if full else ()),
            "--output", REPORT / f"likelihood_{split}.json"))
    driver.run(script(
        "eval_smolyak_rank8_mrr.py", "--ckpt", candidate, "--split", "test",
        "--parent", additive,
        "--trips", 2000 if full else 16, "--rank", rank,
        "--level", level, "--threads", threads,
        "--output", REPORT / "recommendation.json"))
    driver.run(script(
        "audit_particle_counterfactual_generation.py", "--ckpt", candidate,
        "--trips", 64 if full else 2, "--particles", 64 if full else 4,
        "--threads", threads,
        "--output", REPORT / "generation_counterfactual.json"))
    driver.run(script(
        "audit_customer_segments.py", "--ckpt", candidate,
        "--candidate-segments", 3, 4, 5, 6,
        "--contexts-per-segment", 48 if full else 2,
        "--particles", 32 if full else 4,
        "--threads", threads,
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

    # ---- certification ----------------------------------------------------------------
    tail_status = driver.run(script(
        "audit_population_size.py", "--checkpoint", candidate,
        "--rank", rank, "--screen-level", level - 1,
        "--confirm-level", level,
        "--contexts", 0 if full else 128,
        "--confirm-contexts", 2048 if full else 8,
        "--calibration-contexts", 2048 if full else 16,
        "--chunk", 48 if full else 8,
        "--threads", threads,
        "--output", REPORT / "population_size.json"), allow_failure=not full)
    driver.run(script(
        "diagnose_size_phase.py", "--parent", additive, "--child", candidate,
        "--confirmed-panel", REPORT / "population_size_confirm_per_trip.npz",
        "--contexts-per-panel", 64 if full else 2,
        "--threads", threads,
        "--output", REPORT / "size_phase_diagnostic.json"))
    run_pricing_policy(driver, candidate, full=full, threads=threads)
    if dry:
        print("[pipeline] dry run complete; no stage was executed")
    elif full:
        print("[pipeline] fitted-model likelihood/numerical and population-safety gates "
              f"passed; {candidate.name} is accepted for conditional-incidence "
              "likelihood use. Generation calibration, causal price response and policy "
              "value require separate capability verdicts", flush=True)
    else:
        print("[pipeline] smoke integration completed; statistical gates were relaxed "
              f"and tail audit status was {tail_status}; this is not a certified fit")


def run_pricing_policy(driver: Driver, candidate: Path, *, full: bool, threads: int) -> None:
    """Promotion-budget MDP, skipped when the fitted price response is fixed to zero."""
    destination = REPORT / "segment_promotion_mdp.json"
    command = script(
        "run_segment_pricing_mdp.py", "--checkpoint", candidate,
        "--assignments", ART / "customer_segments.npz",
        "--segment-report", REPORT / "customer_segments.json",
        "--contexts-per-segment", 64 if full else 2,
        "--particles", 32 if full else 4,
        "--levels", 17 if full else 5,
        "--context-chunk", 8 if full else 2,
        "--threads", threads,
        "--bundles-per-segment", 3 if full else 1,
        "--products-per-bundle", 5 if full else 3,
        "--horizon-days", 28 if full else 7,
        "--budget-bins", 4000 if full else 200,
        "--minimum-budget-utilization", 0.0,
        "--output", destination)
    if driver.dry_run:
        print("[pipeline] pricing policy is conditional on a nonzero, held-out-supported "
              "price component", flush=True)
        driver.run(command)
        return
    import torch
    price_estimator = torch.load(candidate, map_location="cpu", weights_only=False).get(
        "price_response_estimator")
    if price_estimator != "fixed_zero_after_failed_heldout_support":
        driver.run(command)
        return
    skipped = {
        "status": "not_assessed",
        "reason": ("held-out price evidence failed and the fitted price "
                   "response is fixed to zero; policy computation would be "
                   "uninformative"),
        "checkpoint": str(candidate),
        "price_response_estimator": price_estimator,
    }
    destination.write_text(json.dumps(skipped, indent=2, sort_keys=True) + "\n")
    print(f"[pipeline] skipped pricing policy; verdict written to {destination}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        if RUN_STATUS is not None:
            from run_manifest import write_manifest
            RUN_STATUS.update(status="failed", error=repr(exc))
            write_manifest(RUN_MANIFEST, RUN_STATUS)
        raise
    else:
        if RUN_STATUS is not None:
            from run_manifest import write_manifest
            RUN_STATUS.update(status="completed", current_stage=None)
            write_manifest(RUN_MANIFEST, RUN_STATUS)
