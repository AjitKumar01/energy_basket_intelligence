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
import time
from datetime import datetime, timezone
from pathlib import Path

from version4.provenance import (file_sha256, load_data_fingerprint,
                                 model_data_root as resolve_model_data_root,
                                 require_fingerprint)

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
V4 = ROOT / "scripts" / "version4"
ART = ROOT / "artifacts"
REPORT = ROOT / "reports"
OUT = ROOT / "out"
RAW_DEFAULT = (ROOT.parent / "dunnhumby_The-Complete-Journey" /
               "dunnhumby_The-Complete-Journey CSV")
RAW_LOCAL = (ROOT / "dunnhumby_The-Complete-Journey" /
             "dunnhumby_The-Complete-Journey CSV")
STAGES = ("data", "initialize", "additive", "rank", "interaction",
          "evaluation", "certification")
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


def current_data_fingerprint(*, dry_run: bool) -> str | None:
    path = model_data_root() / "basket_input" / "model_data_fingerprint.json"
    require_files("data", (path,), dry_run=dry_run)
    if dry_run:
        return None
    try:
        return load_data_fingerprint(model_data_root())["fingerprint_sha256"]
    except Exception as exc:
        raise SystemExit(f"cannot read model-data fingerprint {path}: {exc}") from exc


def verify_data_bundle(*, dry_run: bool) -> None:
    """Self-verify the fingerprint and re-hash every model-facing file it names."""
    recorded = current_data_fingerprint(dry_run=dry_run)
    if not dry_run:
        require_fingerprint(recorded, model_data_root())


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


def stage_index(stage: str) -> int:
    return STAGES.index(stage)


def resolve_raw_directory() -> Path:
    configured = os.environ.get("NF_RAW_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    for candidate in (RAW_DEFAULT, RAW_LOCAL):
        if candidate.is_dir():
            return candidate.resolve()
    return RAW_DEFAULT.resolve()


def runs_stage(start_at: str, stage: str) -> bool:
    """Whether a stage belongs to the requested suffix of the pipeline graph."""
    return stage_index(stage) >= stage_index(start_at)


def require_files(stage: str, paths: tuple[Path, ...], *, dry_run: bool) -> None:
    """Fail before compute when a resurrected stage is missing its prerequisites."""
    if dry_run:
        return
    missing = [path for path in paths if not path.is_file()]
    if missing:
        command = f"python scripts/run_pipeline.py --start-at {stage}"
        raise SystemExit(
            f"cannot resurrect stage '{stage}'; missing prerequisite files: "
            + ", ".join(str(path) for path in missing)
            + f". Restore them from the original machine or begin at an earlier stage "
              f"than in: {command}")


def load_checkpoint_blob(path: Path, description: str) -> dict:
    import torch
    try:
        blob = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise SystemExit(f"cannot read {description} {path}: {exc}") from exc
    if not isinstance(blob, dict) or not isinstance(blob.get("model"), dict):
        raise SystemExit(f"{description} {path} is not a pipeline model checkpoint")
    return blob


def validate_initialization(path: Path, *, dry_run: bool) -> dict | None:
    require_files("initialize", (path,), dry_run=dry_run)
    if dry_run:
        return None
    import torch
    try:
        blob = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise SystemExit(f"cannot read initialization checkpoint {path}: {exc}") from exc
    required = ("metadata", "model_state", "model_state_sha256")
    if not isinstance(blob, dict) or any(key not in blob for key in required):
        raise SystemExit(f"{path} is not a Version-4 initialization artifact")
    if not blob["metadata"].get("household_size_rank1", False):
        raise SystemExit(f"{path} does not reserve the household-size rank-one coordinate")
    expected = current_data_fingerprint(dry_run=False)
    if blob["metadata"].get("data_fingerprint_sha256") != expected:
        raise SystemExit(
            f"{path} does not match the current audited model dataset")
    return blob


def validate_additive(path: Path, initialization: Path, *, dry_run: bool) -> dict | None:
    require_files("additive", (initialization, path), dry_run=dry_run)
    initial = validate_initialization(initialization, dry_run=dry_run)
    if dry_run:
        return None
    blob = load_checkpoint_blob(path, "additive checkpoint")
    if blob.get("estimator") != "exact_version4_no_gram_dynamic_program":
        raise SystemExit(f"{path} is not an exact additive checkpoint")
    if blob.get("fresh_artifact_digest") != initial["model_state_sha256"]:
        raise SystemExit(
            f"{path} was not trained from {initialization}; restore the matching "
              "artifacts/initialization.pt")
    if blob.get("data_fingerprint_sha256") != initial["metadata"].get(
            "data_fingerprint_sha256"):
        raise SystemExit(f"{path} and {initialization} use different model data")
    return blob


def validate_completed_additive(best: Path, latest: Path, initialization: Path,
                                *, profile: str, dry_run: bool) -> dict | None:
    """Reject a partial/failed optimizer run even when its best checkpoint exists."""
    require_files("additive", (best, latest), dry_run=dry_run)
    best_blob = validate_additive(best, initialization, dry_run=dry_run)
    latest_blob = validate_additive(latest, initialization, dry_run=dry_run)
    if dry_run:
        return None
    expected_batch = 128 if profile == "full" else 8
    for label, blob in (("best", best_blob), ("latest", latest_blob)):
        config = blob.get("config", {})
        if int(config.get("batch", -1)) != expected_batch:
            raise SystemExit(
                f"{label} additive checkpoint belongs to a different profile; "
                f"expected batch {expected_batch} for --profile {profile}")
    if int(best_blob.get("best_iteration", -1)) != int(latest_blob.get("best_iteration", -2)):
        raise SystemExit("best and latest additive checkpoints do not belong to the same fit")
    if profile == "full":
        config = latest_blob["config"]
        scheduler = latest_blob.get("scheduler", {})
        converged = (
            bool(config.get("require_convergence", False))
            and int(latest_blob.get("iter", 0)) >= int(config["convergence_min_updates"])
            and float(scheduler.get("learning_rate", float("inf")))
            <= float(config["min_lr"]) + 1e-15
            and int(scheduler.get("evaluations_since_best", -1))
            >= int(config["convergence_patience"])
            and scheduler.get(
                "convergence_patience_basis", "legacy_numerical_best")
            in {"material_validation_gain", "legacy_numerical_best"}
        )
        if not converged:
            raise SystemExit(
                f"{latest} is partial or reached its ceiling without convergence; "
                "continue it with --start-at additive --resume-additive "
                f"{latest}")
    return best_blob


def validate_candidate(path: Path, initialization: Path, *, final: bool,
                       profile: str, dry_run: bool) -> dict | None:
    require_files("interaction", (initialization, path), dry_run=dry_run)
    initial = validate_initialization(initialization, dry_run=dry_run)
    if dry_run:
        return None
    blob = load_checkpoint_blob(path, "final candidate" if final else "interaction candidate")
    allowed_estimators = {
        "constrained_crn_monte_carlo_mle_version4_natural_block",
        "size_stratified_sparse_natural_mcle_version4",
    }
    if blob.get("estimator") not in allowed_estimators:
        raise SystemExit(f"{path} is not a constrained Version-4 interaction checkpoint")
    rank = int(blob.get("active_rank", 0))
    if rank < 1 or rank > 8:
        raise SystemExit(f"{path} has invalid active interaction rank {rank}")
    if final and "household_size_rank1" not in blob:
        raise SystemExit(f"{path} has not completed the household-size stage")
    if blob.get("data_fingerprint_sha256") != current_data_fingerprint(dry_run=False):
        raise SystemExit(f"{path} does not match the current audited model dataset")
    if blob.get("fresh_artifact_digest") != initial["model_state_sha256"]:
        raise SystemExit(f"{path} was not descended from {initialization}")
    configured_artifact = blob.get("config", {}).get("artifact")
    if configured_artifact is None or Path(configured_artifact).name != initialization.name:
        raise SystemExit(f"{path} does not reference {initialization.name}")
    expected_batch = 128 if profile == "full" else 8
    if int(blob.get("config", {}).get("batch", -1)) != expected_batch:
        raise SystemExit(
            f"{path} belongs to a different profile; expected additive batch "
            f"{expected_batch} for --profile {profile}")
    return blob


def selected_rank_from_report(basis: Path, *, maximum_rank: int,
                              dry_run: bool,
                              parent_iteration: int | None = None,
                              parent: Path | None = None) -> int:
    require_files("rank", (basis, basis.with_suffix(".json")), dry_run=dry_run)
    if dry_run:
        return maximum_rank
    try:
        report = json.loads(basis.with_suffix(".json").read_text())
    except Exception as exc:
        raise SystemExit(f"cannot read rank report for {basis}: {exc}") from exc
    if (parent_iteration is not None
            and int(report.get("parent_iteration", -1)) != parent_iteration):
        raise SystemExit(
            f"{basis.with_suffix('.json')} was built from additive iteration "
            f"{report.get('parent_iteration')}, not the restored best iteration "
            f"{parent_iteration}")
    if parent is not None and report.get("parent_sha256") != file_sha256(parent):
        raise SystemExit(
            f"{basis.with_suffix('.json')} was not built from {parent}")
    if report.get("basis_sha256") != file_sha256(basis):
        raise SystemExit(f"{basis} failed its rank-report content digest")
    if report.get("data_fingerprint_sha256") != current_data_fingerprint(dry_run=False):
        raise SystemExit(f"{basis.with_suffix('.json')} belongs to another dataset")
    profiles = report.get("rank_stability", {})
    for rank in range(maximum_rank, 0, -1):
        if profiles.get(str(rank), {}).get("accepted"):
            return rank
    raise SystemExit(f"{basis.with_suffix('.json')} contains no accepted rank in 1..{maximum_rank}")


def basis_from_candidate_report(candidate: Path, *, dry_run: bool,
                                profile: str) -> Path:
    report_path = candidate.with_suffix(".json")
    require_files("evaluation", (report_path,), dry_run=dry_run)
    if dry_run:
        return ART / f"interaction_basis_rank{8 if profile == 'full' else 4}.npz"
    try:
        report = json.loads(report_path.read_text())
        reported = Path(report["spectral"])
        expected_digest = report["spectral_sha256"]
    except Exception as exc:
        raise SystemExit(f"cannot recover spectral basis from {report_path}: {exc}") from exc
    if reported.is_file():
        if file_sha256(reported) != expected_digest:
            raise SystemExit(f"spectral basis {reported} failed candidate lineage digest")
        return reported
    portable = ART / reported.name
    require_files("evaluation", (portable, portable.with_suffix(".json")), dry_run=False)
    if file_sha256(portable) != expected_digest:
        raise SystemExit(f"relocated spectral basis {portable} failed candidate lineage digest")
    return portable


def validate_evaluation_outputs(*, profile: str, candidate: Path,
                                dry_run: bool) -> None:
    """Ensure certification cannot follow missing, truncated, or rejected evaluations."""
    json_paths = (
        REPORT / "likelihood_validation.json",
        REPORT / "likelihood_test.json",
        REPORT / "recommendation.json",
        REPORT / "generation_counterfactual.json",
        REPORT / "customer_segments.json",
        REPORT / "interaction_embedding_audit.json",
    )
    assignments = ART / "customer_segments.npz"
    require_files("certification", (*json_paths, assignments), dry_run=dry_run)
    if dry_run:
        return
    candidate_digest = file_sha256(candidate)
    data_digest = current_data_fingerprint(dry_run=False)
    parsed = {}
    for path in json_paths:
        try:
            parsed[path.name] = json.loads(path.read_text())
        except Exception as exc:
            raise SystemExit(f"cannot resurrect evaluation; invalid {path}: {exc}") from exc
        if parsed[path.name].get("checkpoint_sha256") != candidate_digest:
            raise SystemExit(
                f"cannot resurrect evaluation: {path.name} belongs to another checkpoint")
        if parsed[path.name].get("data_fingerprint_sha256") != data_digest:
            raise SystemExit(
                f"cannot resurrect evaluation: {path.name} belongs to another dataset")
    if profile == "full":
        for name in ("likelihood_validation.json", "likelihood_test.json"):
            accepted = parsed[name].get("numerical_certification", {}).get("passed")
            if accepted is not True:
                raise SystemExit(
                    f"cannot start certification: {name} did not pass its full-profile "
                    "numerical/likelihood gate; rerun from --start-at evaluation")
    import numpy as np
    try:
        with np.load(assignments) as stored:
            if not stored.files:
                raise ValueError("archive contains no arrays")
        if parsed["customer_segments.json"].get(
                "assignments_sha256") != file_sha256(assignments):
            raise ValueError("archive content digest differs from segment report")
    except Exception as exc:
        raise SystemExit(
            f"cannot resurrect evaluation; invalid {assignments}: {exc}") from exc


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

    data_root = model_data_root()
    if from_raw:
        if data_root != ROOT.resolve():
            raise SystemExit(
                "--from-raw invokes the Dunnhumby builders and therefore cannot be "
                "combined with ENERGY_MODEL_DATA_ROOT; build an external dataset with "
                "its canonical adapter, then start this pipeline at initialize")
        raw = resolve_raw_directory()
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
            data_root / "data" / "price_week.parquet",
            data_root / "data" / "build_meta.json",
            data_root / "basket_input" / "meta.json",
            data_root / "basket_input" / "items.parquet",
            data_root / "basket_input" / "baskets.parquet",
            data_root / "basket_input" / "promo.npz",
        )
        if data_root == ROOT.resolve():
            required += (
                data_root / "data" / "tx.parquet",
                data_root / "data" / "price_store_week.parquet",
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
        self.environment["NF_RAW_DIR"] = str(resolve_raw_directory())

    def run(self, command: list[str], *, allow_failure: bool = False) -> int:
        self.commands.append(command)
        print(f"[pipeline] {command_text(command)}", flush=True)
        if self.dry_run:
            return 0
        if self.log_dir is None:
            code = subprocess.run(command, cwd=ROOT, env=self.environment, check=False).returncode
        else:
            from run_manifest import write_manifest, source_identity
            if source_identity() != RUN_STATUS["source_sha256"]:
                raise RuntimeError("source changed during original-data pipeline; freeze code and resume explicitly")
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
        if code and not allow_failure:
            raise SystemExit(code)
        return code


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
                          "--trips", contexts, "--draws", 8, "--rank", 8,
                          "--threads", threads,
                          "--output", output))
        return 8, output
    maximum_rank = 4 if smoke else 8
    output = ART / f"interaction_basis_rank{maximum_rank}.npz"
    pending = ART / f".{output.stem}.pending-{os.getpid()}.npz"
    pending_report = pending.with_suffix(".json")
    for path in (pending, pending_report):
        path.unlink(missing_ok=True)

    def reject_pending(message: str) -> None:
        rejected = ART / f"rejected_rank_{time.time_ns()}"
        rejected.mkdir()
        for path in (pending, pending_report):
            if path.exists():
                os.replace(path, rejected / path.name)
        raise SystemExit(message)

    status = driver.run(script("build_spectral_phi_initialization.py", "--parent", parent,
                      "--trips", contexts, "--draws", 2 if smoke else 8, "--rank", maximum_rank,
                      "--threads", threads,
                      "--minimum-stability", -1.0 if smoke else 0.5,
                      "--output", pending), allow_failure=True)
    if status != 0 or not pending.is_file() or not pending_report.is_file():
        reject_pending("spectral rank build failed; no prior basis was reused")
    try:
        report = json.loads(pending_report.read_text())
    except Exception as exc:
        reject_pending(f"new spectral rank report is invalid: {exc}")
    if report.get("parent_sha256") != file_sha256(parent):
        reject_pending("new spectral rank report does not match its additive parent")
    if report.get("basis_sha256") != file_sha256(pending):
        reject_pending("new spectral basis failed its report content digest")
    if report.get("data_fingerprint_sha256") != current_data_fingerprint(dry_run=False):
        reject_pending("new spectral rank report does not match the audited dataset")
    rank = report.get("largest_stable_rank")
    if rank is None or not 1 <= int(rank) <= maximum_rank:
        reject_pending("no rank in 1..8 passed the predeclared split-half gate")
    os.replace(pending, output)
    os.replace(pending_report, output.with_suffix(".json"))
    print(f"[pipeline] selected independently stable rank {rank}")
    return int(rank), output


def main() -> None:
    global ART, REPORT, OUT, RUN_MANIFEST, RUN_STATUS
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path,
                        help="isolated artifacts/reports/out under a new run directory; explicit --start-at permits reuse")
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
    parser.add_argument(
        "--price-coefficients", type=Path,
        help=("certified direct joint-basket price coefficient artifact; when supplied, "
              "skip the legacy price-change evidence stage and use this immutable input"))
    parser.add_argument(
        "--start-at", choices=STAGES,
        help=("resurrect a partially completed pipeline from this stage; all earlier "
              "artifacts are validated and reused"))
    parser.add_argument("--stop-after", choices=STAGES, default="certification")
    parser.add_argument(
        "--rebuild-interaction-bank", action="store_true",
        help="resample the stratified estimator's derived draw cache")
    args = parser.parse_args()
    if args.run_dir is not None:
        run_dir = args.run_dir.resolve()
        if not args.dry_run:
            # A continuation gets a new invocation log without replacing the old one.
            fresh = args.start_at in (None, "data", "initialize") and args.resume_additive is None
            run_dir.mkdir(parents=True, exist_ok=not fresh)
        ART, REPORT, OUT = run_dir / "artifacts", run_dir / "reports", run_dir / "out"
        if not args.dry_run:
            from run_manifest import Tee, write_manifest, source_identity
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            log_dir = run_dir / f"invocation_{stamp}"
            log_dir.mkdir(exist_ok=False)
            sys.stdout = Tee(sys.stdout, log_dir / "pipeline.log")
            sys.stderr = sys.stdout
            RUN_MANIFEST = log_dir / "manifest.json"
            RUN_STATUS = {"status": "running", "pid": os.getpid(), "started_utc": stamp,
                          "run_dir": str(run_dir), "profile": args.profile,
                          "model_data_root": str(model_data_root()),
                          "source_sha256": source_identity(), "stages": [],
                          "configuration": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
            write_manifest(RUN_MANIFEST, RUN_STATUS)
    start_at = args.start_at or ("additive" if args.resume_additive else "data")
    if args.from_raw and args.resume_additive is not None:
        parser.error("--from-raw cannot be combined with --resume-additive")
    if args.from_raw and start_at != "data":
        parser.error("--from-raw requires --start-at data (or no --start-at)")
    if args.resume_additive is not None and start_at != "additive":
        parser.error("--resume-additive requires --start-at additive")
    if args.resume_additive is not None and args.price_coefficients is not None:
        parser.error("--price-coefficients cannot change during --resume-additive")
    if stage_index(args.stop_after) < stage_index(start_at):
        parser.error("--stop-after must be the same as or later than --start-at")
    if args.threads < 0:
        parser.error("--threads cannot be negative")
    preflight(from_raw=args.from_raw, stop_after=args.stop_after)
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
    if not args.dry_run:
        write_runtime_report(
            ART / "runtime_capabilities.json", capabilities,
            requested_device=args.device, selected_device=selected_device,
            cpu_threads=cpu_threads)
    accelerator = (f", CUDA devices={capabilities.cuda_device_count} (not eligible for "
                   "the exact normalizer)" if capabilities.cuda_device_count else "")
    print(f"[pipeline] backend={selected_device}, CPU threads={cpu_threads}, "
          f"RAM={capabilities.memory_gib or 'unknown'} GiB{accelerator}", flush=True)
    print(f"[pipeline] execution window: {start_at} -> {args.stop_after}", flush=True)
    print(f"[pipeline] model data root: {model_data_root()}", flush=True)
    print(f"[pipeline] hardware report: {ART / 'runtime_capabilities.json'}", flush=True)
    driver = Driver(args.dry_run, log_dir if args.run_dir is not None and not args.dry_run else None)

    if args.from_raw:
        driver.run([PY, "-u", "scripts/data/01_build_base.py"])
        driver.run([PY, "-u", "scripts/data/22_basket_data.py"])
        driver.run([PY, "-u", "scripts/data/23_promo_data.py"])
    # Always fail closed on data integrity, including when reusing derived files.
    if start_at == "data":
        if model_data_root() == ROOT.resolve():
            driver.run([PY, "-u", "scripts/data/audit_preprocessing.py"])
            driver.run(script("build_affinity_partition.py"))
            driver.run(script("data.py", "--force"))
            driver.run(script("provenance.py"))
        else:
            # External adapters own their preprocessing and immutable partition.  Running
            # the Dunnhumby audit here would inspect ROOT rather than the selected bundle.
            verify_data_bundle(dry_run=driver.dry_run)
            print("[pipeline] verified canonical external model-data bundle", flush=True)
    else:
        # Reuse the audited input without overwriting shared historical artifacts.
        verify_data_bundle(dry_run=driver.dry_run)
    if args.stop_after == "data":
        return

    driver.run([PY, str(V4 / "setup_poly_degree_native.py"), "build_ext",
                "--build-lib", str(ART / "native" / "lib"),
                "--build-temp", str(ART / "native" / "temp"), "--force"])
    initialization = ART / "initialization.pt"
    if runs_stage(start_at, "initialize"):
        driver.run(script("initialize_version4.py", "--output", initialization,
                          "--manifest", ART / "initialization.json",
                          "--household-size-rank1", "--threads", cpu_threads))
        initialization_blob = validate_initialization(
            initialization, dry_run=driver.dry_run)
    else:
        initialization_blob = validate_initialization(
            initialization, dry_run=driver.dry_run)
        print(f"[pipeline] resurrected initialization: {initialization}", flush=True)
    initialization_nmax = (120 if initialization_blob is None else
                           int(initialization_blob["metadata"]["nmax"]))
    if args.stop_after == "initialize":
        return

    full = args.profile == "full"
    additive_iterations = 30000 if full else 10
    additive = OUT / "v3_pipeline_additive_best.pt"
    additive_latest = OUT / "v3_pipeline_additive.pt"
    additive_blob = None
    if runs_stage(start_at, "additive"):
        price_evidence_dir = ART / "supported_price_response"
        price_evidence_report = price_evidence_dir / "report.json"
        price_coefficients = price_evidence_dir / "coefficients.json"
        supplied_price = args.price_coefficients
        if supplied_price is not None:
            supplied_price = (supplied_price if supplied_price.is_absolute()
                              else ROOT / supplied_price).resolve()
            require_files("additive", (supplied_price,), dry_run=driver.dry_run)
            if not driver.dry_run:
                try:
                    supplied_payload = json.loads(supplied_price.read_text())
                except Exception as exc:
                    raise SystemExit(
                        f"cannot read supplied price coefficients {supplied_price}: {exc}") from exc
                if supplied_payload.get("status") != "certified_observational_predictor":
                    raise SystemExit("--price-coefficients is not a certified artifact")
                if supplied_payload.get("price_feature_contract") != "chain_product_week_only":
                    raise SystemExit(
                        "direct current-price coefficients must declare the chain-only feature contract")
            price_configuration = ("--supported-price-coefficients", supplied_price)
            print(f"[pipeline] using supplied direct joint-basket price coefficients: "
                  f"{supplied_price}", flush=True)
        else:
            # The evidence stage exits 2 when no price component is certified.  That is
            # a verdict, not a crash: fit the basket law with price fixed to zero.
            evidence_status = driver.run(script(
                "fit_supported_price_response.py",
                "--output", price_evidence_report,
                "--coefficients-output", price_coefficients,
                "--seed", 68101), allow_failure=True)
            if driver.dry_run:
                price_configuration = ("--supported-price-coefficients",
                                       price_coefficients)
            else:
                price_configuration = price_configuration_from_evidence(
                    evidence_status, price_evidence_report, price_coefficients)
        if args.resume_additive is not None:
            resume_path = (args.resume_additive if args.resume_additive.is_absolute()
                           else ROOT / args.resume_additive)
            validate_additive(resume_path, initialization, dry_run=driver.dry_run)
            print(f"[pipeline] validated additive continuation checkpoint: {resume_path}",
                  flush=True)
        additive_command = script(
            "fit_exact_additive.py", "--artifact", initialization,
            "--output-dir", OUT,
            "--label", "pipeline_additive", "--iters", additive_iterations,
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
            "--threads", cpu_threads,
            "--rho-c-max-category-reward", 1.5,
            *(('--require-convergence',) if full else ()),
            *(('--resume', args.resume_additive)
              if args.resume_additive is not None else ()))
        driver.run(additive_command)
    else:
        # Certification also reads the additive parent (size-phase diagnostic), so its
        # lineage is validated on every resumed invocation.
        additive_blob = validate_completed_additive(
            additive, additive_latest, initialization,
            profile=args.profile, dry_run=driver.dry_run)
        print(f"[pipeline] resurrected converged additive parent: {additive}", flush=True)
    if args.stop_after == "additive":
        return

    if runs_stage(start_at, "rank"):
        rank, basis = rank_selection(driver, additive, 50000 if full else 128,
                                      smoke=not full, threads=cpu_threads)
    elif start_at == "interaction":
        basis = ART / f"interaction_basis_rank{8 if full else 4}.npz"
        rank = selected_rank_from_report(
            basis, maximum_rank=8 if full else 4, dry_run=driver.dry_run,
            parent_iteration=(int(additive_blob["iter"])
                              if additive_blob is not None else None),
            parent=additive if not driver.dry_run else None)
        print(f"[pipeline] resurrected rank selection: rank={rank}, basis={basis}",
              flush=True)
    else:
        # Evaluation recovers the basis from candidate.json. Certification needs only
        # the active rank stored in the final checkpoint.
        rank, basis = 0, None
    if args.stop_after == "rank":
        return

    interaction_candidate = ART / "candidate.pt"
    candidate = ART / "candidate_rank1.pt"
    if runs_stage(start_at, "interaction"):
        driver.run(script(
            "fit_stratified_natural_interactions.py", "--parent", additive,
            "--spectral", basis, "--contexts", 12000 if full else 64,
            "--band-draws", *stratified_band_draws(
                initialization_nmax, full=full),
            "--batch", 96 if full else 8, "--rank", rank,
            "--score-mass", 1.0, "--spectral-max", 1.0,
            "--category-bound", 0.0,
            "--size-ridge", 1e-3, "--size-smoothness", 1e-1,
            *(("--rebuild-bank",) if args.rebuild_interaction_bank else ()),
            "--threads", cpu_threads,
            "--minimum-crossfit-gain", 0.005 if full else -1.0,
            "--minimum-half-gain", 0.0 if full else -1e9,
            "--minimum-within-band-ess-fraction", 0.20 if full else 0.0,
            "--minimum-within-band-ess", 2.0 if full else 0.0,
            "--output", interaction_candidate))
        if not driver.dry_run:
            interaction_report = json.loads(
                interaction_candidate.with_suffix(".json").read_text())
            eigenvalues = interaction_report["candidate_c_eigenvalues"]
            tolerance = max(eigenvalues) * 1e-10 if eigenvalues else 0.0
            fitted_rank = sum(value > max(tolerance, 1e-12) for value in eigenvalues)
            if fitted_rank < 1:
                raise SystemExit(
                    "natural-parameter solve produced no positive interaction rank")
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
            "--threads", cpu_threads,
            "--output", candidate,
            "--report", ART / "candidate_rank1.json",
            "--population-output", REPORT / "population_size.json"))
    else:
        final_blob = validate_candidate(
            candidate, initialization, final=True, profile=args.profile,
            dry_run=driver.dry_run)
        if final_blob is not None:
            rank = int(final_blob["active_rank"])
        else:
            rank = 8 if full else 4
        if start_at == "evaluation":
            basis = basis_from_candidate_report(
                interaction_candidate, dry_run=driver.dry_run, profile=args.profile)
        print(f"[pipeline] resurrected final candidate: rank={rank}, "
              f"checkpoint={candidate}", flush=True)
    if args.stop_after == "interaction":
        return

    # All claims use fixed panels and complete support; recommendation is read-only.
    if runs_stage(start_at, "evaluation"):
        driver.run(script(
            "compare_rank8_parent_likelihood.py", "--parent", additive,
            "--child", candidate, "--split", "validation",
            "--trips", 4096 if full else 16,
            "--rank", rank, "--target-level", rank + 2,
            "--audit-trips", 128 if full else 4,
            "--threads", cpu_threads,
            *(('--maximum-audit-error-bound', 0.01, '--require-certified-gain')
              if full else ()),
            "--output", REPORT / "likelihood_validation.json"))
        driver.run(script(
            "compare_rank8_parent_likelihood.py", "--parent", additive,
            "--child", candidate, "--split", "test",
            "--trips", 4096 if full else 16,
            "--rank", rank, "--target-level", rank + 2,
            "--audit-trips", 128 if full else 4,
            "--threads", cpu_threads,
            *(('--maximum-audit-error-bound', 0.01)
              if full else ()),
            "--output", REPORT / "likelihood_test.json"))
        driver.run(script(
            "eval_smolyak_rank8_mrr.py", "--ckpt", candidate, "--split", "test",
            "--parent", additive,
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
    else:
        validate_evaluation_outputs(profile=args.profile, candidate=candidate,
                                    dry_run=driver.dry_run)
        print("[pipeline] resurrected completed evaluation artifacts", flush=True)
    if args.stop_after == "evaluation":
        return

    validate_evaluation_outputs(profile=args.profile, candidate=candidate,
                                dry_run=driver.dry_run)

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
        "diagnose_size_phase.py", "--parent", additive, "--child", candidate,
        "--confirmed-panel", REPORT / "population_size_confirm_per_trip.npz",
        "--contexts-per-panel", 64 if full else 2,
        "--threads", cpu_threads,
        "--output", REPORT / "size_phase_diagnostic.json"))
    pricing_command = script(
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
        "--minimum-budget-utilization", 0.0,
        "--output", REPORT / "segment_promotion_mdp.json")
    if driver.dry_run:
        print("[pipeline] pricing policy is conditional on a nonzero, held-out-supported "
              "price component", flush=True)
        driver.run(pricing_command)
    else:
        price_estimator = load_checkpoint_blob(
            candidate, "final candidate").get("price_response_estimator")
        if price_estimator == "fixed_zero_after_failed_heldout_support":
            skipped = {
                "status": "not_assessed",
                "reason": ("held-out price evidence failed and the fitted price "
                           "response is fixed to zero; policy computation would be "
                           "uninformative"),
                "checkpoint": str(candidate),
                "checkpoint_sha256": file_sha256(candidate),
                "price_response_estimator": price_estimator,
            }
            destination = REPORT / "segment_promotion_mdp.json"
            destination.write_text(
                json.dumps(skipped, indent=2, sort_keys=True) + "\n")
            print(f"[pipeline] skipped pricing policy; verdict written to {destination}",
                  flush=True)
        else:
            driver.run(pricing_command)
    if args.dry_run:
        print("[pipeline] dry run complete; no stage was executed")
    elif full:
        print("[pipeline] fitted-model likelihood/numerical and population-safety gates "
              "passed; candidate_rank1.pt is accepted for conditional-incidence "
              "likelihood use. Generation calibration, causal price response and policy "
              "value require separate capability verdicts", flush=True)
    else:
        print("[pipeline] smoke integration completed; statistical gates were relaxed "
              f"and tail audit status was {tail_status}; this is not a certified fit")


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
