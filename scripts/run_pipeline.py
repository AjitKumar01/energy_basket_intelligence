#!/usr/bin/env python3
"""One reproducible driver for the selected Version-4 staged pipeline."""
from __future__ import annotations

import argparse
import hashlib
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
STAGES = ("data", "initialize", "additive", "rank", "interaction",
          "evaluation", "certification")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def current_data_fingerprint(*, dry_run: bool) -> str | None:
    path = ROOT / "basket_input" / "model_data_fingerprint.json"
    require_files("data", (path,), dry_run=dry_run)
    if dry_run:
        return None
    try:
        payload = json.loads(path.read_text())
        recorded = str(payload.pop("fingerprint_sha256"))
        canonical = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            allow_nan=False).encode("utf-8")
        if hashlib.sha256(canonical).hexdigest() != recorded:
            raise ValueError("self digest does not match")
        return recorded
    except Exception as exc:
        raise SystemExit(f"cannot read model-data fingerprint {path}: {exc}") from exc


def stage_index(stage: str) -> int:
    return STAGES.index(stage)


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
    if blob.get("estimator") != "constrained_crn_monte_carlo_mle_version4_natural_block":
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
    for rank in range(maximum_rank, 3, -1):
        if profiles.get(str(rank), {}).get("accepted"):
            return rank
    raise SystemExit(f"{basis.with_suffix('.json')} contains no accepted rank in 4..{maximum_rank}")


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
            ROOT / "data" / "build_meta.json",
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
    pending = ART / f".{output.stem}.pending-{os.getpid()}.npz"
    pending_report = pending.with_suffix(".json")
    for path in (pending, pending_report):
        path.unlink(missing_ok=True)

    def reject_pending(message: str) -> None:
        pending.unlink(missing_ok=True)
        pending_report.unlink(missing_ok=True)
        raise SystemExit(message)

    status = driver.run(script("build_spectral_phi_initialization.py", "--parent", parent,
                      "--trips", contexts, "--draws", 2, "--rank", maximum_rank,
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
    if rank is None or not 4 <= int(rank) <= maximum_rank:
        reject_pending("no rank in 4..8 passed the predeclared split-half gate")
    os.replace(pending, output)
    os.replace(pending_report, output.with_suffix(".json"))
    print(f"[pipeline] selected independently stable rank {rank}")
    return int(rank), output


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
    parser.add_argument(
        "--start-at", choices=STAGES,
        help=("resurrect a partially completed pipeline from this stage; all earlier "
              "artifacts are validated and reused"))
    parser.add_argument("--stop-after", choices=STAGES, default="certification")
    args = parser.parse_args()
    start_at = args.start_at or ("additive" if args.resume_additive else "data")
    if args.from_raw and args.resume_additive is not None:
        parser.error("--from-raw cannot be combined with --resume-additive")
    if args.from_raw and start_at != "data":
        parser.error("--from-raw requires --start-at data (or no --start-at)")
    if args.resume_additive is not None and start_at != "additive":
        parser.error("--resume-additive requires --start-at additive")
    if stage_index(args.stop_after) < stage_index(start_at):
        parser.error("--stop-after must be the same as or later than --start-at")
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
    print(f"[pipeline] execution window: {start_at} -> {args.stop_after}", flush=True)
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
    driver.run(script("provenance.py"))
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
    else:
        validate_initialization(initialization, dry_run=driver.dry_run)
        print(f"[pipeline] resurrected initialization: {initialization}", flush=True)
    if args.stop_after == "initialize":
        return

    full = args.profile == "full"
    additive_iterations = 30000 if full else 10
    additive = ROOT / "out" / "v3_pipeline_additive_best.pt"
    additive_latest = ROOT / "out" / "v3_pipeline_additive.pt"
    additive_blob = None
    if runs_stage(start_at, "additive"):
        if args.resume_additive is not None:
            resume_path = (args.resume_additive if args.resume_additive.is_absolute()
                           else ROOT / args.resume_additive)
            validate_additive(resume_path, initialization, dry_run=driver.dry_run)
            print(f"[pipeline] validated additive continuation checkpoint: {resume_path}",
                  flush=True)
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
    elif start_at != "certification":
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
                raise SystemExit(
                    "natural-parameter solve produced no positive interaction rank")
            if fitted_rank != rank:
                print(f"[pipeline] convex solve reduced certified basis rank {rank} "
                      f"to active rank {fitted_rank}")
            rank = fitted_rank
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
