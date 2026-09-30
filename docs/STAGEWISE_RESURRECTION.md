# Stage-wise recovery and checkpoint portability

## 1. Purpose

The pipeline is staged because the additive fit, rank audit, interaction fit, refinement,
evaluation and certification have different costs. A failure after a completed stage must not
force earlier stages to run again. The driver supports two recovery operations:

1. **Continue an interrupted additive optimization.** This restores the model, Adam state,
   learning-rate scheduler, validation history and deterministic minibatch stream.
2. **Start at the next completed stage.** This validates and reuses earlier artifacts, then
   executes only the requested suffix of the pipeline.

Neither weakens a statistical gate. Recovery reuses a valid checkpoint; it never declares a
partial model converged.

## 2. Run directory and stage graph

Every run lives under `--run-dir <run>`:

- `<run>/artifacts/`: initialization, spectral basis, fitted checkpoints, native extension;
- `<run>/out/`: additive checkpoints and their optimisation log;
- `<run>/reports/`: evaluation and certification reports;
- `<run>/invocation_<UTC time>/`: one folder per invocation, with `pipeline.log`, one log per
  stage command, and `manifest.json` (status, per-stage exit codes and run times, and the
  source-file digests of the code that ran).

The ordered stages are:

| Stage | Work performed | Principal durable output |
|---|---|---|
| `data` | verify the model-data bundle's fingerprint and re-hash every model-facing file | (read only) |
| `initialize` | fresh Version-4 parameter initialization | `artifacts/initialization.pt` |
| `additive` | price evidence, then exact complete-support additive maximum likelihood | `out/v3_pipeline_additive_best.pt`, `out/v3_pipeline_additive.pt` |
| `rank` | split-half interaction-rank audit and spectral basis | `artifacts/interaction_basis_rank8.{npz,json}` (full profile) |
| `interaction` | constrained interaction solve, then household-size recalibration | `artifacts/candidate_rank1.{pt,json}` |
| `refinement` | optional (`--joint-refinement`) joint refinement with its acceptance gates | `artifacts/candidate_refined.pt` if accepted; `reports/joint_refinement.json` |
| `evaluation` | validation/test likelihood, recommendation, generation, segmentation, embedding audit | files under `reports/` plus `artifacts/customer_segments.npz` |
| `certification` | complete-population size audit, size-phase diagnostic, promotion MDP | `reports/population_size.json`, `reports/segment_promotion_mdp.json` |

`--start-at STAGE` reuses everything before `STAGE`; `--stop-after STAGE` stops after it.

The model-data fingerprint (`<bundle>/basket_input/model_data_fingerprint.json`) is the root of
the artifact lineage. Every checkpoint and report names it; matching dimensions, file names or
iteration numbers are not a substitute for matching content. While a run is in progress the
driver also refuses to continue if any source file under `scripts/` or `tests/` changes.

## 3. Fresh execution and deliberate stopping

```bash
python scripts/run_pipeline.py --model-data-root <bundle> --run-dir <run> --profile full
python scripts/run_pipeline.py --model-data-root <bundle> --run-dir <run> --profile full --stop-after rank
```

Stopping after a stage is different from killing a process during it: the former has completed
the stage gate; the latter may leave only a partial checkpoint.

## 4. Continuing an interrupted additive fit

`out/v3_pipeline_additive.pt` holds the latest resumable state and
`out/v3_pipeline_additive_best.pt` the best validation model. Resume from the latest, keep
`artifacts/initialization.pt` (the checkpoint is tied to it by digest), and run:

```bash
python scripts/run_pipeline.py --model-data-root <bundle> --run-dir <run> --profile full \
    --start-at additive --resume-additive <run>/out/v3_pipeline_additive.pt
```

Before restarting, the driver verifies the initialization's parameterization and data
fingerprint, that the checkpoint is an exact Version-4 additive checkpoint with the matching
initialization digest, and that the batch and seed contract match. The full run keeps its
declared update ceiling; a checkpoint that reached it without converging is a statistical
failure, not an interrupted run.

## 5. Starting after a completed stage

| Start at | Required files (under the run directory) |
|---|---|
| `additive` | `artifacts/initialization.pt` |
| `rank` | the above, `out/v3_pipeline_additive_best.pt`, `out/v3_pipeline_additive.pt` |
| `interaction` | the above, `artifacts/interaction_basis_rank8.{npz,json}` |
| `refinement` | the above, `artifacts/candidate.{pt,json}`, `artifacts/candidate_rank1.pt` |
| `evaluation` | the same as `refinement`; with `--joint-refinement`, also `reports/joint_refinement.json` |
| `certification` | `artifacts/initialization.pt`, the final candidate, and every evaluation report plus `artifacts/customer_segments.npz` |

Notes:
- Both additive checkpoints are required: the best file supplies the model; the latest proves
  the optimizer met its convergence rule.
- The rank stage's JSON receipt must name the restored best additive checkpoint, and the NPZ
  hash must match it.
- With `--joint-refinement`, the evaluation candidate is `candidate_refined.pt` only if
  `reports/joint_refinement.json` records an accepted decision whose parent digest is that of
  `candidate_rank1.pt`; otherwise it is `candidate_rank1.pt`.
- Each evaluation report contains the SHA-256 of the evaluated checkpoint and the data
  fingerprint, so a partial or mixed set of reports is rejected before certification starts.
- Smoke and full artifacts cannot be mixed (`--profile smoke` uses a rank-4 basis).
- The population-screen cache files `reports/population_size.screen-<digest>.*` are optional;
  without them certification recomputes the screen.

## 6. Moving a run to another machine

Use the same repository revision and exact dependencies, rebuild the same bundle (the
fingerprint must match), then copy the run directory preserving relative paths:

```bash
git clone <repository> && cd <repository>
python -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/prepare_model_bundle.py --dataset-config configs/datasets/<name>.json
rsync -av training-host:/path/to/<run>/ <run>/
python scripts/run_pipeline.py --model-data-root <bundle> --run-dir <run> --profile full --start-at <stage>
```

The checkpoint loader relocates stale absolute initialization paths to the new run directory.
Learned artifacts are git-ignored; transfer them through approved artifact storage.

## 7. Validation before reuse

Recovery fails before expensive computation when it finds: a missing prerequisite; an
unreadable or wrong-format checkpoint; an additive checkpoint with the wrong initialization
digest; best and latest additive checkpoints from different runs; a full-profile additive run
that did not converge; a smoke/full mismatch; a rank report built from a different additive
iteration; a candidate with no positive interaction rank or without the household-size stage;
a refinement report built from a different staged checkpoint; a missing, truncated or invalid
evaluation report; or a full-profile likelihood evaluation whose numerical certification did
not pass. Restore the matching artifact set or start at an earlier stage; never rename an
unrelated checkpoint to the expected file name.

## 8. Inspection before spending compute

```bash
python scripts/run_pipeline.py --model-data-root <bundle> --run-dir <run> --start-at interaction --dry-run
python scripts/run_pipeline.py --help
```

## 9. Recovery decision table

| Last trustworthy state | Action |
|---|---|
| Bundle prepared | fresh run |
| Initialization completed | `--start-at additive` |
| Additive interrupted below its ceiling | `--start-at additive --resume-additive <run>/out/v3_pipeline_additive.pt` |
| Additive converged | `--start-at rank` |
| Rank report completed | `--start-at interaction` |
| `candidate_rank1.pt` completed | `--start-at refinement` (or `evaluation` without refinement) |
| Refinement report completed | `--start-at evaluation` |
| All evaluation files completed | `--start-at certification` |
| Unsure whether a stage completed | begin at that stage, not the next |
