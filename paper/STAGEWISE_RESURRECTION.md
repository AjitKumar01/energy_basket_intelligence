# Stage-wise recovery and checkpoint portability

## 1. Purpose

The full pipeline is deliberately staged because the exact additive fit, rank audit,
interaction fit, evaluation, and production certification have different costs. A machine
failure after a completed stage must not force earlier stages to run again.

The driver supports two distinct recovery operations:

1. **Continue an interrupted additive optimization.** This restores the model, Adam
   state, learning-rate scheduler, validation history, and deterministic minibatch stream.
2. **Start at the next completed stage.** This validates and reuses earlier artifacts,
   then executes only the requested suffix of the pipeline.

These operations do not weaken any statistical gate. “Recovery” means reusing a valid
checkpoint, not declaring a partial model converged.

## 2. The stage graph

The ordered stages are:

| Stage | Work performed | Principal durable output |
|---|---|---|
| `data` | preprocessing audit, training-only affinity partition, ragged cache | `basket_input/preprocessing_manifest.json` |
| `initialize` | fresh Version-4 parameter initialization | `artifacts/initialization.pt` |
| `additive` | exact complete-support additive maximum likelihood | `out/v3_pipeline_additive_best.pt` and `out/v3_pipeline_additive.pt` |
| `rank` | split-half interaction-rank audit and spectral basis | `artifacts/interaction_basis_rank8.{npz,json}` for full runs |
| `interaction` | constrained interaction solve and household-size recalibration/safety projection | `artifacts/candidate_rank1.{pt,json}` |
| `evaluation` | validation/test likelihood, recommendation, generation, segmentation, embedding audit | files under `reports/` plus `artifacts/customer_segments.npz` |
| `certification` | complete-population size audit and promotion MDP | `reports/population_size.json` and `reports/segment_promotion_mdp.json` |

`--start-at STAGE` reuses everything before `STAGE`. `--stop-after STAGE` stops after that
stage. The requested interval must follow the order above.

Every recovery invocation still performs the data-integrity audit, reconstructs the
deterministic affinity partition and ragged cache, and rebuilds the local native extension.
This is intentional: a checkpoint must never be evaluated against silently different data
or an incompatible local binary.

## 3. Fresh execution and deliberate stopping

Run everything from raw CSV files:

```bash
export NF_RAW_DIR="/absolute/path/to/dunnhumby_The-Complete-Journey CSV"
mkdir -p artifacts
python scripts/run_pipeline.py --from-raw --profile full \
  2>&1 | tee artifacts/pipeline.log
```

To stop at a durable boundary:

```bash
python scripts/run_pipeline.py --profile full --stop-after additive \
  2>&1 | tee artifacts/pipeline.log

python scripts/run_pipeline.py --profile full --stop-after rank \
  2>&1 | tee -a artifacts/pipeline.log
```

Stopping after a stage is different from killing a process during that stage. The former
has completed the stage gate; the latter may have only a partial checkpoint.

## 4. Continuing an interrupted additive fit

During additive training, the latest resumable state is
`out/v3_pipeline_additive.pt`. The best validation model is
`out/v3_pipeline_additive_best.pt`. They serve different purposes:

- resume from `v3_pipeline_additive.pt`, because it contains the latest optimizer and
  scheduler state;
- use `v3_pipeline_additive_best.pt` as the parent after convergence;
- keep `artifacts/initialization.pt`, because the checkpoint is cryptographically tied to
  that initialization state.

Continue after a process interruption with:

```bash
python scripts/run_pipeline.py --profile full --start-at additive \
  --resume-additive out/v3_pipeline_additive.pt \
  2>&1 | tee -a artifacts/pipeline.log
```

If `--resume-additive` is supplied without `--start-at`, the driver automatically chooses
`--start-at additive`. The explicit form is recommended in operational scripts.

Before restarting compute, the driver verifies that:

- the initialization artifact uses the rank-one household-size parameterization;
- the checkpoint is an exact Version-4 additive checkpoint;
- its stored initialization digest matches `artifacts/initialization.pt`; and
- the resumed batch and seed contract match the original run.

The full run still has its declared 30,000-update ceiling. A checkpoint that reached that
ceiling without satisfying convergence is a statistical failure, not an interrupted run.
Changing the ceiling after seeing the result creates a different experiment and is not
performed automatically.

## 5. Starting after a completed stage

### 5.1 Reuse initialization and begin additive fitting

Required file:

```text
artifacts/initialization.pt
```

Command:

```bash
python scripts/run_pipeline.py --profile full --start-at additive \
  2>&1 | tee -a artifacts/pipeline.log
```

Without `--resume-additive`, the additive optimizer starts from the stored fresh
initialization. It does not reuse a partially trained model.

### 5.2 Reuse a completed additive fit and begin rank selection

Required files:

```text
artifacts/initialization.pt
out/v3_pipeline_additive_best.pt
out/v3_pipeline_additive.pt
```

Command:

```bash
python scripts/run_pipeline.py --profile full --start-at rank \
  2>&1 | tee -a artifacts/pipeline.log
```

Both additive checkpoints are required. The best file supplies the model; the latest file
proves that the optimizer actually met its convergence rule. Merely finding a best
checkpoint from an aborted run is not enough.

### 5.3 Reuse rank selection and begin interaction fitting

Required files:

```text
artifacts/initialization.pt
out/v3_pipeline_additive_best.pt
out/v3_pipeline_additive.pt
artifacts/interaction_basis_rank8.npz
artifacts/interaction_basis_rank8.json
```

Command:

```bash
python scripts/run_pipeline.py --profile full --start-at interaction \
  2>&1 | tee -a artifacts/pipeline.log
```

The driver reads the largest accepted rank from the JSON report and verifies that the
basis was built from the restored best additive iteration. It then runs both parts of the
interaction stage: the constrained natural-parameter fit and the post-interaction
household-size recalibration/safety projection.

For a smoke run, use `--profile smoke`; its expected basis is
`interaction_basis_rank4.{npz,json}`. Smoke and full artifacts cannot be mixed.

### 5.4 Reuse the final fitted model and run evaluations

Required files:

```text
artifacts/initialization.pt
out/v3_pipeline_additive_best.pt
out/v3_pipeline_additive.pt
artifacts/candidate.pt
artifacts/candidate.json
artifacts/candidate_rank1.pt
artifacts/interaction_basis_rank8.npz
artifacts/interaction_basis_rank8.json
```

`candidate_rank1.json` is strongly recommended for audit history even though the scoring
programs consume the checkpoint itself.

Command:

```bash
python scripts/run_pipeline.py --profile full --start-at evaluation \
  2>&1 | tee -a artifacts/pipeline.log
```

The active interaction rank comes from `candidate_rank1.pt`. The spectral basis path is
recovered from `candidate.json`; if it contains an absolute path from another computer,
the driver resolves the file by name under the new clone's `artifacts/` directory.

### 5.5 Reuse evaluations and run production certification

Required model files:

```text
artifacts/initialization.pt
artifacts/candidate_rank1.pt
```

Required evaluation files:

```text
reports/likelihood_validation.json
reports/likelihood_test.json
reports/recommendation.json
reports/generation_counterfactual.json
reports/customer_segments.json
reports/interaction_embedding_audit.json
artifacts/customer_segments.npz
```

Command:

```bash
python scripts/run_pipeline.py --profile full --start-at certification \
  2>&1 | tee -a artifacts/pipeline.log
```

The population screen cache files named
`reports/population_size.screen-<digest>.*` are optional. Copying them can save time; if
they are absent, certification recomputes the screen. The checkpoint's active rank is used
directly, so the additive parent and spectral basis are not required for certification.

## 6. Moving a partially completed run to another machine

First use the same repository revision and exact Python dependencies:

```bash
git clone https://github.com/AjitKumar01/energy_basket_intelligence.git
cd energy_basket_intelligence
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
export NF_RAW_DIR="/absolute/path/to/dunnhumby_The-Complete-Journey CSV"
```

Rebuild and audit the derived data before copying learned artifacts:

```bash
python scripts/run_pipeline.py --from-raw --profile full --stop-after data
```

Copy files while preserving their repository-relative paths. For example, to continue at
the interaction stage from a machine named `training-host`:

```bash
rsync -av training-host:/path/to/energy_basket_intelligence/artifacts/initialization.pt artifacts/
rsync -av training-host:/path/to/energy_basket_intelligence/artifacts/interaction_basis_rank8.npz artifacts/
rsync -av training-host:/path/to/energy_basket_intelligence/artifacts/interaction_basis_rank8.json artifacts/
rsync -av training-host:/path/to/energy_basket_intelligence/out/v3_pipeline_additive_best.pt out/
rsync -av training-host:/path/to/energy_basket_intelligence/out/v3_pipeline_additive.pt out/

python scripts/run_pipeline.py --profile full --start-at interaction \
  2>&1 | tee -a artifacts/pipeline.log
```

Learned artifacts are intentionally ignored by Git because they are large and may contain
derived customer information. They must be transferred through the organization's
approved artifact storage, not committed to the public repository.

The checkpoint loader relocates both relative and stale absolute initialization paths. A
checkpoint created under `/old/machine/project/artifacts/initialization.pt` will use
`artifacts/initialization.pt` in the new clone when the old path does not exist.

## 7. What validation happens before reuse

The resurrection path fails before expensive computation when it detects:

- a missing prerequisite file;
- an unreadable or wrong-format checkpoint;
- an additive checkpoint paired with the wrong initialization digest;
- a best checkpoint and latest checkpoint from different additive runs;
- a full-profile additive run that has not met its convergence criterion;
- smoke/full profile mismatch;
- a rank report built from a different additive iteration;
- a candidate with no positive active interaction rank; or
- a final candidate that has not completed the household-size stage;
- a missing, truncated, or invalid evaluation JSON/assignment archive; or
- a full-profile likelihood evaluation whose numerical certification did not pass.

These errors should be fixed by restoring the matching artifact set or choosing an earlier
`--start-at` stage. Do not rename an unrelated checkpoint to the expected filename.

The household-size cross-fit itself no longer strands recovery. If no incremental
post-interaction size gain is supported, the pipeline uses the exact nested zero correction
or a safety-only downward projection and records the decision in
`artifacts/candidate_rank1.json`. Final likelihood and population safety still fail closed.

## 8. Inspection before spending compute

Print the command suffix without executing model stages:

```bash
python scripts/run_pipeline.py --profile full --start-at interaction --dry-run
```

Display CLI help:

```bash
python scripts/run_pipeline.py --help
```

Follow the combined log:

```bash
tail -f artifacts/pipeline.log
```

Follow only additive optimization:

```bash
tail -f out/v3_pipeline_additive.log
```

Always append with `tee -a` when resurrecting a run, so the new stage output does not
overwrite the earlier execution record.

## 9. Recovery decision table

| Last trustworthy state | Correct action |
|---|---|
| Only derived data completed | `--start-at initialize` |
| Initialization completed | `--start-at additive` |
| Additive process interrupted below its ceiling | `--start-at additive --resume-additive out/v3_pipeline_additive.pt` |
| Additive convergence completed | `--start-at rank` |
| Rank report completed | `--start-at interaction` |
| Final `candidate_rank1.pt` completed | `--start-at evaluation` |
| All evaluation files completed | `--start-at certification` |
| Unsure whether a stage completed | begin at that stage, not the following stage |

This last rule is conservative but inexpensive relative to accepting an artifact whose
gate never completed.
