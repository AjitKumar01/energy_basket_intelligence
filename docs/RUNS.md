# Running the pipeline

## 1. One command, one pass

`scripts/run_pipeline.py` runs every stage in order, from a fresh initialization to
certification, in a new run directory:

```bash
python scripts/run_pipeline.py --model-data-root <bundle> --run-dir <run> --profile full
```

There is no resuming: each run starts from scratch, so a run directory always holds the
outputs of a single, complete invocation. `--run-dir` must not exist yet.

`--stop-after STAGE` ends the run after that stage (for example `--stop-after rank`), and
`--dry-run` prints every stage command without executing any of them:

```bash
python scripts/run_pipeline.py --model-data-root <bundle> --run-dir <run> --dry-run
python scripts/run_pipeline.py --help
```

## 2. Stages and their outputs

Every output lives under the run directory:

| Stage | Work performed | Main output |
|---|---|---|
| `data` | check the model-data bundle and the environment | `artifacts/runtime_capabilities.json` |
| `initialize` | fresh Version-4 parameter initialization | `artifacts/initialization.pt` |
| `additive` | price evidence, then exact complete-support additive maximum likelihood | `out/v3_pipeline_additive_best.pt` |
| `rank` | split-half interaction-rank audit and spectral basis | `artifacts/interaction_basis_rank8.{npz,json}` (full profile) |
| `interaction` | constrained interaction solve, then household-size recalibration | `artifacts/candidate_rank1.{pt,json}` |
| `refinement` | optional (`--joint-refinement`) joint refinement with its acceptance gates | `artifacts/candidate_refined.pt` if accepted; `reports/joint_refinement.json` |
| `evaluation` | validation/test likelihood, recommendation, generation, segmentation, embedding audit | files under `reports/` plus `artifacts/customer_segments.npz` |
| `certification` | complete-population size audit, size-phase diagnostic, promotion MDP | `reports/population_size.json`, `reports/segment_promotion_mdp.json` |

With `--joint-refinement`, evaluation and certification use `candidate_refined.pt` when
`reports/joint_refinement.json` records an accepted decision, and `candidate_rank1.pt`
otherwise.

A stage whose statistical gate fails stops the run with a nonzero exit code; the stage's report
says which gate failed. Smoke runs (`--profile smoke`) relax the gates and use a rank-4 basis.

## 3. Logs

Each run writes `<run>/invocation_<UTC time>/` with:

- `pipeline.log`: everything the driver and its stages printed;
- one log per stage command (`01_initialize_version4.log`, ...);
- `manifest.json`: the configuration, and each stage command's status, exit code and run time.

Follow a run with `tail -f <run>/invocation_*/pipeline.log`.
