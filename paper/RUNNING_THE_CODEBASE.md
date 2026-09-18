# Running the codebase on a new machine

Date: 2026-09-18

Everything needed to go from a bare machine to fitted models, audits and a running API.
Commands are run from the repository root unless a step says otherwise. Paths in `data/`,
`basket_input/`, `artifacts/`, `out/` and `reports/` are git-ignored: nothing here
redistributes licensed data.

Companion documents:
[README.md](../README.md) (Dunnhumby route in depth),
[CANONICAL_INPUT_CONTRACT.md](CANONICAL_INPUT_CONTRACT.md) (adding a dataset),
[PARTITION_OPTIONS_AND_PARKED_WORK.md](PARTITION_OPTIONS_AND_PARKED_WORK.md) (grouping options),
[RETAIL_APPLICATION_API.md](RETAIL_APPLICATION_API.md) (API reference),
[SYNTHETIC_RETAIL_API_APPLICATIONS.md](SYNTHETIC_RETAIL_API_APPLICATIONS.md) (worked API examples).

---

## 1. Machine requirements

| Item | Requirement |
|---|---|
| OS | macOS or Linux (Windows: use WSL) |
| Python | 3.11+; developed and certified on 3.13.4 |
| Compiler | a C++ toolchain matching the Python build (Xcode command line tools, or `build-essential`) |
| RAM | 12 GB for Dunnhumby full-population stages; 8 GB is enough for ERIM and the synthetic world |
| Disk | 10 GB free: raw data about 1 GB, model bundles 100–700 MB each, `artifacts/` grows to several GB |
| CPU | any x86-64 or Apple Silicon; the exact likelihood is **CPU-only by design** |
| Network | only to download the source data |

GPUs are not used. The dominant computation is an exact float64 dynamic program with a
custom adjoint; `--device cuda` fails loudly rather than silently degrading the estimator.

Check a new machine before any data or fitting:

```bash
python scripts/inspect_hardware.py                       # prints the report
python scripts/inspect_hardware.py --output hardware.json  # and saves it
```

Every pipeline run writes its own copy to `<run-dir>/artifacts/runtime_capabilities.json`.

## 2. Install

```bash
git clone https://github.com/AjitKumar01/energy_basket_intelligence.git
cd energy_basket_intelligence
git checkout retail-application-api          # this branch
python --version                             # must be 3.11+
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
pytest -q                                    # 258 tests, about 35 s, no data needed
```

Pinned versions are the certification environment. PyTorch C++ extensions are ABI-specific,
so changing a pin requires a fresh smoke run.

`pytest.ini` puts `.`, `scripts` and `scripts/version4` on the path, which is why standalone
scripts import `data`, `features`, `ragged` and friends directly.

## 3. Choose a dataset

| Dataset | What it is | Data needed | Fit time (full pipeline) |
|---|---|---|---|
| **Synthetic** | generated world with known truth; the only dataset where answers can be scored against truth | none: generate it | about 8 min |
| **ERIM** | 8 tracked grocery categories, 464 products, 3,862 households, public archives | download, about 600 MB | about 8–11 min, plus a 4-stage price route of about 50 min |
| **Dunnhumby** | The Complete Journey, 5,455 products | licensed CSVs you supply | about 45 min |

Start with the synthetic world: it needs no downloads and exercises every capability.

---

## 4. Synthetic world, end to end

```bash
# 1. generate the world (canonical format + truth.npz), about 2 min
python -u scripts/synthetic/generate_canonical_world.py \
    --output data/synthetic_capability_world

# 2. build a model bundle (category grouping)
python -u scripts/prepare_model_bundle.py \
    --dataset-config configs/datasets/synthetic_capability_world_category.json

# 3. fit and audit, about 8 min
python -u scripts/run_pipeline.py \
    --model-data-root data/synthetic_capability_world/model_input_category \
    --run-dir artifacts/synthetic_capability_test_category/full --profile full --threads 4

# 4. score every capability against the truth, about 30 s
python -u scripts/synthetic/evaluate_capabilities.py \
    --bundle data/synthetic_capability_world/model_input_category \
    --run artifacts/synthetic_capability_test_category/full \
    --output artifacts/synthetic_capability_test_category/capability_report.json
```

Step 4 also writes `capability_verdicts.json` next to the checkpoint, which is what lets the
API serve price scenarios for this deployment (§8).

Generator options: `--households --stores --products-per-category --weeks --train-weeks
--validation-weeks --visit-probability --nmax --seed`. Changing any of them changes the
truth, so regenerate the bundle and refit.

## 5. ERIM, end to end

```bash
# 1. download the official archives (about 600 MB, 8 categories)
python -u scripts/data/fetch_erim_categories.py --output-dir data/erim_basket

# 2. build the canonical directory (hashes are verified against source_manifest.json)
python -u scripts/build_erim_basket_dataset.py --config configs/erim_basket.json

# 3. optional: product types for the catalogue-hierarchy grouping
python -u scripts/build_erim_product_types.py

# 4. build a model bundle (category grouping is the ERIM standard)
python -u scripts/prepare_model_bundle.py \
    --dataset-config configs/datasets/erim_availability_category.json

# 5a. quick route: fit and audit only, about 9 min
python -u scripts/run_pipeline.py \
    --model-data-root data/erim_basket/model_input_availability_category \
    --run-dir artifacts/erim_category_refit/full --profile full --threads 4

# 5b. full route used for every published ERIM result, about 50 min
scripts/run_erim_partition_refit.sh category 4
```

The 4-stage route (5b) is what the ERIM reports use, and it is what `--price-coefficients`
requires:

1. **price-zero parent** — `run_pipeline.py --stop-after additive`;
2. **joint price utility** — `fit_joint_price_utility.py` (the long stage, about 27 min);
3. **price alignment audit** — `audit_joint_price_alignment.py` (placebo checks);
4. **full pipeline** — `run_pipeline.py --price-coefficients …`.

Available ERIM configs: `erim_availability` (co-purchase groups), `erim_availability_category`
(**standard**), `erim_availability_catalogue` (category × product type),
`erim_availability_evidence` and `_evidence_residual` (groups learned from co-purchase
evidence; never fitted, see the parked-work doc).

## 6. Dunnhumby, end to end

Dunnhumby predates the canonical contract and keeps its own raw route with the model data at
the repository root (`basket_input/`, `data/`).

```bash
export NF_RAW_DIR="/absolute/path/to/dunnhumby_The-Complete-Journey CSV/"   # optional if the
# CSVs sit in dunnhumby_The-Complete-Journey/dunnhumby_The-Complete-Journey CSV/
python scripts/run_pipeline.py --from-raw --dry-run        # command graph, no compute
python scripts/run_pipeline.py --from-raw --profile smoke  # software path, undertrained
python scripts/run_pipeline.py --from-raw --profile full 2>&1 | tee artifacts/pipeline.log
```

Required files: `transaction_data.csv`, `product.csv`, `causal_data.csv`. Their SHA-256
digests are locked; a modified or partial release fails closed. Smoke-profile numbers are
never research results.

To try another product grouping on Dunnhumby, build the partition into a **new** bundle
directory and never overwrite `basket_input/items_affinity.parquet`, which existing
checkpoints depend on:

```bash
cd scripts/version4
python build_catalogue_partition.py --basket-input ../../basket_input \
    --rule finest_catalogue_level --validate-only          # inspect first
```

---

## 7. What a pipeline run produces

Inside `--run-dir` (or the repository root for the Dunnhumby default):

| Path | Contents |
|---|---|
| `invocation_<timestamp>/manifest.json` | every stage, its command, exit code and runtime |
| `invocation_<timestamp>/NN_<stage>.log` | per-stage logs |
| `out/v3_pipeline_additive_best.pt` | fitted additive parent |
| `artifacts/candidate_rank1.pt` | final checkpoint the API and audits use |
| `artifacts/interaction_basis_rank*.json` | rank selection, including rejected ranks |
| `reports/likelihood_{validation,test}.json` | held-out likelihood, paired |
| `reports/recommendation.json` | MRR and recall |
| `reports/generation_counterfactual.json` | generation checks and price scenarios |
| `reports/customer_segments.json` | segments |
| `reports/population_size.json` | population and tail safety |
| `reports/segment_promotion_mdp.json` | promotion policy scenario |
| `reports/interaction_embedding_audit.{json,md}` | learned pair structure |

**Fail-closed behaviour.** In the full profile a failed gate exits nonzero and keeps the
artifacts for diagnosis. A zero exit for the smoke profile means the code path ran, nothing
more.

**Resuming.** `--start-at {data,initialize,additive,rank,interaction,evaluation,certification}`
reuses validated earlier artifacts; `--resume-additive <checkpoint>` continues an interrupted
optimizer; `--rebuild-interaction-bank` forces the draw cache to be resampled.

**Source freeze.** A run hashes `scripts/**/*.py`, `*.cpp` and `tests/*.py` at startup and
aborts if any of them changes while it runs. Never edit code during a run; documentation and
`artifacts/` are fine.

## 8. Serving the retail API

Defaults point at the ERIM category checkpoint. Start it:

```bash
python scripts/run_retail_api.py --host 127.0.0.1 --port 8000     # one worker
```

Serve another deployment with environment variables:

```bash
RETAIL_API_DATA_ROOT=data/synthetic_capability_world/model_input_category \
RETAIL_API_CHECKPOINT=artifacts/synthetic_capability_test_category/full/artifacts/candidate_rank1.pt \
RETAIL_API_COMPLETION_AUDIT=artifacts/synthetic_capability_test_category/retail_application/real_basket_completion_corrected.json \
RETAIL_API_SEGMENT_REPORT=artifacts/synthetic_capability_test_category/full/reports/customer_segments.json \
python scripts/run_retail_api.py --port 8000
```

| Variable | Meaning |
|---|---|
| `RETAIL_API_DATA_ROOT` | served model-data bundle |
| `RETAIL_API_CHECKPOINT` | checkpoint; must match the bundle's fingerprint |
| `RETAIL_API_COMPLETION_AUDIT` | passing completion audit for that checkpoint |
| `RETAIL_API_SEGMENT_REPORT` | segment report from the same run |
| `RETAIL_API_CAPABILITY_VERDICTS` | optional; defaults to `capability_verdicts.json` beside the checkpoint |
| `RETAIL_API_THREADS` | CPU threads (default 4) |

**A new checkpoint needs a completion audit before it can be served:**

```bash
cd scripts/version4
ENERGY_MODEL_DATA_ROOT=<bundle> V3_AFFINITY=1 python -u audit_basket_completion_corrected.py \
    --checkpoint <run>/artifacts/candidate_rank1.pt \
    --output <run>/retail_application/real_basket_completion_corrected.json \
    --threads 4 --level-offset 3
```

Try `--level-offset 2` first; if its numerical gates fail, use 3 (both ERIM and the synthetic
world needed 3). The service reads the certified levels from the audit.

**Endpoints:** `GET /live`, `GET /ready`, `GET /v1/capabilities`, `GET /v1/products/search`,
`POST /v1/baskets/complete`, `POST /v1/baskets/price_scenario`,
`GET /v1/households/{id}/segment`, plus `/docs` and `/openapi.json`.

**Decision capabilities are refused by default.** Price scenarios answer only where a verdict
file claims causal price optimization with evidence bound to that checkpoint and dataset;
otherwise the endpoint returns HTTP 403. ERIM has no verdict file by design.

Exercise a deployment:

```bash
python scripts/smoke_retail_api.py --output artifacts/<run>/retail_application/retail_api_smoke.json
python scripts/benchmark_retail_api.py --base-url http://127.0.0.1:8011 \
    --smoke-artifact artifacts/<run>/retail_application/retail_api_smoke.json \
    --output artifacts/<run>/retail_application/retail_api_latency.json
python scripts/verification/api/synthetic_api_applications.py --cases 200 --households 600
```

## 9. Verification, audits and research scripts

**Tests**

```bash
pytest -q                                 # all 258
pytest -q tests/test_retail_api.py        # API contracts
pytest -q tests/test_canonical_contract.py tests/test_catalogue_partition.py \
         tests/test_evidence_partition.py # input contract and groupings
```

**Partition diagnostics** (model-free, minutes)

```bash
python scripts/verification/partition/check_evidence_partition.py \
    --basket-input <bundle>/basket_input --output artifacts/<name>.json
cd scripts/version4 && python build_catalogue_partition.py \
    --basket-input <bundle>/basket_input --rule category --validate-only
```

**ERIM research checks**

```bash
python scripts/verification/nested_rho/substructure_check.py   # sub-groups inside categories
python scripts/verification/nested_rho/promotion_check.py      # promotions and loyalty
python scripts/verification/nested_rho/type_check.py           # product types
python scripts/run_erim_coupon_experiment.py --help            # the randomized coupon lottery
```

**Nested-model exactness prototypes** (no pipeline code)

```bash
python scripts/verification/nested_rho/verify_exactness.py
python scripts/verification/nested_rho/verify_sampler.py
python scripts/verification/nested_rho/verify_recovery.py --per-context 150
```

**Cart-conditional and price counterfactual queries offline**

```bash
cd scripts/version4
ENERGY_MODEL_DATA_ROOT=<bundle> python basket_counterfactual_query.py \
    --checkpoint <run>/artifacts/candidate_rank1.pt --query <query.json> --output <out.json>
```

## 10. Adding a new dataset

1. **Write an adapter** that produces the canonical directory: `transactions.parquet`,
   `products.parquet`, `store_week_prices.parquet`, `promotions.parquet`, optional
   `shopping_opportunities.parquet`, and `build_audit.json`. Columns and invariants are in
   [CANONICAL_INPUT_CONTRACT.md](CANONICAL_INPUT_CONTRACT.md). The adapter is the only place
   dataset-specific parsing belongs.
2. **Write a dataset config** in `configs/datasets/`: paths, `price_basis`, promotion feature,
   availability rule, grouping option and its settings, metadata defaults.
3. **Build the bundle:** `python scripts/prepare_model_bundle.py --dataset-config …`. It
   validates the contract, builds model files, writes the grouping, builds the index and the
   fingerprint.
4. **Fit:** `python scripts/run_pipeline.py --model-data-root … --run-dir … --profile full`.
5. **Serve, optionally:** run the completion audit, then point the API variables at the new
   run.

## 11. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `artifact data fingerprint differs from the current audited dataset` | the checkpoint was fitted on a different bundle. Set `ENERGY_MODEL_DATA_ROOT` to the bundle that produced it, or refit. |
| `source changed during … pipeline` | a `.py` under `scripts/` or `tests/` was edited mid-run. Commit first, discard the aborted run, rerun. |
| `<root> already holds a bundle; pass --force to rebuild it` | `prepare_model_bundle.py` will not overwrite silently. Add `--force`, or choose a new `model_data_root`. |
| completion audit `status: failed` at offset 2 | use `--level-offset 3`; the service reads certified levels from the audit. |
| `adjacent quadrature precision gate failed` (HTTP 503) | the two quadrature rules disagree beyond tolerance for that request; the service refuses rather than answering. Re-audit the checkpoint at a higher offset. |
| `this deployment does not claim causal_price_optimization` (HTTP 403) | no verdict file, or it does not claim that capability. See §8. |
| `products are not offered in this store and week` | the price scenario named a product outside that context's assortment. |
| `zsh: no matches found: …*.log` | zsh does not expand a glob with no matches. Quote it or use `setopt NULL_GLOB`. |
| Native extension build failure | install the compiler toolchain; the pipeline builds `setup_poly_degree_native.py` on every run. |
| Memory pressure in full-population stages | lower `--threads`, close other work, or run the smoke profile to verify the path first. |
| `pytest` import errors when running a script directly | run from the repository root, or rely on `pytest.ini`'s path setup. |

## 12. Conventions worth knowing

- **The model is frozen.** Product groupings are inputs; changing the model's form (for
  example nested groups) is a separate decision. See
  [PARTITION_OPTIONS_AND_PARKED_WORK.md](PARTITION_OPTIONS_AND_PARKED_WORK.md).
- **Evidence gates everything.** Audits, capability verdicts and numerical certificates all
  fail closed. Nothing is reported as validated without its evidence.
- **Test data never chooses anything.** Model and setting selection use validation data only.
- **Bundles are immutable identities.** Any change to a bundle changes its fingerprint and
  detaches every checkpoint fitted on it.
- **Reproducing a published number:** the reports under `paper/` name their run directory,
  bundle and fingerprint; rebuild that bundle from its config and rerun that run directory's
  commands.
