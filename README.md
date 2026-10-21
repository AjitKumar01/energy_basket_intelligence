# Version-4 energy basket model: the pipeline

One joint probability model of shopping baskets with an exact normaliser, fitted to transaction
data, used for basket likelihood, realistic basket generation and price and promotion
counterfactuals (own product and substitutes). SHOPPER, a sequential basket model, is included
as a baseline and scored the same way.

| Document | Content |
|---|---|
| [`docs/THEORY.md`](docs/THEORY.md) | the model, the exact normaliser, interactions through a Gaussian latent, exact sampling and generation, counterfactuals, estimation |
| [`docs/STRESS_WORLD.md`](docs/STRESS_WORLD.md) | the synthetic stress world: how its baskets are generated, its causal price truth, what it tests |
| [`docs/PIPELINE.md`](docs/PIPELINE.md) | the stages, which parameters each learns, what is decided on validation data, checks, results, known weaknesses |
| [`docs/SHOPPER_BASELINE_PROTOCOL.md`](docs/SHOPPER_BASELINE_PROTOCOL.md) | the locked protocol for the SHOPPER comparison |

| Folder | Content |
|---|---|
| `v4/` | the model: data bundle, features, batches, utilities and energy, exact normaliser, sampler |
| `v4/native/` | C++ kernels, built on first use into `build/native/` |
| `scripts/` | one script per stage: partition, discount panel, initialize, base-model fit, Version-4 fit, evaluate |
| `checks/` | the real-data discount and promotion check; the support-cap check |
| `shopper/` | SHOPPER (PyTorch reimplementation), its exactness checks and evaluator |
| `synthetic/` | the stress-world generator |
| `run/` | the recipes below |
| `tests/` | exact tests against enumeration |

## 1. Setup (once per machine)

1. Python 3.11+ and a C++ compiler (clang++ or g++; `CXX` chooses one).
2. `pip install -r requirements.txt`
3. `python3 -m pytest -q`: compiles the two native kernels and checks the exact law against
   enumeration of every basket of a tiny world. It needs no data and takes about 20 seconds
   the first time.

Create the output folders once: `mkdir -p runs/stress runs/dunnhumby runs/shopper`.
Every recipe below runs detached (`nohup ... &`) and writes a log per stage. Use at most 4
threads per job (`THREADS`, default 4) and keep the total below the machine's core count.

## 2. Data

The pipeline reads a **model-data bundle**: a folder with `basket_input/` and `data/` (contents in
`docs/PIPELINE.md` §2). Checkpoints record the bundle's fingerprint and load only with it.

- **Dunnhumby.** Copy the prepared bundle, the folder that holds `basket_input/` and `data/`
  (in this project, `energy_basket_intelligence/`). Building it from the raw Dunnhumby files is
  done outside this repository.
- **Stress world.** Either copy `data/stress_world/` (the world plus its bundle,
  `model_input_category/`, 26 MB), or generate it:
  ```bash
  python3 synthetic/generate_stress_world.py --output data/stress_world        # about 80 s
  ```
  and then build the bundle with the branch `v4-training`:
  `scripts/prepare_model_bundle.py --config configs/datasets/stress_world_category.json`.
  That converter is not on this branch yet.
- **Reference trips** for test-week scoring: an `.npz` with `trips` and `household`, and any
  other model's per-trip log-likelihoods to compare with. Stress world:
  `artifacts/stress_clean_confirm/reports/likelihood_test_paired_per_trip.npz` (4,096 test
  trips). Dunnhumby: `artifacts/corrected_complete_rank5_20260913/reports/likelihood_test_per_trip.npz`.

## 3. Stress world

```bash
export WORLD=/path/to/data/stress_world        # holds model_input_category/, truth files
export REFERENCE=/path/to/likelihood_test_paired_per_trip.npz
```

**Our model**: every stage, then both Version-4 variants scored on the test weeks with the
32 price scenarios:

```bash
nohup run/stress_world.sh > runs/stress/pipeline.out 2>&1 &
```

| Stage | What runs | Output in `runs/stress/` |
|---|---|---|
| 2 | `scripts/initialize.py` | `initialization.pt` |
| 3 | `scripts/fit_additive.py` (base model) | `additive/v3_pipeline_additive_best.pt` |
| 4 | `scripts/fit_interactions.py --recency [--m4c]` | `v4_rec.pt`, `v4_m4c.pt` and their `.log` |
| 5 | `scripts/evaluate.py --world $WORLD` | `v4_*.eval.json`, `v4_*_3seed_auc.json`, `v4_*.per_trip.npz` |

`FROM=4` starts at stage 4, and so on. The stress world has no regular-price series, so it has
no discount split.

**SHOPPER**: fit K = 8, 16, 32, choose on validation, then score the chosen one with
log P(S) (8,192 orderings), realism and the price scenarios:

```bash
DATA=$WORLD/model_input_category CONTRACT=chain_and_store OUT=runs/shopper/stress \
  nohup run/shopper.sh > runs/shopper/stress.out 2>&1 &
# then, with K chosen from the printed validation scores:
DATA=$WORLD/model_input_category CONTRACT=chain_and_store OUT=runs/shopper/stress \
  CHOSEN=8 REFERENCE=$REFERENCE WORLD=$WORLD run/shopper.sh
```

**What to read:** `likelihood.mean_ll` and the `vs_*` paired differences;
`classifier.*_auc_mean` (0.5 is perfect); `price.*.correlation` and `mean_model` against
`mean_truth`.

## 4. Dunnhumby

```bash
export DATA=/path/to/dunnhumby/bundle          # holds basket_input/ and data/
```

**Our model, everything from the bundle** (about 6 hours for the base model plus 8–12 hours for
Version 4):

```bash
nohup run/dunnhumby.sh > runs/dunnhumby/pipeline.out 2>&1 &
```

| Stage | What runs | Output in `runs/dunnhumby/` |
|---|---|---|
| 2 | `scripts/build_discount_panel.py`, `scripts/initialize.py` | `log_discount.npy`, `initialization.pt` |
| 3 | `scripts/fit_additive.py` with chain-only prices and the discount split | `additive/` |
| 3c | `checks/discount_promotion.py events`, then `model` on the base model | `discount_events/`, `additive/discount_check.json` |
| 4 | `scripts/fit_interactions.py` (recency; recency + M4c), then the discount check and the support-cap check on each | `v4_rec.pt`, `v4_m4c.pt`, `v4_*_discount_check.json`, `v4_*_support_cap.json` |
| 5 | test weeks, once, for the chosen variant: `CHOSEN=m4c REFERENCE=... FROM=5 run/dunnhumby.sh` | `v4_m4c.eval.json` |

**Version 4 only, from an existing base model** (the update-matched schedule, at most 40
products per category):

```bash
INIT=/path/to/initialization.pt ADDITIVE=/path/to/v3_pipeline_additive_best.pt \
  nohup run/dunnhumby_step4.sh > runs/dunnhumby/step4.out 2>&1 &
```

**SHOPPER**:

```bash
CONTRACT=chain_product_week_only OUT=runs/shopper/dunnhumby \
  nohup run/shopper.sh > runs/shopper/dunnhumby.out 2>&1 &
CONTRACT=chain_product_week_only OUT=runs/shopper/dunnhumby CHOSEN=8 \
  REFERENCE=/path/to/likelihood_test_per_trip.npz run/shopper.sh
```

Real data has no causal truth, so SHOPPER is scored on likelihood and realism. Its
discount-depth slope (the protocol's third measure) is not implemented yet.

**What to read:** the Version-4 `.log` rows (`val_ll`, `val_ess`, `chain_mean_size` against
`data_mean_size`); in `discount_check.json`, `depth_slope` against `empirical_effects`
(train / validation / test) and `promotion_on_vs_off`.

## 5. Watching runs

```bash
tail -f runs/dunnhumby/v4_rec.log runs/dunnhumby/v4_m4c.log     # one JSON row per epoch
tail -f runs/dunnhumby/4_v4_rec.out                               # everything, including errors
ps -eo pid,etime,rss,pcpu,command | grep "[f]it_"                 # running jobs, memory, CPU
```

This branch (`v4-pipeline`) holds only the current pipeline. The research history, diagnostics
and earlier pipelines are on `v4-training`.
