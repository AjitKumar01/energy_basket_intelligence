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
| `checks/` | checks of a fitted model: discounts and promotions, support cap, recommendation, learned embeddings, customer segments, the price halo, basket size by period |
| `sim/` | a Gymnasium environment for promotion policies on a fitted model (see `docs/PIPELINE.md` §7 before using it) |
| `shopper/` | SHOPPER (PyTorch reimplementation), its exactness checks and evaluator |
| `data_prep/` | building bundles: Dunnhumby from its raw CSVs; any dataset from a canonical directory |
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
`docs/PIPELINE.md` §2). Both bundles are built from scratch by this repository. Checkpoints
record the bundle's fingerprint and load only with it.

- **Dunnhumby**, from the raw "The Complete Journey" CSVs (about 5 minutes):
  ```bash
  RAW="/path/to/dunnhumby_The-Complete-Journey CSV" OUT=data/dunnhumby run/prepare_dunnhumby.sh
  ```
  It needs `transaction_data.csv`, `product.csv` and `causal_data.csv`. Four stages (base
  tables, bundle, promotions, audit), then the co-purchase categories and the fingerprint
  (`data_prep/dunnhumby/`). The result is byte-identical to the bundle every existing
  Dunnhumby checkpoint was fitted on, fingerprint included.
- **Stress world**, generated (about 2 minutes):
  ```bash
  OUT=data/stress_world run/prepare_stress_world.sh
  ```
  This runs `synthetic/generate_stress_world.py`, then the general canonical converter
  `data_prep/canonical/prepare_bundle.py` with `data_prep/configs/stress_world.json`. Any other
  dataset enters the same way: write its canonical directory (`data_prep/canonical/contract.py`)
  and a config.
- **Reference trips** for test-week scoring: the recipes make them with
  `scripts/reference_trips.py` (4,096 test trips in a seeded order, with the base model's exact
  per-trip log-likelihood for paired comparisons). These are the same trips every earlier
  result used.

## 3. Stress world

```bash
export WORLD=$PWD/data/stress_world            # from run/prepare_stress_world.sh
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
| 5 | `scripts/reference_trips.py`, then `scripts/evaluate.py --world $WORLD` | `reference_test_trips.npz`, `v4_*.eval.json`, `v4_*_3seed_auc.json`, `v4_*.per_trip.npz` |

`FROM=4` starts at stage 4, and so on. The stress world has no regular-price series, so it has
no discount split.

**SHOPPER**: fit K = 8, 16, 32, choose on validation, then score the chosen one with
log P(S) (8,192 orderings), realism and the price scenarios:

```bash
DATA=$WORLD/model_input_category CONTRACT=chain_and_store OUT=runs/shopper/stress \
  nohup run/shopper.sh > runs/shopper/stress.out 2>&1 &
# then, with K chosen from the printed validation scores:
DATA=$WORLD/model_input_category CONTRACT=chain_and_store OUT=runs/shopper/stress \
  CHOSEN=8 REFERENCE=runs/stress/reference_test_trips.npz WORLD=$WORLD run/shopper.sh
```

**What to read:** `likelihood.mean_ll` and the `vs_*` paired differences;
`classifier.*_auc_mean` (0.5 is perfect); `price.*.correlation` and `mean_model` against
`mean_truth`.

## 4. Dunnhumby

```bash
export DATA=$PWD/data/dunnhumby                # from run/prepare_dunnhumby.sh
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
| 5 | test weeks, once, for the chosen variant: `CHOSEN=rec FROM=5 run/dunnhumby.sh` | `reference_test_trips.npz`, `v4_rec.eval.json` |

**The current model: commodity partition** (one category per Dunnhumby commodity, within-
category competition, common display and mailer slopes; `docs/PIPELINE.md` §7.0). From a
Dunnhumby run (its `initialization.pt` and base model), build the commodity bundle and base
model, then Version 4 and its checks (discount, basket size by period, real halo):

```bash
SRC=data/dunnhumby DEST=runs/bundle_commodity run/commodity_bundle.sh          # the partition
PANEL=runs/dunnhumby/log_discount.npy WARM=runs/dunnhumby/additive/v3_pipeline_additive_best.pt \
  nohup run/commodity.sh > runs/commodity/pipeline.out 2>&1 &                  # base model
nohup run/commodity_comp.sh > runs/commodity/comp_promo.out 2>&1 &             # Version 4 + checks
```

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
  REFERENCE=runs/dunnhumby/reference_test_trips.npz run/shopper.sh
```

Real data has no causal truth, so SHOPPER is scored on likelihood and realism. Its
discount-depth slope (the protocol's third measure) is not implemented yet.

**What to read:** the Version-4 `.log` rows (`val_ll`, `val_ess`, `chain_mean_size` against
`data_mean_size`); in `discount_check.json`, `depth_slope` against `empirical_effects`
(train / validation / test) and `promotion_on_vs_off`.

**Further checks on a fitted model** (each writes one JSON file; add `--initialization` for old
checkpoints that do not record their parents):

```bash
CK=runs/dunnhumby/v4_rec.pt
DEMO="/path/to/raw/hh_demographic.csv"
python3 checks/recommendation.py --checkpoint $CK --split validation --out runs/dunnhumby/v4_rec_recommendation.json
python3 checks/embeddings.py --checkpoint $CK --demographics "$DEMO" --out runs/dunnhumby/v4_rec_embeddings.json
python3 checks/segments.py   --checkpoint $CK --demographics "$DEMO" --out runs/dunnhumby/v4_rec_segments.json
python3 checks/size_by_period.py --checkpoint $CK --out runs/dunnhumby/v4_rec_size_by_period.json
python3 sim/run_policies.py  --checkpoint runs/dunnhumby/additive/v3_pipeline_additive_best.pt \
  --initialization runs/dunnhumby/initialization.pt --simulator base --out runs/dunnhumby/sim/base_policies.json
```

On a synthetic world, the price halo (does a one-product price change move the rest of the basket?):

```bash
ENERGY_MODEL_DATA_ROOT=$WORLD/model_input_category python3 checks/halo.py \
  --checkpoint runs/stress/v4_rec.pt --world $WORLD --out runs/stress/halo_v4_rec.json
```

## 5. Watching runs

```bash
tail -f runs/dunnhumby/v4_rec.log runs/dunnhumby/v4_m4c.log     # one JSON row per epoch
tail -f runs/dunnhumby/4_v4_rec.out                               # everything, including errors
ps -eo pid,etime,rss,pcpu,command | grep "[f]it_"                 # running jobs, memory, CPU
```

This branch is self-contained: from raw data to fitted, checked models, it needs nothing
outside it. `bench/` times the normaliser and sampler and checks that a change keeps the law
(`bench/README.md`). It was distilled from the research branch `v4-training`, which keeps the
full history of experiments and diagnostics (see `docs/PIPELINE.md` §9).
