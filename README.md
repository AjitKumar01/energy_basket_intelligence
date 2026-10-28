# Version-4 basket model, mixture form

One probability model of shopping baskets, fitted to transaction data, for basket likelihood,
realistic basket generation and price and promotion counterfactuals. Each trip has a taste
z ~ N(0, I); given z, the basket law is exact (a dynamic program over categories and sizes),
and the probability of a basket is its average over the taste prior. This branch holds only
this model: no Gibbs chains, pseudo-likelihood or the earlier energy-form training.

| Document | Content |
|---|---|
| [`docs/why_mixture.pdf`](docs/why_mixture.pdf) | the earlier form, where it fails (a hidden taste peak with giant baskets), and the mixture form from prior to posterior |
| [`docs/mixture_theory.pdf`](docs/mixture_theory.pdf) | the model in full: prior, law given z, exact normaliser, hierarchical sampling, likelihood estimate, training |

| Folder | Content |
|---|---|
| `v4/` | the model: data bundle, features, batches, utilities and energy, exact normaliser, sampler given z, mixture likelihood and generation, training blocks and priors |
| `v4/native/` | C++ kernels, built on first use into `build/native/` (or `$V4_BUILD_DIR`) |
| `scripts/` | partitions, discount panel, initialize, base-model fit (`fit_additive.py`), mixture fit (`fit_mixture.py`) |
| `data_prep/` | building bundles: Dunnhumby from its raw CSVs; any dataset from a canonical directory |
| `run/` | the recipes below |
| `tests/` | exact tests against enumeration of every basket of a tiny world |

## 1. Setup (once per machine)

1. Python 3.11+ and a C++ compiler (clang++ or g++; `CXX` chooses one).
2. `pip install -r requirements.txt`
3. `python3 -m pytest -q`: compiles the native kernels and checks the law against enumeration
   (56 tests, about 30 seconds the first time).

Every recipe runs detached and writes a log per stage. Use at most 4 threads per job.

## 2. Data

```bash
RAW="/path/to/dunnhumby_The-Complete-Journey CSV" OUT=data/dunnhumby run/prepare_dunnhumby.sh
SRC=data/dunnhumby DEST=runs/bundle_commodity run/commodity_bundle.sh     # one category per commodity
```

Checkpoints record the bundle's fingerprint and load only with it. Any other dataset enters
through a canonical directory (`data_prep/canonical/contract.py`) and
`data_prep/canonical/prepare_bundle.py`.

## 3. Fit

```bash
DATA=runs/bundle_commodity OUT=runs/commodity nohup run/fit.sh > runs/commodity/fit.out 2>&1 &
```

| Stage | What runs | Output in `$OUT` |
|---|---|---|
| 2 | `scripts/build_discount_panel.py`, `scripts/initialize.py` (skipped if present) | `log_discount.npy`, `initialization.pt` |
| 3 | `scripts/fit_additive.py`: the exact base model, phi = 0 (skipped if present) | `additive/v3_pipeline_additive_best.pt` |
| 4 | `scripts/fit_mixture.py`: every Version-4 block on the base model, trained on the mixture likelihood | `mixture.pt`, `mixture_best.pt`, `mixture.log` |

`FROM=4` starts at stage 4 (with stages 2 and 3 already in `$OUT`). `WARM=<additive fit>`
warm-starts the base model. A stopped mixture fit continues from its best checkpoint with
`scripts/fit_mixture.py --init <out>_best.pt --start-step <its step>` and the same seed and epochs.

**What to read in `mixture.log`:** `val_ll` (held-out log-likelihood per trip, a lower bound in
expectation), `val_ess_mean`, `gen_size_mean` and `gen_size_q99` against `data_size_mean` and
`data_size_q99` (generated against real basket sizes), `whiten_trace_over_rank` (near 1 when the
taste scale is right), `comp_iv` rows (the competition weights).

```bash
tail -f runs/commodity/mixture.log                       # one JSON row per validation / every 30 steps
tail -f runs/commodity/4_mixture.out                     # everything, including errors
```

## 4. Synthetic world

A synthetic world with a known true law is fitted the same way: point `ENERGY_MODEL_DATA_ROOT`
at its bundle and give its initialization and base model. The generator is not on this branch;
the world used so far is `disc_c48_s3` (48 categories, 522 products, true within-category
competition 0.05), built on branch `v4-scale`, with its base model fitted for baskets of up to
80 products in `runs/syn_giant/`:

```bash
mkdir -p runs/syn_mixture
ENERGY_MODEL_DATA_ROOT=/path/to/disc_c48_s3/world/model_input_category \
OMP_NUM_THREADS=4 nohup python3 -u scripts/fit_mixture.py \
  --initialization runs/syn_giant/init_nmax80.pt --additive runs/syn_giant/additive_nmax80.pt \
  --out runs/syn_mixture/mixture --recency --centre-history mean --within-competition --promo-common \
  --max-per-category 40 --batch 64 --epochs 2 --val-every 30 --threads 4 \
  > runs/syn_mixture/fit.out 2>&1 &
```

Settings are those of the real-data recipe except `--batch 64` (7,695 training trips) and
`--val-every 30`. About 2 seconds per step, 242 steps. Read `mixture.log` as in §3; the
`comp_iv` rows can be compared with the true competition weight.

The earlier energy form, its checks and its history are on the branches `v4-final` and
`v4-giant-mode`.
