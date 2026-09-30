# Version-4 energy basket model

An energy-based model of shopping baskets and the pipeline that trains it. For a shopping
trip \(x\) (household, store, week, prices, promotions, offered products) the model assigns
every nonempty basket \(S\) the probability

\[
p_\theta(S\mid x)=\frac{\exp E_\theta(S,x)}{Z_\theta(x)},
\]

where the energy adds product utilities (taste, price, promotion, season, store), a
within-group penalty on buying several products from one substitute group, a basket-size
potential, and low-rank pairwise interactions between products. The normalizer
\(Z_\theta(x)\), a sum over every possible basket, is computed exactly through a Gaussian
identity, elementary symmetric polynomials and sparse-grid quadrature.

One set of fitted parameters gives the basket likelihood, recommendations (rank a missing
item given the rest), synthetic basket generation, and price counterfactuals, all from the
same law. The theory is in [`docs/THEORY.md`](docs/THEORY.md); the documentation map is
[`docs/README.md`](docs/README.md).

## Pipeline

`scripts/run_pipeline.py` runs these stages on a prepared model-data bundle:

| Stage | What it does |
|---|---|
| `data` | verifies the bundle's fingerprint |
| `initialize` | fresh, seeded initialization |
| `additive` | price-response evidence, then exact maximum likelihood without interactions |
| `rank` | chooses the interaction directions and rank from the additive model's pair residual |
| `interaction` | fits the interaction strengths and a size correction; then a per-household size direction |
| `refinement` | *optional* (`--joint-refinement`): refits utilities, tastes, penalties, size and interaction strengths jointly; kept only if it passes held-out and certification gates |
| `evaluation` | validation and test likelihood, recommendation, generation and counterfactual audit, customer segments, interaction audit |
| `certification` | population basket-size safety audit, size diagnostic, promotion-budget MDP |

Every stage writes durable artifacts, so a run can be resumed or moved
([`docs/STAGEWISE_RESURRECTION.md`](docs/STAGEWISE_RESURRECTION.md)).

## Requirements

- Python 3.11 or newer (developed on 3.13)
- a C++ compiler on `PATH` (`clang++` or `g++`); the pipeline builds a small PyTorch
  extension on first use
- CPU only; the full profile expects at least 12 GB RAM and 5 GB free disk

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Quick start on the synthetic retailer

The repository ships a synthetic retailer generator (about 476 products, 3,000 households,
15 stores, 52 weeks) whose baskets are **not** produced by this model, with known causal price
effects ([`docs/SYNTHETIC_STRESS_WORLD.md`](docs/SYNTHETIC_STRESS_WORLD.md)).

```bash
# 1. generate the data (about 1–2 minutes)
python -u scripts/synthetic/generate_stress_world.py --output data/stress_world

# 2. prepare the model-data bundle (under a minute)
python -u scripts/prepare_model_bundle.py --dataset-config configs/datasets/stress_world_category.json

# 3a. software check: every stage with small settings (a few minutes); not a certified fit
python -u scripts/run_pipeline.py --model-data-root data/stress_world/model_input_category \
    --run-dir artifacts/stress_smoke --profile smoke --joint-refinement --evaluation-level-offset 3

# 3b. full run (about an hour on a 15-core laptop)
python -u scripts/run_pipeline.py --model-data-root data/stress_world/model_input_category \
    --run-dir artifacts/stress_full --profile full --joint-refinement --evaluation-level-offset 3
```

`--evaluation-level-offset 3` evaluates the normalizer one quadrature level finer than the
default, which this world needs to pass the numerical audit. Leave out `--joint-refinement` to
run the staged pipeline only.

## Outputs

Under the run directory (`artifacts/stress_full/` above):

| Path | Content |
|---|---|
| `artifacts/candidate_rank1.pt` | the staged model |
| `artifacts/candidate_refined.pt` | the refined model, present only if refinement was accepted |
| `reports/joint_refinement.json` | refinement rounds and the acceptance decision |
| `reports/likelihood_{validation,test}.json` | held-out log-likelihood against the additive parent, with the numerical audit |
| `reports/recommendation.json` | hide-one-item ranking (MRR, recall) |
| `reports/generation_counterfactual.json` | generated baskets and model price scenarios |
| `reports/population_size.json` | basket-size safety certification |
| `invocation_<time>/` | `pipeline.log`, one log per stage command, and `manifest.json` with each stage's status and run time |

Follow a run with `tail -f <run>/invocation_*/pipeline.log`.

## Other datasets

Write the dataset in the canonical format, add a config under `configs/datasets/`, and run
`prepare_model_bundle.py` and `run_pipeline.py` as above. The format, config keys and
product-partition options are in
[`docs/CANONICAL_INPUT_CONTRACT.md`](docs/CANONICAL_INPUT_CONTRACT.md).

## Tests

```bash
python scripts/version4/setup_poly_degree_native.py build_ext \
    --build-lib artifacts/native/lib --build-temp artifacts/native/temp
python -m pytest -q
```

Tests that need the compiled extension skip, with the build command, if it has not been built.

## Repository layout

```text
scripts/run_pipeline.py            pipeline driver
scripts/prepare_model_bundle.py    canonical directory + config -> model-data bundle
scripts/synthetic/                 synthetic retailer generator
scripts/version4/                  model, estimators, stages and audits
configs/datasets/                  dataset configurations
docs/                              theory and pipeline documentation
tests/                             unit tests
```
