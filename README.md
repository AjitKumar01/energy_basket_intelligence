# Version-4 energy basket model: the pipeline

One joint probability model of shopping baskets with an exact normaliser, fitted to
transaction data, used for basket likelihood, realistic basket generation and price and
promotion counterfactuals (own product and substitutes).

- **Write-up:** [`docs/PIPELINE.md`](docs/PIPELINE.md): the model, the stages, which parameters
  each stage learns, what is decided on validation data, the checks, current results and known weaknesses.
- **Run:** `run/dunnhumby.sh`, `run/stress_world.sh`, `run/shopper.sh` (see the write-up, §8).
- **Check:** `python3 -m pytest -q`: exact tests against enumeration on a tiny world (no data).

| Folder | Content |
|---|---|
| `v4/` | the model: data bundle, features, batches, utilities and energy, exact normaliser, sampler |
| `v4/native/` | C++ kernels (built on first use into `build/native/`) |
| `scripts/` | one script per stage: partition, discount panel, initialize, additive fit, Version-4 fit, evaluate |
| `checks/` | the real-data discount and promotion check |
| `shopper/` | the SHOPPER baseline (PyTorch reimplementation) and its protocol checks |
| `tests/` | exact tests |
| `run/` | the full recipes |

## On a new machine

1. **Software:** Python 3.11+, a C++ compiler (clang++ or g++; set `CXX` to choose), then
   `pip install -r requirements.txt`.
2. **Check:** `python3 -m pytest -q`.  This compiles the two native kernels into `build/native/`
   and checks the exact law against enumeration; it needs no data.
3. **Data:** copy a model-data bundle, i.e. the folder holding `basket_input/` and `data/`
   (with `basket_input/model_data_fingerprint.json` and `basket_input/v3_index_affinity.npz`),
   and set `DATA` to it.  Checkpoints only load with the bundle they were fitted on.
4. **Run:**

   ```bash
   DATA=/path/to/bundle nohup run/dunnhumby.sh > dunnhumby.out 2>&1 &          # every stage
   DATA=... INIT=... ADDITIVE=... nohup run/dunnhumby_step4.sh > step4.out 2>&1 &   # Version 4 only
   WORLD=/path/to/stress_world REFERENCE=... nohup run/stress_world.sh > stress.out 2>&1 &
   DATA=... nohup run/shopper.sh > shopper.out 2>&1 &
   ```

   To continue from earlier results instead of refitting, copy the initialization and the
   fitted base model too and pass them as `INIT` and `ADDITIVE`.  `THREADS` sets threads per
   job (default 4).

This branch (`v4-pipeline`) holds only the current pipeline. The research history, diagnostics
and earlier pipelines are on `v4-training`.
