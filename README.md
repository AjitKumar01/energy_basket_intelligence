# Version-4 energy basket model: the pipeline

One joint probability model of shopping baskets with an exact normaliser, fitted to
transaction data, used for basket likelihood, realistic basket generation and price and
promotion counterfactuals (own product and substitutes).

- **Write-up:** [`docs/PIPELINE.md`](docs/PIPELINE.md): the model, the stages, which parameters
  each stage learns, how models are chosen and checked, current results and known weaknesses.
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

This branch (`v4-pipeline`) holds only the current pipeline. The research history, diagnostics
and earlier pipelines are on `v4-training`.
