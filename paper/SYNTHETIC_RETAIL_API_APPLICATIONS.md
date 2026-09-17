# Using the retail API: worked applications on the known-truth synthetic retailer

Date: 2026-09-18

## 1. The problem, in plain words

A grocer sells 120 products in 8 categories, through 20 stores, to 3,000 regular
households, over a year. Every week they must answer practical questions:

- **At the till:** three items are on the belt. Is there a fourth this shopper usually
  takes, so the screen can offer it?
- **Packing and staffing:** two items are scanned. Is this a "grab two things" trip or a
  big shop?
- **Reporting:** which broad group does this household belong to, so results can be read by
  segment?
- **Governance:** which of these answers can the retailer act on, and which are guesses the
  data cannot support?

The retail API answers the first three from the fitted basket model and refuses the rest.
Ordinarily you cannot check whether the answers are right, only whether they look
plausible. On the synthetic retailer we know the law that generated every basket, so each
answer can be scored two ways:

- **against the truth:** what the generating model says the best answer was;
- **against outcomes:** what the shopper actually did, on held-out weeks the model never saw.

## 2. Setup

| Piece | Value |
|---|---|
| Served bundle | `data/synthetic_capability_world/model_input_category` |
| Checkpoint | `artifacts/synthetic_capability_test_category/full/artifacts/candidate_rank1.pt` (`326b6638…`) |
| Completion audit | run here; passed at Smolyak levels 9–11 (`level_offset` 3) |
| Segment report | the same run's `customer_segments.json` |
| Script | `scripts/verification/api/synthetic_api_applications.py` |
| Report | `artifacts/synthetic_api_demo/report.json` |
| Sample | 200 held-out test trips, 600 households |

Everything is driven through HTTP requests to the running app, not by calling model code.

## 3. Application by application

### A. Operations: is the service serving what it claims? (`GET /live`, `GET /ready`)

**Use.** A deployment check before traffic is sent.

**Result.** `alive`; ready returns HTTP 200 with the checkpoint hash, and its data
fingerprint matches the bundle on disk. A checkpoint from another dataset would be rejected
at startup.

### B. Governance: which decisions are in scope? (`GET /v1/capabilities`)

**Use.** Before anyone builds on the model, the API states what it was validated for.

**Result.**
- **Available:** basket completion (validated for masked-pair evaluation), cross-sell
  (promising offline, MRR 0.45), customer segmentation (descriptive only).
- **Not available:** causal price optimization, stockout substitution, promotion policy,
  assortment optimization, total demand forecasting, chronological stopping, personalized
  bundle policy.
- **Served data:** the category partition with its 8 groups, and store availability with a
  0.012% weight for products never confirmed in a store.

This is the part that stops a "what if we cut the price 20%?" question turning into a
decision the data cannot support.

### C. Catalogue: turn a cart into product ids (`GET /v1/products/search`)

**Use.** The till has product text; the API needs ids.

**Result.** A search for `bakery` returned 5 products, all from the bakery category, with
department, category, sub-category and brand for each.

### D. Checkout cross-sell: what else will this shopper buy? (`POST /v1/baskets/complete`)

**Use.** A cart of 3.3 items on average; ask for the top 5 products the shopper would add.

**How it was scored.** One item was hidden from each real test basket. Does the API's top 5
contain it?

| Measure | Result |
|---|---|
| Hit rate at 5 (API) | **0.370** |
| Hit rate at 5, best possible (truth-based ranking) | 0.445 |
| Hit rate at 5, popularity baseline | 0.225 |
| Share of attainable lift over popularity | **66%** |
| Overlap of the API's top 5 with the truth's top 5 | 3.44 of 5 |
| Median true rank of the API's top suggestion | 1 |
| Latency, median / p95 | 18.7 ms / 19.3 ms |

**Reading.** The API's first suggestion is usually the truly best product to suggest, and it
recovers two thirds of what a perfect recommender would gain over simply offering the
best-selling product.

### E. "Will they add more?" (`POST /v1/baskets/complete`, audited masked-pair protocol)

**Use.** Two items scanned. How many more are coming, and is the shopper about to stop?
This is the protocol the completion audit certifies, so it requires exactly two revealed
products.

| Measure | API | Simple baseline |
|---|---:|---:|
| Mean additional items, predicted vs observed | 1.944 vs **1.940** | — |
| Mean absolute error, items | **1.114** | 1.317 (always predict the average) |
| Stop rate, predicted vs observed | 0.210 vs **0.240** | — |
| Stop probability Brier score | **0.164** | 0.182 (always predict the base rate) |
| At least one remaining item in the top 5 | 0.42 | — |

**Reading.** Basket size is well calibrated on average and beats the naive baseline, though
per-trip error stays about one item: the model knows the distribution, not the individual
shopper's list. Every response also carries a numerical certificate; levels 10 and 11 were
used here, and a request is refused rather than answered if the two quadrature rules
disagree beyond tolerance.

### F. Segmentation: which group is this household in? (`GET /v1/households/{id}/segment`)

**Use.** Reporting, and stratifying an experiment.

**Result.** 600 households were assigned to 3 segments; agreement with the true segments is
0.61 (adjusted Rand index; 1.0 is perfect, 0 is chance). Every response repeats the
limitation: segments describe taste and price surfaces, and do not justify different offers
to different people.

### G. Refusals: what the service will not answer

| Request | Response |
|---|---|
| Unknown product id | 400 `unknown_product` |
| Duplicate products in the cart | 422 schema validation |
| Empty cart | 422 schema validation |
| One revealed item under the audited masked-pair protocol | 400 `unsupported_anchor_size` |
| Week 999 (outside fitted coverage) | 400 `context_out_of_range` |
| Unknown field (for example `discount`) | 422 "Extra inputs are not permitted" |
| Unknown household | 400 |
| Unknown trip | 400 |

All eight were rejected with a reason, not answered with a guess.

### H. A live basket, not a recorded trip

**Use.** At the till the basket is new, so the request gives household, store, day and week
instead of a historical trip id.

**Result.** For the same household, store and week, the explicit context returned the
identical top 5 and the same expected basket size (difference 0.0), so the live path
matches the historical path.

## 4. What this shows, and what it does not

**Shows.**
- Every capability the API exposes works end to end on a second dataset, with no code
  changes: only the served bundle, checkpoint, audit and segment report differ.
- The answers are good where the model is validated: cross-sell close to the best possible
  ranking, basket size calibrated on average, segments broadly right.
- The refusals hold. Unsupported questions and malformed requests are rejected.

**Does not show.**
- **Nothing causal.** The API never claims a price cut or promotion will produce a gain;
  those capabilities report as unavailable.
- **Per-shopper precision** is limited: basket-size error is about one item per trip, and
  63% of held-out items are not in the top 5.
- **Synthetic-world results are an upper bound** for how well the model can do, because the
  fitted family matches the generating family. ERIM, a real dataset, is harder.

## 5. Reproduce

```bash
# once: certify the completion audit for the synthetic checkpoint (level offset 3)
cd scripts/version4
ENERGY_MODEL_DATA_ROOT=$PWD/../../data/synthetic_capability_world/model_input_category V3_AFFINITY=1 \
  python audit_basket_completion_corrected.py \
    --checkpoint $PWD/../../artifacts/synthetic_capability_test_category/full/artifacts/candidate_rank1.pt \
    --output $PWD/../../artifacts/synthetic_capability_test_category/retail_application/real_basket_completion_corrected.json \
    --threads 4 --level-offset 3

# the applications
cd ../..
python scripts/verification/api/synthetic_api_applications.py --cases 200 --households 600
```

Serving the synthetic model from a real server instead of the test client:

```bash
RETAIL_API_DATA_ROOT=data/synthetic_capability_world/model_input_category \
RETAIL_API_CHECKPOINT=artifacts/synthetic_capability_test_category/full/artifacts/candidate_rank1.pt \
RETAIL_API_COMPLETION_AUDIT=artifacts/synthetic_capability_test_category/retail_application/real_basket_completion_corrected.json \
RETAIL_API_SEGMENT_REPORT=artifacts/synthetic_capability_test_category/full/reports/customer_segments.json \
python scripts/run_retail_api.py --host 127.0.0.1 --port 8000
```
