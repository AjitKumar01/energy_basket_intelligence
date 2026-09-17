# Retail Application API

Retail operations and merchandising teams should use the plain-language
[`RETAILER_API_USER_MANUAL.md`](RETAILER_API_USER_MANUAL.md). This document is the
technical interface and evidence reference for engineering and model-governance teams.

## Served model

Since 2026-09-17 the default service loads the ERIM category-partition refit
([ERIM_CATEGORY_PARTITION_REFIT.md](ERIM_CATEGORY_PARTITION_REFIT.md)):

| Artifact | Default path |
|---|---|
| Model-data bundle | `data/erim_basket/model_input_availability_category` |
| Checkpoint | `artifacts/erim_category_refit/full/artifacts/candidate_rank1.pt` |
| Completion audit | `artifacts/erim_category_refit/retail_application/real_basket_completion_corrected.json` |
| Segment report | `artifacts/erim_category_refit/full/reports/customer_segments.json` |

The catalogue is ERIM's 464 products in eight tracked categories. Baskets are
tracked-category sub-baskets, not whole grocery checkouts.
- **Substitution groups.** The exact within-group penalty uses the eight merchandise
  categories, so brands in one category can substitute for each other.
- **Store availability.** Availability is part of the model: a product not yet confirmed
  at the context's store keeps 0.3% of its weight.
- **Reporting.** `GET /v1/capabilities` reports the dataset, the substitution-group
  partition and the availability contract under `model_data`.

The completion audit certifies Smolyak levels 11, 12 and 13 (`level_offset` 3). The
standard levels 10–12 fail the 1e-4 stop, size and incidence gates on this checkpoint, so
the API reads the certified levels from the audit rather than using a fixed offset.

Masked-pair completion accuracy on the 256 audited validation contexts:

| Measure | Category checkpoint | Previous (co-purchase groups) |
|---|---:|---:|
| MAE, additional items (baseline 0.771) | 0.737 | 0.737 |
| Stop log loss | 0.646 | 0.644 |
| Stop ROC AUC | 0.658 | 0.658 |

The switch was made for the refit's better held-out likelihood and next-item ranking, and
because the model can now represent substitution; masked completion accuracy is unchanged.

The previous co-purchase-group ERIM checkpoint remains servable. Set
`RETAIL_API_DATA_ROOT=data/erim_basket/model_input_availability`, and point the three
artifact variables (`RETAIL_API_CHECKPOINT`, `RETAIL_API_COMPLETION_AUDIT`,
`RETAIL_API_SEGMENT_REPORT`) at `artifacts/erim_availability_refit/...`.

The previous Dunnhumby checkpoint is still servable. Set `RETAIL_API_DATA_ROOT` to the
repository root, and point `RETAIL_API_CHECKPOINT`, `RETAIL_API_COMPLETION_AUDIT` and
`RETAIL_API_SEGMENT_REPORT` at `artifacts/corrected_complete_rank5_20260913/...` and
`artifacts/retail_application_audit_20260915/...`. The bread-basket case study below and the
retailer manual's example use that Dunnhumby checkpoint.

## Scope

The API exposes only applications supported by the current audits:

1. masking-aware basket-size and stopping probabilities;
2. cross-sell ranking from masking-aware conditional item incidence;
3. descriptive customer-segment lookup; and
4. price scenarios, **only** where the deployment's verdict file claims causal price
   optimization (`POST /v1/baskets/price_scenario`; HTTP 403 otherwise). The response gives
   each product's purchase probability with and without the declared price change, the
   change per category, the change in expected basket size, and an adjacent-rule numerical
   certificate. Price multipliers are limited to the declared scenario window (0.2, 3.0].

The basket and cross-sell outputs are two views of one normalized conditional joint law,
not separate models. Every response is computed with adjacent deterministic quadrature
rules. A request is rejected with HTTP 503 if the higher rule does not confirm the lower
rule within the declared tolerances. Startup also verifies the checkpoint, data
fingerprint, audit result, segment assignment digest, and trained capabilities.

Decision capabilities (causal price optimization, stockout substitution, promotion policy,
assortment optimization, total demand forecasting, chronological stopping, personalized
bundle policy) are **refused by default**, and no decision endpoint exists for them. A
deployment may claim one only through a declared verdict file:

- **Path:** `RETAIL_API_CAPABILITY_VERDICTS`, else `capability_verdicts.json` beside the
  checkpoint.
- **Contract:** `schema_version` 1, the serving checkpoint's `checkpoint_sha256` and
  `data_fingerprint_sha256`, and per capability a `status` of `supported`, `limited`,
  `unsupported` or `untested`. A `supported` or `limited` verdict must carry `evidence` and
  a `source`; anything else is rejected with HTTP 503.
- **Reporting:** `GET /v1/capabilities` returns claimed capabilities under
  `available_decisions` with their evidence, the rest under `unavailable`, and the verdict
  file's path and hash under `capability_verdicts`.
- **ERIM has no verdict file,** so every decision capability stays refused: its prices are
  observational and its policy evaluation reports `not_identifiable`. The synthetic
  deployment does have one, written by `scripts/synthetic/evaluate_capabilities.py` from
  oracle comparisons; see
  [SYNTHETIC_RETAIL_API_APPLICATIONS.md](SYNTHETIC_RETAIL_API_APPLICATIONS.md).

## Start the service

Install the pinned environment and run one worker:

```bash
python -m pip install -r requirements.txt
python scripts/run_retail_api.py --host 127.0.0.1 --port 8000
```

Interactive OpenAPI documentation is available at `http://127.0.0.1:8000/docs`.
The model loads on the first endpoint that requires it. `/live` tests the web process;
`/ready` loads and verifies all model artifacts.

Configuration can be overridden with:

```text
RETAIL_API_DATA_ROOT
RETAIL_API_CHECKPOINT
RETAIL_API_COMPLETION_AUDIT
RETAIL_API_SEGMENT_REPORT
RETAIL_API_THREADS
```

Use one server worker unless memory has been sized for multiple independent checkpoint,
feature-panel, and catalogue copies.

## Product and context identifiers

Public basket requests use the source `PRODUCT_ID` from `items.parquet`. Product search
returns this ID together with the internal model index and readable metadata. For the ERIM
bundle, `PRODUCT_ID` equals the internal item index, and the UPC-based identifier is
`product_id` (`category:UPC`) in `items.parquet`.

`household_index` and `store_index` refer to the fitted cohort's contiguous internal
indices because the prepared data do not retain a source household-ID lookup. A production
integration must persist that mapping during preprocessing. `day` is zero-based and must
be supplied together with the source `WEEK_NO`; the two source fields are deliberately not
re-derived from one another. Requests are restricted to the served bundle's price and
promotion coverage: weeks 1 through 51 and days 0 through 356 for ERIM, or weeks 9 through
101 and days 0 through 711 for Dunnhumby.

A `historical_trip` context is supplied for reproducible demonstrations. It reconstructs
the household, store, day, week, factual prices, and factual promotions from the prepared
trip. It must not be confused with a new retail event.

## Endpoints

### `GET /v1/products/search?q=cereal&limit=10`

Search the fitted catalogue and retrieve source product IDs suitable for basket requests.

### `POST /v1/baskets/complete`

```json
{
  "context": {
    "kind": "retail_context",
    "household_index": 42,
    "store_index": 10,
    "day": 650,
    "week": 93
  },
  "revealed_product_ids": [818980, 818981],
  "top_k": 20,
  "protocol": "literal_cart"
}
```

The response contains:

- the probability that the completion is empty;
- expected additional item count;
- the full additional-count distribution;
- top products ranked by conditional completion incidence; and
- the adjacent-rule numerical certificate used for that request.

`literal_cart` evaluates

```text
P(T | A is a subset of the eventual basket, x).
```

It is the direct query supplied by the fitted set law, but the real-data pair-incidence
size audit was weak and the data contain no scan order.

`uniform_random_subset` evaluates the protocol that passed the retrospective audit:

```text
q(T | A,x) proportional to P(T | A subset S,x) / C(|A|+|T|, |A|).
```

Use this protocol when `A` was uniformly masked from a completed basket. Do not use it for
a live chronological cart unless the live observation mechanism has been shown to match.
The current real-data certificate covers exactly two revealed products, and the endpoint
rejects any other anchor size for this protocol.

### `GET /v1/households/{household_index}/segment`

Returns the audited three-segment descriptive label and summary. The endpoint explicitly
does not assert that segments have different causal treatment effects.

### Operational and evidence endpoints

- `GET /live`: HTTP process is running.
- `GET /ready`: checkpoint and evidence artifacts loaded and verified.
- `GET /v1/capabilities`: available applications, measured evidence, and excluded uses.

## Deployment boundary

Cross-sell results are suitable for candidate generation followed by a randomized online
test. They are not evidence of incremental sales. Basket completion is validated for a
retrospective uniform-subset masking protocol; chronological use requires cart-event
timestamps and checkout labels. Customer segments are suitable for reporting and
experiment stratification, not treatment targeting.

## Measured latency

### ERIM category-partition checkpoint (current default)

Same method: one worker, four threads, sequential HTTP/1.1, 30 cheap and 10 inference
repeats.

| Endpoint | First/cold | Warm p50 | Warm p95 |
|---|---:|---:|---:|
| `GET /ready` | 0.647 ms | 0.512 ms | 0.662 ms |
| `GET /v1/capabilities` | — | 0.648 ms | 0.732 ms |
| `GET /v1/products/search` | — | 1.060 ms | 1.153 ms |
| `GET /v1/households/{id}/segment` | — | 0.591 ms | 0.642 ms |
| `POST /v1/baskets/complete`, masked pair | 147.364 ms | 129.611 ms | 131.175 ms |
| `POST /v1/baskets/complete`, literal cart | 123.145 ms | 123.876 ms | 129.466 ms |

Exact completion runs at roughly 7.7–8 single-context requests per second on one worker,
slightly faster than the previous ERIM checkpoint (134.8 ms p50). Smoke, audit and
benchmark results are in `artifacts/erim_category_refit/retail_application/`. The cold
`/ready` figure excludes model loading, which happened in an earlier request.

### Dunnhumby checkpoint (earlier default)

The service was benchmarked through a real localhost HTTP connection, not by timing its
Python methods. The run used one uvicorn worker, four PyTorch CPU threads and sequential
HTTP/1.1 requests. Cheap endpoints received 30 requests. Each steady-state inference
endpoint received 20 requests. These are single-machine development measurements, not a
production load test.

| Endpoint | First/cold | Warm p50 | Warm p95 | Median response |
|---|---:|---:|---:|---:|
| `GET /live` | — | 0.434 ms | 0.571 ms | 40 B |
| `GET /ready` | 366.196 ms | 0.500 ms | 0.639 ms | 198 B |
| `GET /v1/capabilities` | — | 0.512 ms | 0.581 ms | 1,073 B |
| `GET /openapi.json` | — | 0.344 ms | 0.392 ms | 9,088 B |
| `GET /docs` | — | 0.310 ms | 0.348 ms | 954 B |
| `GET /redoc` | — | 0.302 ms | 0.317 ms | 911 B |
| `GET /v1/products/search` | — | 1.325 ms | 1.418 ms | 167 B |
| `GET /v1/households/{id}/segment` | — | 0.572 ms | 0.627 ms | 381 B |
| `POST /v1/baskets/complete`, masked pair | 640.813 ms | 441.616 ms | 442.977 ms | 12,115 B |
| `POST /v1/baskets/complete`, literal cart | 441.261 ms | 442.451 ms | 446.112 ms | 12,065 B |
| Rejected unsupported mask size | — | 0.545 ms | 0.740 ms | 137 B |

Exact basket inference is the bottleneck. One worker currently serializes it behind a
model lock and sustains approximately 2.25 single-context requests per second. The API is
therefore suitable for analyst use, batch validation and a controlled low-volume pilot.
It should not block a high-throughput POS transaction. If an online experiment succeeds,
the next engineering step is a cached or distilled retrieval layer with this exact API as
the shadow evaluator and audit oracle.

The machine-readable benchmark and console log are
`artifacts/retail_api_latency.json` and `artifacts/retail_api_latency.log`.

## Case study: bread-basket completion at a regional grocer

Assume a regional grocer wants to test recommendations after two products are known. The
following case is an actual held-out validation context, not a fabricated product story.

The context is fitted household 1745, store 40, day 601 and source week 87. The two
revealed source product IDs are:

- `849843`: mainstream wheat/multigrain bread;
- `1045586`: fruit/breakfast bread.

The completed historical basket contained three additional products: jumbo eggs, cream
cheese and butter. This outcome remains hidden from the API request.

### Step 1: verify the deployed lineage

The deployment first calls `/ready` and requires HTTP 200. It records the returned
checkpoint and data fingerprints. It then calls `/v1/capabilities`; if basket completion
or cross-sell is not listed as available, the recommendation workflow remains disabled.

### Step 2: resolve retailer product IDs

During integration, the catalogue team calls `/v1/products/search` to verify that source
product IDs map to the correct model products. This is a configuration check, not
something the checkout path needs to repeat for every basket.

### Step 3: reproduce the audited offline protocol

For retrospective validation, the retailer uniformly masks two products from a completed
basket and sends:

```json
{
  "context": {
    "kind": "retail_context",
    "household_index": 1745,
    "store_index": 40,
    "day": 601,
    "week": 87
  },
  "revealed_product_ids": [849843, 1045586],
  "top_k": 10,
  "protocol": "uniform_random_subset"
}
```

For this case, the API returned expected additional count `3.071`, stop probability
`0.273`, and selected the 341-node rule without a follow-up. The actual additional count
was three. Its highest-ranked products included bananas, extra-large eggs, white milk,
condensed soup and lettuce. Extra-large eggs at rank two are commercially related to the
hidden jumbo eggs, but are not the same SKU. This distinction is why the retailer should
score exact-SKU recall as well as category relevance.

### Step 4: construct an online candidate list

For an actual live cart, the two scanned items were not uniformly masked from a final
basket. The retailer must request `literal_cart` and treat its output as an experimental
candidate ranking—not as a certified remaining-count or checkout prediction. Before any
candidate is displayed, the retailer intersects the top results with current inventory,
removes regulated or suppressed products, applies merchandising constraints and records
the exact model score and checkpoint hash.

At roughly 447 ms p95, the exact call can populate a non-blocking app carousel or a
low-volume pilot with a 600 ms service budget. It should not delay payment. The client
must time out, show no model recommendation on failure, and continue checkout normally.

### Step 5: use segments only to stratify the experiment

`GET /v1/households/1745/segment` assigns this household to segment 0, labelled
“REFRIGERATED / ORGANICS FRUIT & VEGETABLES; medium price sensitivity.” The retailer uses
that label to balance randomization and report heterogeneous descriptive results. It does
not assign a larger discount or a different treatment merely because of the segment.

### Step 6: run the commercial test

Randomize household-weeks within store and segment into three arms:

1. no recommendation;
2. training-popularity recommendation; and
3. model-ranked recommendation after inventory and policy filtering.

For every eligible opportunity, log the revealed cart, full candidate list, inventory,
prices, exposure, rank, click/add event, final purchased SKU, units, cost, margin, latency,
timeout and checkout abandonment. The primary decision metrics are incremental exact-SKU
attach rate and incremental margin versus both controls. Category attach rate is a
secondary diagnostic. Latency and abandonment are safety metrics.

### Step 7: promote or stop the application

Promote the model only if the randomized interval for incremental margin is positive and
the latency/abandonment guardrails pass. If ranking wins but exact latency fails, distil
the successful ranking into a faster serving model and keep the current exact API running
in shadow mode to detect approximation drift. No result from this experiment validates
price, promotion, stockout or assortment interventions; those remain separate gated
applications.
