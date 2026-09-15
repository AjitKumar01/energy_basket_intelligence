# Retail Application API

Retail operations and merchandising teams should use the plain-language
[`RETAILER_API_USER_MANUAL.md`](RETAILER_API_USER_MANUAL.md). This document is the
technical interface and evidence reference for engineering and model-governance teams.

## Scope

The API exposes only applications supported by the current audits:

1. masking-aware basket-size and stopping probabilities;
2. cross-sell ranking from masking-aware conditional item incidence; and
3. descriptive customer-segment lookup.

The basket and cross-sell outputs are two views of one normalized conditional joint law,
not separate models. Every response is computed with adjacent deterministic quadrature
rules. A request is rejected with HTTP 503 if the higher rule does not confirm the lower
rule within the declared tolerances. Startup also verifies the checkpoint, data
fingerprint, audit result, segment assignment digest, and trained capabilities.

The following are deliberately not decision endpoints: causal price optimization,
stockout substitution, promotion policy, assortment optimization, total demand
forecasting, chronological stopping, and personalized bundle policy. Their current data
or application gates did not pass. `GET /v1/capabilities` reports these exclusions.

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
RETAIL_API_CHECKPOINT
RETAIL_API_COMPLETION_AUDIT
RETAIL_API_SEGMENT_REPORT
RETAIL_API_THREADS
```

Use one server worker unless memory has been sized for multiple independent checkpoint,
feature-panel, and catalogue copies.

## Product and context identifiers

Public basket requests use the source `PRODUCT_ID` from `items.parquet`. Product search
returns this ID together with the internal model index and readable metadata.

`household_index` and `store_index` refer to the fitted cohort's contiguous internal
indices because the prepared data do not retain a source household-ID lookup. A production
integration must persist that mapping during preprocessing. `day` is zero-based and must
be supplied together with the source `WEEK_NO`; the two source fields are deliberately not
re-derived from one another. Requests are restricted to the trained promotion coverage,
weeks 9 through 101, and days 0 through 711.

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

The service was benchmarked through a real localhost HTTP connection, not by timing its
Python methods. The run used one uvicorn worker, four PyTorch CPU threads and sequential
HTTP/1.1 requests. Cheap endpoints received 30 requests. Each steady-state inference
endpoint received 20 requests. These are single-machine development measurements, not a
production load test.

| Endpoint | First/cold | Warm p50 | Warm p95 | Median response |
|---|---:|---:|---:|---:|
| `GET /live` | — | 0.409 ms | 0.577 ms | 40 B |
| `GET /ready` | 261.150 ms | 0.513 ms | 0.657 ms | 198 B |
| `GET /v1/capabilities` | — | 0.531 ms | 0.552 ms | 1,073 B |
| `GET /openapi.json` | — | 0.361 ms | 0.407 ms | 9,088 B |
| `GET /v1/products/search` | — | 1.325 ms | 1.582 ms | 167 B |
| `GET /v1/households/{id}/segment` | — | 0.579 ms | 0.611 ms | 381 B |
| `POST /v1/baskets/complete`, masked pair | 512.136 ms | 442.771 ms | 445.220 ms | 12,115 B |
| `POST /v1/baskets/complete`, literal cart | 442.813 ms | 444.578 ms | 445.922 ms | 12,065 B |
| Rejected unsupported mask size | — | 0.528 ms | 0.668 ms | 137 B |

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

At roughly 445 ms p95, the exact call can populate a non-blocking app carousel or a
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
