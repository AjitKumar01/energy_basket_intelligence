# Retail Application API

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
satisfy `week = day // 7 + 1`; requests are restricted to the trained promotion coverage,
weeks 9 through 101.

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
