# Retailer User Manual: Basket Suggestions and Customer Groups

## Who should use this manual

This manual is for retail managers, merchandisers, digital-commerce teams, store
operations teams and IT staff. You do not need to understand probability theory or how
the model was trained. The important points are what information to send, what the answer
means, what action is safe, and what the system must not be used for.

The service can currently help with three jobs. It can suggest products that may belong
with products already in a basket. It can estimate how many more products may appear in a
completed basket under the tested offline procedure. It can also place a known customer
in one of three descriptive groups for reporting and for balancing a business experiment.
These are decision-support functions. They are not automatic proof that a recommendation
caused a sale.

## What the retailer must not use it for

Do not use this service to set prices, promise profit, select discounts, remove products
from stores, estimate lost demand during stockouts, forecast total store demand, or decide
that a customer has finished shopping. The available transaction data did not contain the
experiments, stock records, shopping order, costs or visit opportunities needed to verify
those decisions. The service reports these applications as unavailable instead of
guessing.

## The two basket modes in ordinary language

The basket endpoint has two modes. Choosing the correct mode is essential.

The `uniform_random_subset` mode is the verified offline test mode. Start with a completed
historical basket, hide all but two randomly selected products, and ask the service what
else it expects. Use this mode to measure performance on past baskets. Exactly two
products must be supplied. Do not use this mode for a live shopping cart because the first
two products scanned by a shopper were not randomly selected from the final basket.

The `literal_cart` mode is the live-pilot candidate mode. It accepts the products that are
actually known in a current cart and returns possible additions. Its product ranking may
be used to prepare an experiment, but its remaining-count and checkout estimates are not
certified for live shopping. A retailer must run a controlled online test before using
this output as a normal customer-facing feature.

## Before the first use

The retail IT team must connect the retailer's product catalogue to the model catalogue.
The basket endpoint accepts the retailer's source `PRODUCT_ID`, not the model's internal
product number. The product-search endpoint can be used to confirm the mapping. The
current prepared files do not retain the retailer's original household-ID mapping, so IT
must securely maintain a table that translates the retailer's customer key to the fitted
`household_index`. The same applies to the fitted `store_index` where store systems use a
different identifier.

IT must also provide both the transaction day and the source week number. These are two
separate source fields and must not be calculated from one another. The current service
accepts days 0 through 711 and weeks 9 through 101 because that is the period covered by
the fitted data and promotions. A request outside that period is not evidence about a new
calendar period; the model needs a controlled refresh and drift check before later weeks
are served.

Start one copy of the service with:

```bash
python scripts/run_retail_api.py --host 127.0.0.1 --port 8000
```

The IT operator first opens `/live`. A successful answer means only that the web process
is running. The operator then opens `/ready`. A successful answer means the model,
catalogue, customer groups and audit records were loaded and belong together. Traffic
must not be sent to a server that does not return `200` from `/ready`.

## Step-by-step offline evaluation

### Step 1: select historical baskets fairly

Select completed baskets from a period that was not used to train the model. Do not select
only large baskets, only loyal customers or only successful promotions. Such selection
would make the result look better or worse than ordinary trade. Keep baskets containing
at least two distinct products and within the model's supported basket size.

### Step 2: hide products without looking at their identity

For each selected basket, randomly choose exactly two products to reveal. Hide every
other product. The random choice must be made by software before anyone looks at which
products were selected. This creates the same kind of test that passed the current audit.

### Step 3: send the test request

Send the customer, store, day and source week together with the two visible source product
IDs. Set `protocol` to `uniform_random_subset`. For example:

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

### Step 4: read the answer

`expected_additional_items` is the service's average estimate of how many hidden products
remain. It is not a promise for an individual basket. An answer of `3.1` means that many
similar cases average about three additional products; it does not mean that this shopper
must purchase exactly three.

`stop_probability` is the model's estimated chance that the two visible products are the
whole completed basket under this offline testing procedure. A value of `0.27` means about
27 cases out of 100 similar model cases would contain nothing else. It does not mean the
live shopper has a 27% chance of going to checkout.

Each recommendation contains `probability_in_completion`. This is the model's estimate
that the product belongs somewhere in the hidden remainder. Products can appear together,
so these values do not need to add to 100%. Use the values to rank candidates, not as a
claim of extra sales.

The `numerical_certificate` is an automatic calculation-quality check. Business users do
not need to interpret its individual fields. The important rule is simple: the service
returns an error instead of a recommendation when the calculation is not precise enough.

### Step 5: score the offline result

For each basket, check whether the exact hidden products occur in the top 5, 10 and 20
recommendations. Also record whether a related category was recovered, but keep that
separate from exact-product success. Compare the model with a simple popularity list on
the same baskets. Report average remaining-count error, exact-product recall, ranking
quality and calculation failures. Do not remove failed or difficult cases from the
report.

The current 256-basket audit found average remaining count `6.519` versus `6.688`
observed. Mean absolute error was `6.119` versus `6.867` for the training-only size
baseline. On 200 baskets that truly had hidden products, model ranking MRR was `0.367`
versus `0.218` for popularity, and hidden-product recall at 20 was `23.47%` versus
`11.54%`. These are encouraging offline results, not measured incremental revenue.

## Step-by-step live pilot

### Step 1: define one small customer-facing location

Choose one non-essential recommendation surface, such as an optional mobile-app carousel.
Do not place the service in the payment path. The exact basket request currently takes
about 447 milliseconds at the 95th percentile on the development machine. A timeout or
service failure must leave checkout unchanged and display no model recommendation.

### Step 2: construct the request from the current cart

When a known customer has one or more products in a cart, translate the customer, store
and products to the fitted identifiers. Send the actual products with `protocol` set to
`literal_cart`. Do not label its stopping or remaining-count fields as verified live
predictions. The live pilot uses the returned product order only to form candidates.

### Step 3: apply retail safety rules

Before displaying anything, remove products that are out of stock, legally restricted,
incompatible with the channel, suppressed by the customer, or already in the cart. Apply
frequency limits so the same suggestion is not repeatedly shown. Retain the original
model rank and probability in the event log even if a business rule removes an item.

### Step 4: randomize the recommendation

Assign eligible household-weeks to one of three groups: no recommendation, a normal
popularity recommendation, or the model recommendation. Keep the assignment fixed for the
household-week so that a shopper does not jump between experiences. Balance the groups by
store and by the descriptive customer segment. The segment is used to make the test fair,
not to decide who deserves an offer.

### Step 5: record what happened

For every eligible opportunity, record the request time, checkpoint fingerprint, visible
cart, full model candidate list, candidates removed by business rules, inventory, shelf
price, products actually displayed, click or add event, final purchases, units, cost,
margin, response latency, timeout and checkout abandonment. Also record eligible control
opportunities where nothing was displayed. Without those control records, the retailer
cannot know whether a product would have been bought anyway.

### Step 6: use business decision measures

The main measures are the change in exact recommended-product attachment and the change
in contribution margin relative to both controls. Report uncertainty around each change.
Category attachment is useful as a secondary measure but cannot replace exact-product
sales. Check that timeouts, page delay, abandonment, complaints and repeated exposure did
not get worse.

A practical launch rule is: do not expand unless the lower confidence limit for
incremental margin is above zero, the model beats the popularity arm, the latency target
is met, and no customer-safety measure is worse. The retailer should set the exact margin,
latency and safety limits before looking at the result.

### Step 7: decide what happens next

If the model does not improve margin, stop the customer-facing treatment and retain the
offline service for analysis. If ranking improves margin but 447 milliseconds is too slow,
build a cached or smaller serving model and run the exact service in the background as a
quality reference. Compare the fast model with the exact service regularly so that speed
does not silently change the recommendation logic.

## Worked example in ordinary business language

In one real held-out case, the visible basket contained wheat/multigrain bread and
fruit/breakfast bread. The hidden completed basket also contained jumbo eggs, cream cheese
and butter. The offline test endpoint estimated `3.071` additional products; three were
actually hidden. It gave a 27.3% chance of no additional products. Extra-large eggs were
ranked second, which is relevant to the hidden jumbo eggs but is not an exact-SKU hit.

A merchandiser should read this as follows: the model recognized a plausible breakfast or
pantry shopping pattern and produced an egg alternative near the top. It did not recover
all exact hidden products, and the high stop probability shows uncertainty. This is a
useful candidate-generation example, not proof that displaying eggs would cause an extra
sale. Only the randomized live pilot can answer that business question.

## How to use customer segments

The segment endpoint returns a segment number and a plain label. Use the segment to create
balanced experimental groups, summarize trading patterns and check whether the service
behaves very differently across existing types of customers. Do not infer that a segment
is more persuadable, should pay a different price, or should receive a bigger discount.
Those are treatment decisions and need randomized evidence.

For the worked example, household 1745 belongs to segment 0, labelled “REFRIGERATED /
ORGANICS FRUIT & VEGETABLES; medium price sensitivity.” This describes the fitted
household profile. It does not change the basket recommendation and does not authorize a
price action.

## What to do when the service returns an error

An HTTP `400` response means the request is not supported, such as an unknown product or
the wrong number of products for offline mask mode. Correct the input; do not retry the
same request repeatedly. An HTTP `422` response means the request does not follow the
published format or an index is outside its permitted range. An HTTP `503` response means
the model, audit lineage or numerical calculation did not pass a safety check. Show no
model recommendation and alert the service owner. A timeout has the same business
fallback: continue the customer journey without the recommendation.

## Daily operator checklist

Before traffic begins, confirm `/live`, `/ready` and `/v1/capabilities` return successfully.
Confirm that the checkpoint fingerprint equals the approved release. Check that inventory
and product-ID mapping feeds are current. During operation, watch p50 and p95 latency,
timeouts, HTTP 400/422/503 counts and request volume. After operation, reconcile exposures
with purchases and margins, and confirm that every treatment opportunity has a valid
randomized assignment or control record.

Do not replace the checkpoint or audit files independently. A new checkpoint requires the
full application audit, a new approved fingerprint, a shadow comparison with the current
release and a fresh pilot decision. The `/ready` lineage checks prevent many accidental
mismatches, but they cannot determine whether the retailer collected the right live
business data.
