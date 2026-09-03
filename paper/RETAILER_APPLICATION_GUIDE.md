# What the current model can do for a retailer

## Short answer

With the data currently available, this model should be sold as a **basket-intelligence
and campaign-planning application**.

It can help a retailer understand:

- what customers tend to buy together;
- which products are sensible recommendations for an existing basket;
- how baskets differ across customer groups;
- which products and bundles appear sensitive to price changes; and
- which promotions are worth shortlisting for a controlled test.

It should **not yet** be sold as an autonomous pricing engine or a complete digital twin.
The current real data records purchases, but it does not fully describe non-purchase
opportunities, inventory, product costs or randomized treatment effects.

---

## 1. What probability the model learns

The main model learns

\[
p(S\mid x,\text{a purchase trip occurred}),
\]

where $S$ is the basket and $x$ contains the known customer, product, price and context
information.

In simple terms, it answers:

> Given that this customer is shopping under this context, what basket are they likely
> to purchase?

It does not currently estimate the complete probability

\[
p(\text{customer shops})
\times
p(S\mid\text{customer shops}).
\]

Therefore, it is much stronger at predicting **basket composition** than predicting
whether a customer will visit the retailer at all.

---

## 2. How to interpret the current results

### Basket likelihood

On the locked test data, the interaction model improves over its exact additive version
by

\[
0.03275\pm0.00239
\]

nats per basket. This is statistically clear and is much larger than the measured
numerical-integration error.

This means the interaction model assigns systematically better probability to baskets
that customers actually purchased. It is evidence that basket relationships contain
useful information beyond individual product popularity, household preference and basket
size alone.

The model also beats the three external likelihood baselines on identical test baskets:

| Baseline | Model's gain in nats per basket |
|---|---:|
| Bernoulli | $2.25204\pm0.09842$ |
| DPP | $2.26164\pm0.09767$ |
| NDPP | $1.81225\pm0.09239$ |

These are strong probability-model comparisons. They do not automatically imply the
same-sized improvement in revenue.

### Recommendation

The current test MRR is

\[
0.09525\pm0.00607.
\]

MRR measures how highly the model ranks a product hidden from a real basket. It is not a
9.5% accuracy figure. Higher is better, but the business-facing application should also
report Recall@5, Recall@10, catalogue coverage and performance by customer segment.

The full model recommends meaningfully, but the recommendation improvement attributable
only to interactions is not yet statistically conclusive. The strongest current evidence
for interactions is the normalized basket likelihood and the held-out co-incidence audit.

### Interaction embeddings

The strongest learned cross-affinity product pairs occur together 1.216 times as often as
expected under a frequency-and-basket-size matched null. Matched control pairs have lift
0.998.

This supports the presence of aggregate interaction information. It does not prove that
every high-scoring pair is a causal complement. A retailer should treat these pairs as
**bundle and merchandising candidates to test**, not guaranteed promotion effects.

### Basket generation

The model no longer creates the previously observed extreme large-basket failure. The
complete population-tail certification passes.

However, generated baskets remain too small and insufficiently varied:

| Statistic | Observed baskets | Generated baskets |
|---|---:|---:|
| Mean size | 10.03 | 7.34 |
| Size variance | 136.28 | 63.83 |

Generation is therefore suitable for demonstrations, qualitative scenarios and model
checking. It is not yet accurate enough for inventory planning or revenue forecasts.

---

## 3. What the retailer can use immediately

### A. Basket completion and cross-sell recommendations

For a basket already in progress, the application can rank products that are most
compatible with its contents and the customer's context.

Examples include:

- checkout recommendations;
- “frequently bought together” suggestions;
- personalized coupons for the next trip; and
- associate-facing recommendations in assisted sales.

The retailer should evaluate these suggestions with Recall@K, catalogue coverage,
incremental basket value and an online A/B test.

### B. Bundle discovery

The interaction matrix can identify product pairs and small groups whose relationship is
not explained by popularity alone.

The application can produce a shortlist containing:

- the products in the candidate bundle;
- the learned interaction score;
- observed co-purchase support;
- lift against the matched null; and
- the customer segments in which the relationship is strongest.

This can guide bundle design, adjacent placement, email campaigns and coupon tests.

### C. Customer-segment summaries

The model can group customers using their learned preference and basket patterns. For each
segment, the retailer can receive:

- common products and categories;
- typical basket size and composition;
- products that distinguish the segment from the population;
- apparent price sensitivity; and
- candidate bundles or recommendations.

These are behavioural descriptions, not permanent customer identities. Segment quality
should be monitored because customer behaviour changes over time.

### D. Price and promotion scenario screening

The model can change a product's price input and recompute the conditional basket law. It
can then show estimated changes in:

- product incidence;
- expected basket size;
- related-product incidence; and
- the relative ranking of candidate promotions.

This is useful for **screening** many ideas before running an experiment. With the current
observational data, it should not be presented as a causal revenue forecast. The safest
output is a ranked shortlist with uncertainty, followed by a controlled retailer test.

### E. Assortment and merchandising diagnostics

The model can identify:

- products that act as anchors for many baskets;
- products with narrow or broad interaction neighbourhoods;
- categories whose products frequently appear together;
- weakly connected products that may need different placement; and
- customer groups for which an assortment is poorly represented.

Removing a product from an assortment is a stronger intervention than changing a model
input, so assortment decisions still require availability and substitution data.

---

## 4. What the application should look like

A practical first version can contain five views.

1. **Customer or segment profile** — preferred products, typical baskets and price
   sensitivity.
2. **Basket recommender** — a partial basket enters; ranked products and reasons come out.
3. **Bundle explorer** — supported complements, co-purchase evidence and segment lift.
4. **Scenario lab** — proposed prices or discounts enter; predicted basket changes and
   uncertainty come out.
5. **Model-health page** — held-out likelihood, recommendation metrics, generation
   calibration and warnings about unsupported contexts.

The application should always label outputs as one of:

- **descriptive** — learned from historical behaviour;
- **predictive** — checked on held-out data; or
- **causal** — supported by randomized or otherwise identified treatment evidence.

Most current outputs are descriptive or predictive.

---

## 5. A sensible retailer workflow

### Step 1: Data audit

Check product identifiers, customer identifiers, trip construction, prices, promotion
coverage, quantities and time splits. Report missing or unsupported fields before model
training.

### Step 2: Train and certify

Train from scratch, select model complexity using validation data and report all final
metrics once on the locked test period.

### Step 3: Deliver insights

Provide recommendations, segments, candidate bundles and price-scenario rankings through
the application.

### Step 4: Select a small intervention set

Choose a limited number of high-support, low-risk recommendations or promotions. Avoid
actions based on rare products or poorly represented customer contexts.

### Step 5: Run a randomized pilot

Randomly assign the shortlisted interventions, retain a control group and record exposure,
purchase, quantity, margin and inventory outcomes.

### Step 6: Calibrate business value

Use the pilot to estimate incremental profit and update the model. Only after this stage
should the retailer use monetary policy-value estimates for budget allocation.

---

## 6. Data needed for the complete application

The current transaction data is enough for basket intelligence. A realistic retailer
simulator additionally needs:

| Required data | Why it is needed |
|---|---|
| Customer opportunities or visits, including no-purchase outcomes | Estimate whether a purchase occurs |
| Logged offer exposure and assignment propensity | Estimate causal promotion response |
| Product cost and margin | Optimize profit rather than sales |
| Inventory and availability | Avoid treating stockouts as customer rejection |
| Quantities and returns | Model units and realized value |
| Promotion duration and carryover | Represent sequential effects |
| Loyalty and retention outcomes | Measure long-term customer impact |

Once these are available, the same basket model can serve as the basket-composition
component inside a larger retailer simulator.

---

## 7. Recommended commercial position

The honest current product statement is:

> The application learns a normalized, customer-aware distribution over retail baskets.
> It improves held-out basket likelihood, supports contextual recommendations, discovers
> testable product relationships and screens price or promotion scenarios.

The application should not yet claim:

- guaranteed causal complements;
- exact promotion ROI;
- autonomous price optimization;
- reliable inventory simulation; or
- a complete probability of customer purchase.

The immediate commercial value is reducing the retailer's search space: instead of
testing thousands of arbitrary bundles or promotions, the retailer can test a small set
of model-supported candidates. The next source of value comes from feeding those test
results back into the system so its descriptive basket intelligence becomes calibrated
decision intelligence.

