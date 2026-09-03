# Segment-targeted promotion-budget MDP

## 1. Decision scope

The retailer has a fixed markdown budget and a promotion that lasts a fixed number of
days. The Version-4 basket law is used as the demand environment. It remains conditional
on a shopping trip, so this exercise estimates basket response among arriving shoppers;
it does not estimate whether a discount causes a customer to visit the store.

The current experiment uses a 28-day horizon and the three locked customer segments.
The action design is based only on training outcomes. Held-out test contexts are used to
estimate action response.

The underlying sampling, counterfactual density-ratio, segment-distance,
Bellman-optimality and budget-rounding propositions are derived in
[INFERENCE_AND_SIMULATION.md](INFERENCE_AND_SIMULATION.md). This document specializes
those results to the measured three-segment experiment.

## 2. State and action spaces

The state on promotion day \(t\) is

\[
s_t=(\tau_t,B_t),
\]

where \(\tau_t\) is the number of promotion days remaining and \(B_t\) is remaining
expected markdown budget. Budget is represented on a 4,000-cell grid.

An action is either no promotion or

\[
a=(g,k,d),
\]

where \(g\in\{0,1,2\}\) is the target customer segment, \(k\) is one of three
five-product bundles for that segment, and \(d\in\{0.10,0.20\}\) is the discount.
Only one segment-specific bundle is promoted per day in this first operationally small
action space.

For each segment, candidate categories are selected by training-only evidence-weighted
over-indexing. The five most frequent training SKUs in each selected category form the
bundle. No validation or test outcome is used to choose an action.

## 3. Model-implied action response

For a promoted set \(A\), the price intervention is

\[
\log p'_j=\log p_j+\log(1-d)\,\mathbf 1\{j\in A\}.
\]

If \(S^{(m)}\) is a factual particle, the exact energy likelihood-ratio weight is

\[
w_m(a)\propto
\exp\left\{
E_{p'}(S^{(m)},x)-E_p(S^{(m)},x)
\right\}.
\]

The same factual particles evaluate every action. This gives paired counterfactuals and
avoids rerunning SMC for every discount. Actions with minimum normalized reweighting ESS
below 0.2 or a context with \(P(N\ge60)\ge0.5\) are removed before policy optimization.

Let \(q_{jgk}(d)\) be the reweighted incidence probability of SKU \(j\). Expected
markdown spend per eligible trip is

\[
c_{gkd}
=
d\sum_{j\in A_{gk}}p_jq_{jgk}(d).
\]

The basket-value response is evaluated at undiscounted shelf prices,

\[
R_{gkd}
=
\sum_jp_jq_{jgk}(d)-\sum_jp_jq_{jg}(0).
\]

This separates demand created by the action from the budget used to create it. Actual
post-discount sales are reported as

\[
R^{\mathrm{net}}_{gkd}=R_{gkd}-c_{gkd}.
\]

These are distinct-SKU sales: the basket law generates incidence, not units. They are not
gross margin or profit because wholesale costs are unavailable.

## 4. Finite-horizon Bellman recursion

Segment traffic is scaled by its training-trip share. Therefore each action has an
expected daily cost \(C_a\), conservative daily reward \(L_a\), and post-discount sales
response \(G_a\). The policy objective uses the paired 95% lower confidence bound

\[
L_a=\widehat R_a-1.96\,\operatorname{se}(\widehat R_a),
\]

rather than optimizing raw sample means over many candidate actions.

With discretized cost \(\widetilde C_a\), the Bellman recursion is

\[
V_\tau(b)
=
\max_{a:\widetilde C_a\le b}
\left\{L_a+V_{\tau-1}(b-\widetilde C_a)\right\}.
\]

At the terminal day,

\[
V_0(b)=
\begin{cases}
0,&b\le(1-u)B,\\
-\infty,&\text{otherwise},
\end{cases}
\]

with required utilization \(u=0.95\). Positive action costs are rounded upward, which
prevents overspend. The terminal budget threshold is tightened by one grid cell per
promotion day, ensuring that accumulated rounding cannot falsely certify 95% utilization.

## 5. Evaluated environment

- Horizon: 28 promotion days.
- Contexts: 64 distinct-household-balanced test contexts per segment.
- Particles: 32 per context with 17 annealing levels.
- Actions: 18 segment/bundle/discount actions plus no promotion.
- Expected traffic: 309.49 modeled trips per day.
- Baseline distinct-SKU basket value: 4,716.11 price units per day.
- Minimum factual SMC ESS: 0.9970.

The budgets are expressed as 25%, 50%, and 75% of the largest spend achievable under the
restricted one-bundle-per-day action space. They equal only 0.026%, 0.053%, and 0.079% of
baseline 28-day basket value. A real budget outside this range requires a supplied currency
amount and an expanded concurrent-action space.

## 6. Segment response

Segment 0 is the tropical-fruit/soft-drink segment. Its reliable positive action is the
tropical-fruit bundle. A 20% discount gives

\[
\Delta N=0.1244\pm0.0284,
\qquad
\Delta R=0.2179\pm0.0519
\]

per segment trip, with lower 95% basket-value bound 0.1162 and reweighting ESS 0.9766.

Segment 1 is the milk/checklane-candy segment and contains the strongest supported
responses. A 10% milk discount gives

\[
\Delta N=0.5026\pm0.0858,
\qquad
\Delta R=0.9764\pm0.1511,
\]

with lower 95% bound 0.6803 and ESS 0.5758. The 20% milk action has a strong raw response
but is excluded because its minimum ESS is 0.1582, below the declared 0.2 threshold.
Checkout candy at 10% and 20% has smaller but positive conservative value.

Segment 2 receives no promotion in the optimized policies. Every evaluated refrigerated,
organic produce, and baby-food bundle has a negative paired basket-size and basket-value
response. In this conditional basket law, the promoted products weakly substitute into
smaller baskets for these contexts. This is a model implication to validate experimentally,
not a causal claim.

## 7. Budget policies

| Budget | Expected utilization | Conservative value | Mean basket value | Post-discount sales | Day allocation |
|---:|---:|---:|---:|---:|---|
| 34.81 | 99.40% | 358.50 | 873.53 | 838.93 | S1 milk 10%: 6; S1 candy 10%: 8; S1 candy 20%: 14 |
| 69.62 | 98.93% | 693.65 | 1,315.91 | 1,247.03 | S1 milk 10%: 13; S1 candy 20%: 15 |
| 104.43 | 99.18% | 1,021.53 | 1,598.80 | 1,495.23 | S0 tropical fruit 20%: 2; S1 milk 10%: 20; S1 candy 20%: 6 |

All values are in the input price panel's currency units. The apparently high return per
markdown unit is conditional on a trip and includes complement purchases. It must not be
marketed as incremental profit or campaign ROI until an A/B test validates the response
and the model includes visit probability, units, wholesale costs, and inventory.

## 8. Operational conclusion

The fitted model can serve as a finite-horizon promotion-allocation environment. The
current safe use is policy ranking and experiment design:

1. use the MDP to shortlist segment/product/discount campaigns;
2. randomize a real pilot among the shortlisted actions and controls;
3. estimate trip, quantity, margin, and substitution effects;
4. update the environment with visit, inventory, and cost transitions; and
5. only then optimize deployable profit.

The machine-readable result is `reports/segment_promotion_mdp.json`, and the executable
entry point is `scripts/version4/run_segment_pricing_mdp.py`.
