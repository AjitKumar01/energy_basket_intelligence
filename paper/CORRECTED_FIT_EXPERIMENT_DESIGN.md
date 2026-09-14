# Corrected synthetic verification and fresh original-data fit

## Changes and declared interpretation

The preceding audit found three different issues. They are not interchangeable:
historical original-data tail-band importance sampling had low absolute ESS;
the original-data generation panel was undersized; and the complete synthetic
retailer experiment overpredicted selected-policy profit.

The corrected synthetic experiment:

- Profiles basket-size potentials against training counts after selecting the
  other basket parameters. This is a convex likelihood update, not modification
  of observed baskets or matching to test/oracle moments. It preserves the
  within-size odds of every pair of baskets.
- Separates exact fitted-law moments, observed held-out moments and generated
  moments. It uses 32 independent generation replicates, reports sampling SE and
  household-clustered observed-minus-model uncertainty, and checks mean and size
  distribution fidelity rather than interpreting one noisy draw as model bias.
- Pools arrival response using observed offer, discount-depth and mission/segment
  alignment features. It no longer fits an independent effect to every sparse
  segment/action cell. No oracle response coefficient is supplied to this fit.
- Uses 1,200 customers rather than 240 in the full retailer profile, or 216,000
  randomized opportunities. Changes in results therefore reflect both estimation
  changes and a larger design; they cannot all be attributed to one code fix.
- Freezes policy selection on validation covariates, derives its budget without
  oracle costs, allows no action rather than forcing budget utilization, and
  reserves 10% of predicted markdown costs. That reserve is an engineering
  precaution, not a probabilistic cost guarantee.
- Evaluates the frozen policy using held-out augmented inverse-propensity-weighted
  profit contrasts, including no-purchase rows and household-clustered uncertainty.
  Wide intervals remain wide; a positive plug-in estimate is not proof of profit.
- Adds an enumerated tail-band recovery stress test. A pilot chooses a larger
  draw budget; a separate bank is tested against absolute/fractional ESS and
  the known normalizer. Original-data banks must still pass their own gates.

The full suite includes the existing probability foundation and interaction
recovery stages and six retailer runs: the original well-specified/misspecified
seeds plus both worlds at seeds 93021 and 113021. The latter pairs use the same
seed within each pair. The default base seed is 73021.

## Full-retailer acceptance gates

These are finite-world engineering tolerances, not universal statistical proofs:

1. Generated mean lies within five model-based Monte Carlo SEs of the fitted law.
2. Held-out observed-minus-model mean is within four household-cluster SEs plus
   0.02 items.
3. Fitted versus oracle average size bias is at most 0.10 items.
4. Selected-policy plug-in profit overprediction is at most 25% of oracle value.
5. Oracle expected spend for the selected policy is within its budget.
6. The selected policy has positive oracle value in the synthetic world.

Every full-retailer stage must pass these gates for the driver to complete.
The independent profit interval is reported separately. Its lower endpoint may
be negative even when synthetic oracle value is positive; this explicitly means
the logged holdout sample has not established a positive value statistically.
Expected-budget feasibility also does not guarantee a realized stochastic budget.

The pricing exercise is a fixed-context repeated allocation, not a dynamic
recency-transition oracle. No causal or quantity capabilities are inferred for
the original-data model from these synthetic additions.

### Same-budget development control

As an auxiliary control, the corrected retailer code was also run with
`Config(customers=240, threads=4, seed=73021, world="well_specified")` and
`Config(customers=240, threads=4, seed=73122, world="misspecified")`, retaining
the historical number of opportunities. Both selected policies had zero oracle
expected-budget violation. Their plug-in/oracle selected-policy profits were
194.77/166.61 and 175.83/171.88, respectively: overprediction of 16.9% and 2.3%,
versus approximately 149% and 194% in the historical reports. This is a joint
correction comparison, not an ablation attributing improvement to one component.
The corrected control was run through `audit_synthetic_retailer.run`; the final
hashed pipeline remains the primary reproducible acceptance artifact.

At that small budget, the misspecified control's observed mean size remained
3.182 versus generated 3.322. More generation replicates do not remove observed
sampling noise or model misspecification. The full experiment's exact moments,
larger design and explicit calibration gates are consequently still necessary.

## Fresh original-data fit

Use a new run directory and a fresh initialization, not continuation of the old
proxy optimizer. The common/relative price specification remains unchanged. The
existing target -0.121 is retained as a calibration assumption, not represented
as an identified causal elasticity.

The driver supports isolated `artifacts/`, `out/`, and `reports/`, together with a
timestamped invocation directory containing the aggregate log, stage logs, source
hashes and an atomic status manifest. Resumed non-data stages verify and reuse
the audited inputs instead of reconstructing shared historical data artifacts.

```sh
python -u scripts/run_pipeline.py --profile full --threads 4 \
  --run-dir artifacts/corrected_fit_full_20260913 --start-at initialize
```

The run must retain early-stopping, cross-fit, numerical, ESS and population
calibration checks. Launching it does not mean it has passed those checks.
The full original-data draw allocation is now 16/16/12/8/16/16/16 across the
seven size bands (100 draws per context); both ESS gates remain active. These
larger tail budgets do not guarantee passing the measured gates.

## Recommendation evaluation

Synthetic reports include additive and interaction MRR and Recall@5/10 on the
same hidden-item cases. The original-data pipeline uses exact conditional
add-one energy over the contemporaneous assortment, so a normalizer is not
needed for that ranking task.

The corrected original-data evaluator additionally compares the full model to
the separately fitted additive parent on exactly the same trip/hidden-item
manifest. It retains popularity and within-model ablations, reports MRR and
Recall@5/10/20/100, and uses household-clustered paired uncertainty for the
full-minus-parent gains. Per-case ranks and hidden-item IDs are saved in the
report for replay. This prevents confusing a frozen-parameter no-interaction
ablation with an independently fitted additive baseline.
