# Architecture guide: understanding this codebase

Date: 2026-09-18

A guided tour for someone who has to work on this repository. It explains what each layer
does, which file does it, and why the design is the way it is.

**Related documents.** `version4.html` and [THEORY.md](THEORY.md) are the mathematical
source of truth; [ESTIMATOR.md](ESTIMATOR.md) has the numerical detail;
[PIPELINE_TEXTBOOK.md](PIPELINE_TEXTBOOK.md) tells the estimation story end to end;
[CODEBASE_ARCHITECTURE.md](CODEBASE_ARCHITECTURE.md) is the older, deeper implementation
audit; [RUNNING_THE_CODEBASE.md](RUNNING_THE_CODEBASE.md) is the operating manual. This
guide is the map that ties them together.

**Suggested reading order.** §1 → §2 → §3, then follow whichever layer you must change.

---

## 1. What the system is

One probability law over shopping baskets, fitted exactly, then used for everything.

For a trip context x (household, store, week, prices, what is in stock) and a basket S of
1..nmax products:

$$
p(S \mid x) = \frac{\exp E(S, x)}{Z_+(x)},\qquad
E(S,x) = \underbrace{\sum_{j \in S} b_j(x)}_{\text{appeal}}
       + \underbrace{\sum_{j<k \in S} \phi_j^\top \phi_k}_{\text{goes together}}
       - \underbrace{\sum_c \rho_c \binom{n_c}{2}}_{\text{substitutes}}
       - \underbrace{\rho_0(|S|)}_{\text{basket size}}
$$

Everything the system offers — recommendations, basket completion, generation, price
scenarios, segments — is a query against this one law. There is no separate recommender.

Three properties drive the whole architecture:

1. **The normalizer is exact.** Z₊(x) sums over every possible basket, computed by a
   dynamic program plus deterministic quadrature, not by sampling. That is what allows
   likelihood claims and per-request numerical certificates.
2. **Evidence gates use.** Every capability the API exposes is backed by an audit that can
   fail. Failures stop the pipeline or refuse the request.
3. **Identity is content-addressed.** Data bundles, checkpoints and reports carry hashes,
   and a mismatch is an error rather than a warning.

## 2. The layers

```text
 raw data ──adapter──▶ canonical directory ──prepare_model_bundle──▶ model bundle
                                                                          │
                                                                    run_pipeline
                            ┌───────────────────────────────────────────┘
                            ▼
  initialize ─▶ price evidence ─▶ additive fit ─▶ rank ─▶ interaction ─▶ size ─▶ audits
                            │                                                     │
                            ▼                                                     ▼
                     checkpoints (.pt)                                    reports (.json)
                            │                                                     │
                            └───────────────▶ retail API ◀────────────────────────┘
```

| Layer | Where | Role |
|---|---|---|
| Adapters | `scripts/data/`, `scripts/version4/external_basket.py`, `scripts/synthetic/` | turn a source into the canonical directory |
| Input contract | `scripts/version4/canonical_contract.py` | validate that directory |
| Bundle preparation | `scripts/prepare_model_bundle.py`, `canonical_basket_input.py`, partition modules | build the model's own files, groups, index and fingerprint |
| Feature and batch | `features.py`, `fit.py`, `data.py` | gather conditioning at every offered slot |
| Model core | `ragged.py`, `poly_degree_native.*`, `conditional_basket.py`, `price_response.py` | the law, its normalizer, and exact conditional queries |
| Estimation stages | `fit_*.py`, `build_spectral_phi_initialization.py` | fit the parameters in a fixed order |
| Audits | `audit_*.py`, `eval_*.py`, `compare_*.py`, `diagnose_*.py` | certify or refuse |
| Orchestration | `scripts/run_pipeline.py`, `run_manifest.py`, `provenance.py` | run stages, record identity, fail closed |
| Application | `retail_api/` | serve only what the audits allow |
| Verification and research | `scripts/verification/`, `scripts/synthetic/evaluate_capabilities.py` | prove the code is right; study the data |

## 3. The model in code

| Term in the energy | Parameter | Where it is built |
|---|---|---|
| Product appeal b_j(x) | `lam` (product), `alpha`·`theta` (household taste), price and promotion coefficients, availability offset | `RaggedModel.b_at` in `ragged.py`, features from `features.py` |
| "Goes together" φ_jᵀφ_k | `phi` | `ragged.py`; fitted in the interaction stage |
| "Substitutes" ρ_c | `rho_c` | `ragged.py`; groups come from the bundle's partition |
| Basket-size curve ρ₀(n) | `rho_0_free`, plus a household rank-one direction | `ragged.py`, `fit_household_size_rank1.py` |
| Normalizer Z₊(x) | — | `RaggedModel.log_Z` → `_log_Z_quad` (Smolyak) and the native dynamic program |

**Why "ragged".** Stores carry different products, so trips have different assortments.
Rather than padding every trip to the full catalogue, `RaggedIndex` (in `ragged.py`, built
by `data.py`) stores one flat array of offered slots with row pointers, and every
computation walks those slots. This is the single most important data structure in the
repository.

**The two normalizer paths.**
- **Exact quadrature (used by every certified stage).** The inner sum over baskets is a
  closed-form elementary-symmetric-polynomial recursion per category, combined by a
  category/size dynamic program; the outer Gaussian integral over the interaction variable
  z is a deterministic Smolyak rule. `smolyak_grid` and `_log_Z_quad` in `ragged.py`, installed by `pipeline_support.install_quadrature`.
- **Importance sampling** is the fallback when no quadrature rule is installed. It is not
  used for certified numbers: verified against enumeration, 4,096 draws were off by tens of
  nats.

**The C++ extension.** `poly_degree_native.py` wraps a compiled degree-aware product
(`setup_poly_degree_native.py` builds it into git-ignored `artifacts/native/`). It speeds up
the audit oracles; a fresh clone without it skips 11 tests with the build command in the
message.

**Conditional queries.** `conditional_basket.py` turns "the cart already contains R" into a
smaller problem: revealed items shift the utilities of the rest, fix category counts and
shrink the size budget, after which the same dynamic program gives exact completion
probabilities. `price_response.py` applies a price change to a context and updates the
assortment mean. Both are what the API calls.

## 4. Data layer

### 4.1 Three ways in

| Source | Adapter | Notes |
|---|---|---|
| Dunnhumby *The Complete Journey* | `scripts/data/01_build_base.py`, `22_basket_data.py`, `23_promo_data.py`, audited by `audit_preprocessing.py` | predates the canonical contract; writes `basket_input/` at the repository root |
| ERIM public archives | `scripts/data/fetch_erim_categories.py` → `scripts/build_erim_basket_dataset.py` (using `external_basket.py`) | verifies archive hashes, then writes a canonical directory |
| Synthetic world | `scripts/synthetic/generate_canonical_world.py` (law in `basket_world.py`) | known truth, written independently of the pipeline |

### 4.2 The canonical directory

`canonical_contract.py` defines the dataset-neutral input: `transactions.parquet`,
`products.parquet`, `store_week_prices.parquet`, `promotions.parquet`, optional
`shopping_opportunities.parquet`, and `build_audit.json`. It validates identifiers, splits,
price sources, training support and the period/day limits. Full column tables are in
[CANONICAL_INPUT_CONTRACT.md](CANONICAL_INPUT_CONTRACT.md).

### 4.3 Bundle preparation

`scripts/prepare_model_bundle.py` is the one command per dataset. It:

1. validates the canonical directory (after merging an optional `product_metadata` file);
2. builds model files with `canonical_basket_input.py`: `items.parquet`, `baskets.parquet`,
   price panels, promotion panel, state, and the availability panel;
3. writes the **product partition** (§4.4);
4. builds the ragged index (`data.py --force`);
5. writes the content fingerprint (`provenance.py`).

**Availability.** With `availability.rule = retail_first_sale`, the builder infers when each
product first appeared in each store from the independent sales feed, and the model adds a
log-availability offset so products not yet stocked are nearly impossible rather than merely
unlikely.

### 4.4 Product partitions (which products share ρ_c)

| Rule | Module | Summary |
|---|---|---|
| `affinity` | `build_affinity_partition.py` | co-purchase groups; cannot express substitution |
| `category` | `prepare_model_bundle.write_category_partition` | one group per category; the ERIM standard |
| `finest_catalogue_level`, `catalogue_hierarchy` | `catalogue_partition.py` | declared catalogue levels, keyed by full path, with a partition check |
| `substitution_evidence` | `evidence_partition.py` | groups from household-level co-purchase shortfall, merged under constraints |

All are model-free and fixed before training. Rationale and measurements:
[PARTITION_OPTIONS_AND_PARKED_WORK.md](PARTITION_OPTIONS_AND_PARKED_WORK.md).

### 4.5 Identity

`provenance.py` hashes the partition, manifests, metadata, ragged index and optional panels
into one `model_data_fingerprint.json`. Every checkpoint stores that fingerprint, and
`checkpoint_io.load_checkpoint` (in `scripts/version4/`) refuses a checkpoint whose fingerprint does not match the
bundle in `ENERGY_MODEL_DATA_ROOT`. This is why moving or rebuilding a bundle detaches its
checkpoints.

## 5. Feature and batch layer

- **`features.py`** gathers conditioning at **every offered slot**, not at purchase lines:
  price deviation from the training mean (dense product × day, plus optional store-week),
  promotion indicators, and the availability offset. Sparse panels are looked up by packed
  integer keys, which is why periods are capped at 127 and days at 1023.
- **`fit.py` `Batcher`** turns a list of trips into the ragged index slice plus the context
  dictionaries the model needs, so the energy and the normalizer see identical inputs.
- **`data.py`** builds and caches the ragged index (`v3_index_affinity.npz`), and reports
  the assortment shape. `V3_PARTITION` can point at any `(item_id, cat_id)` parquet, so a
  partition can be swapped without code changes; the partition sets the number of groups,
  and checkpoints across partitions are not comparable.

## 6. Estimation stages, in pipeline order

`scripts/run_pipeline.py` runs these as separate processes, recording each in a manifest.
`--dry-run` prints the graph without executing.

| # | Stage | Script | What it does | Failure behaviour |
|---|---|---|---|---|
| 1 | native build | `setup_poly_degree_native.py` | compiles the C++ helper | stops |
| 2 | initialize | `initialize_version4.py` | reproducible untrained artifact; validates the partition manifest | stops |
| 3 | price evidence | `fit_supported_price_response.py` | smallest price response the same-store evidence supports; ERIM instead uses the joint route (`fit_joint_price_utility.py` + `audit_joint_price_alignment.py`, passed in with `--price-coefficients`) | declares "no support" and continues with zero price response |
| 4 | additive fit | `fit_exact_additive.py` | exact maximum likelihood of the φ = 0 model, with the category safety projection (`category_safety.py`) | convergence gate |
| 5 | rank | `build_spectral_phi_initialization.py` | split-half spectral pass choosing the largest stable interaction rank | rejects ranks, keeps the audit |
| 6 | interaction | `fit_stratified_natural_interactions.py` with `stratified_natural.py`, `interaction_particles.py`, `tempered_block_gibbs.py` | size-stratified natural-parameter Monte Carlo likelihood for φ and a smooth ρ₀ correction | cross-fit and ESS gates |
| 7 | household size | `fit_household_size_rank1.py` | one concave rank-one recalibration, capped by the tail screen | cross-fit gate |
| 8+ | audits | §7 | likelihood, recommendation, generation, segments, embeddings, population, size phase, promotion MDP | full profile exits nonzero on failure |

**Why interaction is a separate stage.** With φ = 0 the normalizer is exact and cheap, so
the additive parent is fitted exactly first. The interaction term is then estimated in a
natural-parameter space where the objective is concave given fixed draws, instead of
running stochastic gradients through the whole law.

## 7. Audits and reports

| Report | Producer | Certifies |
|---|---|---|
| `likelihood_{validation,test}.json` | `compare_rank8_parent_likelihood.py` | paired complete-support likelihood against the φ = 0 parent |
| `recommendation.json` | `eval_smolyak_rank8_mrr.py` | exact add-one ranking (MRR, recall); no normalizer needed |
| `generation_counterfactual.json` | `audit_particle_counterfactual_generation.py` | sampler validity, price scenarios, ESS |
| `customer_segments.json` | `audit_customer_segments.py` | descriptive segments and their price surfaces |
| `interaction_embedding_audit.{json,md}` | `audit_interaction_embeddings.py` | invariant pair scores and a held-out co-incidence check |
| `population_size.json` | `audit_population_size.py` | full-population size and tail safety, resumable |
| `size_phase_diagnostic.json` | `diagnose_size_phase.py` | splits size log-odds into catalogue pressure and ρ₀ |
| `segment_promotion_mdp.json` | `run_segment_pricing_mdp.py` | budget-constrained promotion policy, with an independent frozen-policy evaluation |
| `real_basket_completion_corrected.json` | `audit_basket_completion_corrected.py` | the API's completion capability, and the certified quadrature levels |

Supporting theory audits (`audit_probability_foundations.py`,
`audit_original_probability.py`, `audit_cart_conditionals_quadrature.py`,
`audit_counterfactual_query_synthetic.py`, `audit_synthetic_interactions.py`) check the
implementation against brute-force enumeration on small worlds.

`uncertainty.py` provides the household-cluster-robust paired intervals used across
reports; `tempered_ais.py` provides annealed SMC for contexts where a scalar normalizer is
otherwise hard; `adaptive_sparse.py` holds the sparse quadrature construction.

## 8. The retail API

```text
request → schema (strict) → service → context → exact conditional computation
                                   ↘ evidence gates ↘ numerical certificate → response
```

| Module | Role |
|---|---|
| `retail_api/schemas.py` | strict request and response contracts; unknown fields are rejected |
| `retail_api/app.py` | FastAPI routes and the error handler |
| `retail_api/service.py` | loads the checkpoint (via `checkpoint_io.load_checkpoint`), audit, segments and capability verdicts; resolves contexts; computes answers |
| `retail_api/runtime.py` | selects the served bundle before model modules import |
| `retail_api/errors.py` | the single error type with a code and message |

**Startup gates:** checkpoint and bundle fingerprints must agree; the completion audit must
have passed and belong to this checkpoint; segment assignments must match their digest; and
the quadrature levels come from the audit rather than a fixed offset.

**Per-request gate:** every basket answer is computed with two adjacent quadrature rules;
if they disagree beyond tolerance the service escalates once, then refuses with HTTP 503.

**Capability verdicts.** Decision capabilities (price, promotion, assortment, and so on) are
refused by default. A deployment may claim one only through a verdict file bound to that
checkpoint and fingerprint, carrying evidence and a source. That is what lets the synthetic
deployment serve `POST /v1/baskets/price_scenario` while ERIM refuses it. See
[RETAIL_APPLICATION_API.md](RETAIL_APPLICATION_API.md) and
[SYNTHETIC_RETAIL_API_APPLICATIONS.md](SYNTHETIC_RETAIL_API_APPLICATIONS.md).

## 9. Verification and research

| Area | Entry point | Purpose |
|---|---|---|
| Known-truth capabilities | `scripts/synthetic/evaluate_capabilities.py` | scores a run against the generating law, and emits capability verdicts |
| API applications | `scripts/verification/api/synthetic_api_applications.py` | drives every endpoint and scores answers against truth |
| Partition diagnostics | `scripts/verification/partition/check_evidence_partition.py` | out-of-sample shortfall, stability, truth recovery |
| ERIM structure research | `scripts/verification/nested_rho/{substructure,promotion,type}_check.py` | do categories hide sub-groups? are promotions the cause? |
| Nested-model prototypes | `scripts/verification/nested_rho/verify_*.py`, `decision_experiment.py` | exactness and recovery for a model extension that is **not** implemented |
| Causal evidence on ERIM | `scripts/run_erim_coupon_experiment.py`, `run_erim_tissue_*` | a randomized coupon lottery and model prediction tests |
| Baselines | `scripts/run_baselines.py`, `baselines.py`, `baselines2.py`, `train_baseline_verified.py` | DPP, non-symmetric DPP, SHOPPER-style and BEMB-style comparisons on the same support |
| External price validation | `external_choice*.py`, `scripts/run_external_price_validation.py` | held-out choice validation on outside datasets |

## 10. Cross-cutting rules

- **Fail closed.** Gates stop the pipeline (nonzero exit, artifacts kept) or refuse the
  request. Nothing degrades silently to a weaker estimate.
- **Source freeze.** A run hashes `scripts/**/*.py`, `*.cpp` and `tests/*.py` at startup and
  aborts if they change mid-run.
- **Test data selects nothing.** Model and setting choices use training and validation only.
- **Determinism.** Seeds, fixed draw banks and content digests make stages reproducible; the
  interaction bank is reused only when its digests match.
- **Hardware policy.** `runtime_capabilities.py` picks the backend and a thread count;
  CUDA/MPS are refused for the certified likelihood rather than silently swapped in.
- **The model is frozen.** Partitions and inputs may change; the energy's form may not,
  without an explicit decision. See
  [NESTED_SUBSTITUTION_GROUPS.md](NESTED_SUBSTITUTION_GROUPS.md).

## 11. Tests

38 files, about 4,200 lines, run in about 35 seconds. Four kinds:

1. **Exactness against enumeration** — normalizer, conditionals, samplers, price responses
   on small worlds.
2. **Contracts** — canonical input, partitions, bundle configs, checkpoint lineage, API
   request and response shapes.
3. **Estimator properties** — concavity, projections, safety caps, cross-fit behaviour.
4. **Integration** — small end-to-end paths, including the API against a fake service.

Tests needing the compiled extension skip with the build command when it is absent.

## 12. Where to make common changes

| Goal | Touch | Do not touch |
|---|---|---|
| Add a dataset | an adapter, a config in `configs/datasets/` | model code |
| Change product groups | a partition rule in `catalogue_partition.py` or `evidence_partition.py`, plus the config | `ragged.py` |
| Add a feature to product appeal | `features.py`, `RaggedModel.b_at`, the bundle builder | the normalizer's structure |
| Add an API endpoint | `retail_api/schemas.py`, `service.py`, `app.py`, tests | evidence gates (extend them, don't bypass) |
| Claim a new decision capability | a verdict file produced by an evaluation with evidence | the default refusals |
| Change the energy's form | a deliberate decision first; then `ragged.py`, all conditional and sampling paths, every audit, and a full re-validation | — |

## 13. Glossary

| Term | Meaning |
|---|---|
| **Assortment slot** | one (trip, offered product) pair; the unit every computation walks |
| **Ragged index** | flat slot arrays plus row pointers, replacing a padded trip × product matrix |
| **Partition / groups** | the products that share one substitution penalty ρ_c |
| **Bundle** | a prepared model-data directory plus its fingerprint |
| **Checkpoint** | fitted parameters plus lineage (`candidate_rank1.pt`) |
| **Gate** | a check that can stop the pipeline or refuse a request |
| **Verdict** | a per-deployment claim that a decision capability is supported, with evidence |
| **Certificate** | the per-request agreement between two adjacent quadrature rules |
| **Smoke profile** | the same stage graph with tiny statistical panels: a software check, not a result |
