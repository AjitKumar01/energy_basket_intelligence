# Documentation

Worked sizes quoted in these documents (for example 5,455 products or baskets of up to 120
items) come from the large grocery panel on which the theory was first developed. The theory
and the code are catalogue-independent; the dataset shipped with this repository is the
synthetic stress world (`SYNTHETIC_STRESS_WORLD.md`).

## Reading order

1. **`THEORY.md`**: the Version-4 basket law, the exact normalizer (Gaussian identity +
   elementary symmetric polynomials + Smolyak quadrature), likelihood and gradients, why
   \(\Phi\) cannot be learned from zero by gradient descent, the fitting flow, and how
   recommendation, generation, price counterfactuals and the promotion MDP follow from the
   same law.
2. **`PIPELINE.md`**: the selected pipeline, stage by stage, and why other designs were
   rejected.
3. **`JOINT_REFINEMENT_STAGE.md`**: the optional joint refinement stage, its evidence on exact
   test worlds and on a complete synthetic run, cost, scaling and limits.
4. **`CANONICAL_INPUT_CONTRACT.md`** and **`SYNTHETIC_STRESS_WORLD.md`**: how data enters, and
   the shipped synthetic retailer.
5. **`STAGEWISE_RESURRECTION.md`**: run directories, logs, resuming and moving runs.

## Theory by stage

| Pipeline stage | Theory | Code (`scripts/version4/` unless noted) |
|---|---|---|
| data | `CANONICAL_INPUT_CONTRACT.md`, `SUBSTITUTION_EVIDENCE_PARTITION.md` | `../prepare_model_bundle.py`, `canonical_contract.py`, `canonical_basket_input.py`, `catalogue_partition.py`, `build_affinity_partition.py`, `evidence_partition.py`, `data.py`, `provenance.py` |
| initialize | `THEORY.md` §§2–4 | `initialize_version4.py`, `sparse_artifact.py` |
| additive (price evidence + exact additive fit) | `THEORY.md` §§3, 5–7, 10 (Phase B), `PROBABILITY_FOUNDATIONS_AND_PRICE_COUNTERFACTUALS.md` | `fit_supported_price_response.py`, `fit_exact_additive.py`, `ragged.py`, `features.py`, `fit.py` |
| rank | `THEORY.md` §§9–10 (Phase C) | `build_spectral_phi_initialization.py`, `basket_incidence.py` |
| interaction | `SIZE_STRATIFIED_JOINT_ESTIMATOR.md`, `HOUSEHOLD_SIZE_AUDIT.md` | `fit_stratified_natural_interactions.py`, `stratified_natural.py`, `fit_household_size_rank1.py`, `category_safety.py` |
| refinement (optional) | `JOINT_REFINEMENT_STAGE.md` | `fit_joint_refinement.py`, `joint_refinement.py`, `compiled_backtrack.py`, `tempered_block_gibbs.py` |
| evaluation | `HELDOUT_LIKELIHOOD_NORMALIZER.md`, `ESTIMATOR.md`, `INFERENCE_AND_SIMULATION.md` | `compare_rank8_parent_likelihood.py`, `eval_smolyak_rank8_mrr.py`, `audit_particle_counterfactual_generation.py`, `audit_customer_segments.py`, `audit_interaction_embeddings.py`, `interaction_particles.py`, `tempered_ais.py`, `price_response.py`, `uncertainty.py` |
| certification | `HOUSEHOLD_SIZE_AUDIT.md`, `SEGMENT_PROMOTION_MDP.md` | `audit_population_size.py`, `diagnose_size_phase.py`, `run_segment_pricing_mdp.py` |

Shared numerics: `pipeline_support.py` (quadrature rules and helpers), `adaptive_sparse.py`
(sparse Gaussian quadrature), `poly_degree_native.{cpp,py}` and `setup_poly_degree_native.py`
(the compiled category/size dynamic program), `checkpoint_io.py` (checkpoint loading).
Driver and run bookkeeping: `../run_pipeline.py`, `../run_manifest.py`,
`../runtime_capabilities.py`.

## All documents

| Document | Content |
|---|---|
| `THEORY.md` | End-to-end theory of the model and every pipeline stage |
| `PIPELINE.md` | The selected pipeline and its stages |
| `JOINT_REFINEMENT_STAGE.md` | Joint refinement: method, evidence, cost, limits |
| `PROBABILITY_FOUNDATIONS_AND_PRICE_COUNTERFACTUALS.md` | One probability measure for incidence, size, recommendation and price response |
| `HELDOUT_LIKELIHOOD_NORMALIZER.md` | Held-out basket likelihood and the exact normalizer |
| `ESTIMATOR.md` | The Smolyak normalizer estimator and the alternatives it replaced |
| `SIZE_STRATIFIED_JOINT_ESTIMATOR.md` | Size-stratified Monte Carlo likelihood for the interaction stage |
| `HOUSEHOLD_SIZE_AUDIT.md` | Household basket-size direction and the population tail gate |
| `INFERENCE_AND_SIMULATION.md` | Sampling, incidence, recommendation, counterfactuals and simulation from a fitted model |
| `SEGMENT_PROMOTION_MDP.md` | The segment promotion-budget MDP used in certification |
| `CANONICAL_INPUT_CONTRACT.md` | Canonical input format, dataset configs and partitions |
| `SUBSTITUTION_EVIDENCE_PARTITION.md` | The substitution-evidence partition option |
| `SYNTHETIC_STRESS_WORLD.md` | The shipped synthetic retailer and its causal price truth |
| `STAGEWISE_RESURRECTION.md` | Run directories, logs, recovery and portability |
