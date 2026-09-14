#!/usr/bin/env python3
"""Consolidate every retailer application into an evidence-calibrated audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from version4.provenance import file_sha256, strict_json_dumps


ROOT = Path(__file__).resolve().parents[1]


def load(path):
    path = Path(path).resolve()
    return json.loads(path.read_text()), path


def pct(value):
    return f"{100 * value:.2f}%"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-evaluation", type=Path, required=True)
    parser.add_argument("--quadrature-evaluation", type=Path, required=True)
    parser.add_argument("--completion-evaluation", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    new, new_path = load(args.new_evaluation)
    quadrature, quadrature_path = load(args.quadrature_evaluation)
    completion_audit, completion_path = load(args.completion_evaluation)
    recommendation, recommendation_path = load(
        ROOT / "artifacts/corrected_complete_rank5_20260913/reports/recommendation.json")
    generation, generation_path = load(
        ROOT / "artifacts/remaining_verification_audited_20260914/real_generation_calibration.json")
    price, price_path = load(
        ROOT / "artifacts/remaining_verification_audited_20260914/real_price_response_evaluation.json")
    segments, segments_path = load(
        ROOT / "artifacts/corrected_complete_rank5_20260913/reports/customer_segments.json")
    promotion, promotion_path = load(
        ROOT / "artifacts/corrected_complete_rank5_20260913/reports/segment_promotion_mdp.json")
    query_manifest, query_manifest_path = load(
        ROOT / "artifacts/retail_counterfactual_queries_20260914/real_final_v2/manifest.json")
    synthetic, synthetic_path = load(
        ROOT / "artifacts/retail_application_audit_20260915/synthetic_forced_cart_exact_audit.json")
    soy, soy_path = load(
        ROOT / "artifacts/retail_counterfactual_queries_20260914/real_final_v2/soy_to_dairy_with_bread.json")
    butter, butter_path = load(
        ROOT / "artifacts/retail_counterfactual_queries_20260914/real_final_v2/drop_butter_keep_milk_bread.json")

    rec_model = recommendation["recommendation"]["full_interaction"]
    rec_pop = recommendation["recommendation"]["popularity"]
    interaction = recommendation["recommendation"]["comparison"][
        "mrr_full_minus_fitted_parent"]
    masked = completion_audit["shopper_uniform_random_pair_mask_target"]
    masked_model = masked["model"]
    masked_baseline = masked["training_only_empirical_size_baseline"]
    masked_cross = masked["cross_sell_retrieval"]
    segment_rows = []
    for row in segments["segments"]:
        distribution = row["simulation"]["distribution"]
        segment_rows.append({
            "segment": row["segment"], "label": row["label"],
            "households": row["households"], "test_trips": row["test_trips"],
            "observed_size_mean": distribution["moments"]["observed_size_mean"],
            "generated_size_mean": distribution["moments"]["generated_size_mean"],
            "item_total_variation": distribution["item"]["total_variation"],
            "category_total_variation": distribution["category"]["total_variation"],
        })
    smc_by_trip = {row["trip"]: row for row in new["per_context"]}
    quadrature_by_trip = {row["trip"]: row for row in quadrature["per_context"]}
    common_trips = sorted(set(smc_by_trip) & set(quadrature_by_trip))
    estimator_agreement = {
        "contexts": len(common_trips),
        "maximum_absolute_stop_probability_difference": max(abs(
            smc_by_trip[trip]["stop_probability"]
            - quadrature_by_trip[trip]["stop_probability"]) for trip in common_trips),
        "mean_absolute_stop_probability_difference": sum(abs(
            smc_by_trip[trip]["stop_probability"]
            - quadrature_by_trip[trip]["stop_probability"]) for trip in common_trips)
            / len(common_trips),
        "maximum_absolute_expected_additional_items_difference": max(abs(
            smc_by_trip[trip]["expected_additional_items"]
            - quadrature_by_trip[trip]["expected_additional_items"])
            for trip in common_trips),
        "mean_absolute_expected_additional_items_difference": sum(abs(
            smc_by_trip[trip]["expected_additional_items"]
            - quadrature_by_trip[trip]["expected_additional_items"])
            for trip in common_trips) / len(common_trips),
    }

    applications = [
        {
            "application": "real_time_cross_sell",
            "retailer_decision": "rank products to show after the currently revealed cart",
            "estimand": "P(j in eventual basket completion | A subset S, customer/store/week)",
            "evidence_status": "promising_offline_not_deployment_certified",
            "masking_corrected_forced_cart_evaluation": masked_cross,
            "large_locked_test": {
                "cases": rec_model["cases"], "model_mrr": rec_model["mrr"],
                "popularity_mrr": rec_pop["mrr"],
                "model_recall_at_20": rec_model["recall_at_20"],
                "popularity_recall_at_20": rec_pop["recall_at_20"],
            },
            "important_caveat": (
                "the full fitted interaction adds no statistically resolved MRR gain over "
                "the fitted additive parent"),
            "next_test": "chronological cart logs or a randomized recommendation-slot A/B test",
        },
        {
            "application": "basket_completion",
            "retailer_decision": "forecast remaining basket size and likely completion set",
            "estimand": "P(T | A subset S,x), including T=empty",
            "evidence_status": "masked_pair_mean_and_mae_pass_distribution_not_certified",
            "evaluation": masked_model,
            "training_only_empirical_size_baseline": masked_baseline,
            "uncertainty": masked["uncertainty"],
            "important_caveat": (
                "the validation protocol reveals a random pair, so its law includes the "
                "known 1/C(final_size,2) observation weight; this is not chronological"),
            "next_test": "repeat on a larger locked panel and then on chronological cart prefixes",
        },
        {
            "application": "stopping_probability",
            "retailer_decision": "decide whether another suggestion is useful or intrusive",
            "estimand": "P(T=empty | A subset S,x)",
            "evidence_status": "modest_discrimination_improvement_vs_baseline_not_resolved",
            "evaluation": {
                "model": masked_model,
                "training_only_empirical_size_baseline": masked_baseline,
                "uncertainty": masked["uncertainty"],
            },
            "next_test": "cart-event timestamps with checkout as the stop label; evaluate Brier, log loss, AUC, calibration",
        },
        {
            "application": "stockout_substitution",
            "retailer_decision": "rank alternatives after an unavailable desired product",
            "estimand": "P(k in S | A subset S, desired j unavailable,x)",
            "evidence_status": "proxy_only_not_identified",
            "evaluation": new["stockout_substitution_proxy"],
            "next_test": "join SKU-store-time availability and search/cart intent; test actual stockout episodes",
        },
        {
            "application": "price_scenarios",
            "retailer_decision": "compare basket probabilities under declared shelf-price vectors",
            "estimand": "fitted-distribution scenario P_model(S|do-like price input,x), not an identified causal effect",
            "evidence_status": "numerically_usable_scenario_only_not_effect_validated",
            "observational_check": {
                "events": price["evaluated_events"],
                "child_mae": price["metrics"]["child_mae"],
                "parent_mae": price["metrics"]["parent_mae"],
                "sign_agreement": price["metrics"]["child_sign_agreement"],
                "model_accuracy_status": price["model_accuracy_status"],
                "causal_identification": price["causal_price_identification"],
            },
            "query_run": {
                "status": query_manifest["status"], "particles": query_manifest["particles"],
                "levels": query_manifest["levels"], "replicates": query_manifest["replicates"],
            },
            "worked_examples": {
                "soy_candidate_set_dairy_probability_change": soy["query_result"][
                    "candidate_basket_comparison"]["changes"][1][
                        "conditional_probability_change"],
                "butter_exact_rest_probability_baseline": butter["query_result"][
                    "exact_rest_additions"][0]["baseline_probability"],
                "butter_exact_rest_probability_after_20pct_price_rise": butter[
                    "query_result"]["exact_rest_additions"][0][
                        "counterfactual_probability"],
                "warning": "declared candidate/exact-rest comparisons, not population substitution effects",
            },
            "next_test": "randomized SKU/store price cells with exposure, availability, margin, units, and visit outcomes",
        },
        {
            "application": "personalized_bundles",
            "retailer_decision": "rank a small set of multi-product offers for a customer/cart",
            "estimand": "candidate-set probability from exact conditional completion scores",
            "evidence_status": "offline_proxy_only",
            "evaluation": new["personalized_bundle_retrieval"],
            "next_test": "randomize bundle exposure and measure attach rate, incremental margin, and cannibalization",
        },
        {
            "application": "promotion_targeting",
            "retailer_decision": "choose segment, bundle, discount, and campaign timing",
            "estimand": "model-conditional incremental basket value among trips",
            "evidence_status": "not_deployable_policy",
            "existing_scenario": {
                "safe_actions": len(promotion["safe_daily_actions"]),
                "maximum_campaign_spend_under_action_space": promotion[
                    "maximum_campaign_spend_under_action_space"],
                "not_identified": promotion["not_identified"],
            },
            "blocking_reason": "factual basket calibration failed and causal promotion lift/profit are not identified",
            "next_test": "cluster-randomized promotion holdout with costs, exposure, inventory, trips, units, and margin",
        },
        {
            "application": "assortment_planning",
            "retailer_decision": "simulate removals/additions and expected substitution across the assortment",
            "estimand": "basket law after changing the offered set",
            "evidence_status": "not_evaluable_from_current_inputs",
            "blocking_reason": (
                "historical assortment and stock availability interventions are absent; "
                "the current fitted factual basket law is also miscalibrated"),
            "next_test": "store-week availability/planogram/cost/capacity data and phased assortment experiments",
        },
        {
            "application": "demand_forecasting",
            "retailer_decision": "forecast SKU/category incidence conditional on a shopping trip",
            "estimand": "basket incidence conditional on observed visit context",
            "evidence_status": "failed_factual_calibration",
            "evaluation": generation["factual_calibration"],
            "sampler_status": generation["sampler_correctness"],
            "scope_limit": "does not forecast whether a trip occurs, so it is not total store demand",
            "next_test": "repair conditional size calibration, then add visit counts and trained unit demand",
        },
        {
            "application": "customer_segmentation",
            "retailer_decision": "define stable descriptive audiences for reporting and experiments",
            "estimand": "clusters of rotation-invariant learned taste and price surfaces",
            "evidence_status": "usable_for_descriptive_stratification_not_response_targeting",
            "selection": {
                "chosen_segments": segments["chosen_segments"],
                "silhouette": segments["candidate_selection"]["3"]["silhouette"],
                "stability_ari": segments["candidate_selection"]["3"]["stability_ari"],
                "minimum_fraction": segments["candidate_selection"]["3"]["minimum_fraction"],
            },
            "segment_calibration": segment_rows,
            "next_test": "prospective segment-stratified experiments; do not infer treatment effect from cluster labels",
        },
    ]

    sources = [new_path, quadrature_path, completion_path, synthetic_path,
               recommendation_path, generation_path,
               price_path, segments_path, promotion_path, query_manifest_path,
               soy_path, butter_path]
    report = {
        "status": "completed",
        "overall_conclusion": (
            "The fitted model supports offline ranking, masked-pair completion-size "
            "prediction, and descriptive segmentation. Stopping improvements are not "
            "statistically resolved, and demand, price-effect, stockout, promotion, and "
            "assortment decisions remain gated by failed calibration or missing causal data."),
        "synthetic_forced_cart_certification": synthetic,
        "real_estimator_agreement": estimator_agreement,
        "real_quadrature_adjacent_rule_diagnostics": quadrature["quadrature"],
        "applications": applications,
        "source_artifacts": [{"path": str(path), "sha256": file_sha256(path)}
                             for path in sources],
    }
    json_path = args.output_json.resolve(); json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(strict_json_dumps(report))

    new_cross = masked_cross["model"]
    completion = masked_model
    completion_baseline = masked_baseline
    stock = new["stockout_substitution_proxy"]
    bundle = new["personalized_bundle_retrieval"]
    factual = generation["factual_calibration"]
    lines = [
        "# Retail Application Evaluation Audit",
        "",
        "## Executive conclusion",
        "",
        "The joint basket law now answers the correct cart-conditional query "
        "`P(T | A ⊆ S, x)` directly. Revealed items are forced into the basket; the "
        "unobserved completion may be empty, so stopping is a proper probability rather "
        "than a heuristic. Exact enumeration certifies the transformation on synthetic "
        "catalogues, and bridge ESS diagnoses the real-data Monte Carlo calculation.",
        "",
        f"The synthetic certification passed: exact forced-completion score error "
        f"{synthetic['errors']['forced_completion_score_max_error']:.3g}, SMC stopping "
        f"error {synthetic['errors']['forced_completion_smc_stop_max_error']:.3g}, "
        f"maximum incidence error {synthetic['errors']['forced_completion_smc_incidence_max_error']:.4f}, "
        f"and minimum ESS {synthetic['forced_completion_smc']['minimum_ess_fraction']:.6f}. "
        f"The new informative-mask quadrature matched enumerated stopping, size, item "
        f"incidence, and normalization to at worst "
        f"{max(synthetic['errors'][key] for key in synthetic['errors'] if key.startswith('masked_completion_quadrature')):.3g}.",
        "",
        f"On real data, the corrected adjacent deterministic rules used "
        f"{completion_audit['numerical_certification']['nodes'][0]} and "
        f"{completion_audit['numerical_certification']['nodes'][1]} nodes, with a "
        f"{completion_audit['numerical_certification']['nodes'][2]}-node follow-up for "
        f"{completion_audit['numerical_certification']['followup_contexts']} contexts. "
        f"The final maximum stop-probability and item-incidence gaps were "
        f"{completion_audit['numerical_certification']['maximum_followup_stop_gap']:.3g} "
        f"and {completion_audit['numerical_certification']['maximum_followup_item_incidence_gap']:.3g}; "
        f"the independent SMC-versus-quadrature mean absolute stop gap was "
        f"{estimator_agreement['mean_absolute_stop_probability_difference']:.3g}.",
        "",
        "The corrected natural-panel audit no longer supports the earlier conclusion that "
        "basket completion size fails. That conclusion came from a balanced tail panel and "
        "an unmodeled random-pair selection mechanism. The corrected masked-pair mean is "
        "well aligned and MAE improves on the training-only size baseline. Stopping gains "
        "are not statistically resolved, and scenario outputs must not be presented as "
        "causal effects, expected profit, or individual customer transitions.",
        "",
        "## Results by application",
        "",
        "| Application | Concrete retailer output | Evidence in these data | Verdict |",
        "|---|---|---|---|",
        f"| Real-time cross-sell | Top-k products after a two-item masked cart | Masking-corrected natural panel ({new_cross['cases']} non-stop cases): MRR {new_cross['mrr']:.4f} vs {masked_cross['training_popularity']['mrr']:.4f} popularity; hidden-set recall@20 {pct(new_cross['mean_hidden_set_recall_at_20'])} vs {pct(masked_cross['training_popularity']['mean_hidden_set_recall_at_20'])}. Locked 1,615-case test: MRR {rec_model['mrr']:.4f} vs {rec_pop['mrr']:.4f} popularity. | Promising offline; A/B test before deployment. |",
        f"| Basket completion | Distribution over remaining products and count | Natural 256-trip panel: actual remaining mean {completion['observed_additional_items_mean']:.3f}, predicted {completion['predicted_additional_items_mean']:.3f}; MAE {completion['mae_items']:.3f} vs {completion_baseline['mae_items']:.3f} training-size baseline. Paired MAE improvement 95% interval {masked['uncertainty']['model_minus_baseline_mae_items']['percentile_95_interval'][0]:.3f} to {masked['uncertainty']['model_minus_baseline_mae_items']['percentile_95_interval'][1]:.3f} items. | Mean calibrated; MAE improves, full distribution not yet certified. |",
        f"| Stopping | Masking-aware `P(T=∅ | A,x)` | Brier {completion['stop_brier_score']:.4f} vs {completion_baseline['stop_brier_score']:.4f}; log loss {completion['stop_log_loss']:.4f} vs {completion_baseline['stop_log_loss']:.4f}; AUC {completion['stop_roc_auc']:.4f}. Both paired score intervals include zero. | Modest discrimination; improvement unresolved. |",
        f"| Stockout substitution | Alternatives within desired SKU's subcommodity | Held-out-choice proxy: model MRR {stock['model']['mrr']:.4f} vs popularity {stock['popularity']['mrr']:.4f} across {stock['model']['cases']} cases. No stockout labels. | Proxy only; actual substitution not identified. |",
        f"| Price scenarios | Basket probabilities under a declared price vector | 32 observational events: sign agreement {pct(price['metrics']['child_sign_agreement'])}; child MAE {price['metrics']['child_mae']:.6f}, parent MAE {price['metrics']['parent_mae']:.6f}; model accuracy explicitly not assessed. | Numerically usable scenario, not validated causal effect. |",
        f"| Personalized bundles | Candidate bundle odds conditional on cart | Matched-negative proxy: model MRR {bundle['model']['mrr']:.4f} vs popularity {bundle['popularity']['mrr']:.4f} across {bundle['model']['cases']} cases. | Offline proxy; randomized offers needed. |",
        f"| Promotion targeting | Segment/bundle/discount scenario table | Existing MDP has {len(promotion['safe_daily_actions'])} numerically admissible actions, but excludes profit, inventory, visits, switching, and quantities. | Do not deploy as policy. |",
        "| Assortment planning | Recompute basket law after SKU addition/removal | No historical availability, planogram, cost, capacity, or lost-demand intervention labels. | Not evaluable with current data. |",
        f"| Demand forecasting | Conditional SKU/category incidence | Fitted mean size {factual['fitted_mean']:.3f} vs {factual['observed_mean']:.3f} observed; error {factual['fitted_minus_observed']['mean']:+.3f}; item TV {factual['item_total_variation']:.3f}. | Failed factual calibration. |",
        f"| Customer segmentation | Stable descriptive audience IDs | 3 clusters; silhouette {segments['candidate_selection']['3']['silhouette']:.3f}, stability ARI {segments['candidate_selection']['3']['stability_ari']:.3f}; segment item TV spans {min(x['item_total_variation'] for x in segment_rows):.3f}–{max(x['item_total_variation'] for x in segment_rows):.3f}. | Useful for description/experiment strata, not response targeting. |",
        "",
        "## What the conditional probability means",
        "",
        "For a revealed cart `A`, the engine sums over every allowable completion `U`:",
        "",
        "```text",
        "P(T | A ⊆ S, x) = exp(score(A ∪ T)) / Σ_U exp(score(A ∪ U))",
        "```",
        "",
        "`U=∅` is included. Therefore `P(stop | A,x)=P(T=∅ | A ⊆ S,x)`. "
        "This is an eventual-basket completion law. The Dunnhumby transactions do not "
        "contain scan/cart order, so it is not a chronological next-item law.",
        "",
        "A retrospective validation case is built by uniformly selecting two products "
        "from the final basket. That observation mechanism is informative about final "
        "size. Its correct evaluation law is `q(T|A,x) ∝ P(T|A⊆S,x) / C(|A|+|T|,2)`. "
        "The earlier balanced 64-case result omitted this factor and is superseded.",
        "",
        "The earlier worked price examples remain candidate-set calculations: a 15% "
        f"increase in the declared soy SKU moved dairy's two-candidate probability by "
        f"{100 * soy['query_result']['candidate_basket_comparison']['changes'][1]['conditional_probability_change']:.4f} "
        "percentage points. A 20% increase in the declared butter SKU moved its exact-rest "
        f"addition probability from {butter['query_result']['exact_rest_additions'][0]['baseline_probability']:.6f} "
        f"to {butter['query_result']['exact_rest_additions'][0]['counterfactual_probability']:.6f}. "
        "These small changes are model scenarios, not evidence that individual shoppers switch.",
        "",
        f"The independent 128-particle SMC and deterministic quadrature agreed closely "
        f"on stopping (mean absolute difference "
        f"{estimator_agreement['mean_absolute_stop_probability_difference']:.3g}). "
        f"Expected remaining size was noisier under SMC (mean absolute difference "
        f"{estimator_agreement['mean_absolute_expected_additional_items_difference']:.3f} "
        f"items; maximum {estimator_agreement['maximum_absolute_expected_additional_items_difference']:.3f}). "
        "A high bridge ESS establishes stable importance weights; it does not by itself "
        "guarantee that 128 draws precisely estimate every downstream moment.",
        "",
        "## Deployment sequence",
        "",
        "1. Use current outputs only for candidate generation and analyst scenario review.",
        "2. Repeat the masking-corrected audit on a larger locked panel and add calibration "
        "curves; require held-out size, Brier, log-loss, and calibration gates.",
        "3. Log cart order, recommendation exposure, stock availability, shelf price, "
        "promotion assignment, units, costs, margin, and trip/no-trip opportunities.",
        "4. Run randomized tests for recommendation, price, bundle, and promotion decisions.",
        "5. Promote an application only when its own decision metric passes; a good joint "
        "likelihood or stable ESS is not a substitute for application validation.",
        "",
        "## Reproducibility",
        "",
        f"New application evaluation: `{new_path}`",
        f"Deterministic cart-conditional audit: `{quadrature_path}`",
        f"Corrected basket-completion audit: `{completion_path}`",
        f"Corrected basket-completion log: `{completion_path.with_suffix('.log')}`",
        f"New evaluation log: `{new_path.with_suffix('.log')}`",
        f"Machine-readable consolidated audit: `{json_path}`",
    ]
    md_path = args.output_md.resolve(); md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines) + "\n")
    print(strict_json_dumps({"status": "completed", "json": str(json_path),
                             "markdown": str(md_path)}), end="")


if __name__ == "__main__":
    main()
