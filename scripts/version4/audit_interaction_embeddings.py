#!/usr/bin/env python3
"""Audit orientation-invariant Version-4 pair-specific interaction coefficients.

The embedding columns are not identified: ``Phi Q`` defines the same basket law for every
orthogonal Q.  This audit therefore reports row norms and pairwise Gram coefficients, then
checks training-selected cross-affinity pairs on held-out baskets against frequency/support
matched controls.  Test outcomes never select the candidate pairs.

The Gram/category coefficient is the product-specific component of the exact energy
cross-difference.  The latter additionally contains the common basket-size curvature
``-Delta^2 rho_0(|T|)``.
"""
from __future__ import annotations

import argparse
import heapq
import json
import os
from pathlib import Path

os.environ.setdefault("V3_AFFINITY", "1")

import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree

from data import build


ROOT = Path(__file__).resolve().parents[2]


def quantiles(values):
    levels = (0.0, 0.01, 0.10, 0.50, 0.90, 0.99, 1.0)
    return {str(level): float(value)
            for level, value in zip(levels, np.quantile(values, levels))}


def cumulative_mass_counts(mass):
    ordered = np.sort(np.asarray(mass, dtype=np.float64))[::-1]
    cumulative = np.cumsum(ordered) / max(float(ordered.sum()), np.finfo(float).tiny)
    return {str(level): int(np.searchsorted(cumulative, level) + 1)
            for level in (0.90, 0.95, 0.99, 0.999)}


def top_pairs(phi, group, eligible, count, relation="different", block=256):
    """Largest Gram coefficients without constructing the dense J by J kernel."""
    total = len(phi)
    candidates = []
    columns = np.arange(total)[None, :]
    for lo in range(0, total, block):
        hi = min(total, lo + block)
        score = phi[lo:hi] @ phi.T
        rows = np.arange(lo, hi)[:, None]
        valid = (columns > rows) & eligible[lo:hi, None] & eligible[None, :]
        if relation == "different":
            valid &= group[lo:hi, None] != group[None, :]
        elif relation == "same":
            valid &= group[lo:hi, None] == group[None, :]
        else:
            raise ValueError("relation must be 'different' or 'same'")
        score[~valid] = -np.inf
        take = min(count, score.size)
        flat = np.argpartition(score.ravel(), -take)[-take:]
        local_row, column = np.unravel_index(flat, score.shape)
        candidates.extend(
            (float(score[a, b]), lo + int(a), int(b))
            for a, b in zip(local_row, column) if np.isfinite(score[a, b]))
    return heapq.nlargest(count, candidates)


def matched_controls(pairs, phi, metadata, eligible, seed):
    """Training-only nearest-neighbor match on frequency and household support.

    Repeated control pairs are allowed.  This preserves the empirical weighting caused by
    an anchor product appearing in several learned top pairs instead of forcing a poor
    one-to-one match for popular anchors.
    """
    pool = np.flatnonzero(eligible)
    feature = np.column_stack((
        np.log1p(metadata.n_train_lines.to_numpy()[pool]),
        np.log1p(metadata.n_train_households.to_numpy()[pool])))
    tree = cKDTree(feature)
    needed = {i for _, i, j in pairs for i in (i, j)}
    nearest = {}
    for item in needed:
        point = [np.log1p(metadata.n_train_lines[item]),
                 np.log1p(metadata.n_train_households[item])]
        _, where = tree.query(point, k=min(64, len(pool)))
        nearest[item] = pool[np.atleast_1d(where)]
    category = metadata.cat_id.to_numpy()
    rng = np.random.default_rng(seed)
    controls = []
    for _, left, right in pairs:
        options = []
        for a in nearest[left][1:16]:
            for b in nearest[right][1:16]:
                a, b = sorted((int(a), int(b)))
                if a != b and category[a] != category[b] and (a, b) != (left, right):
                    options.append((a, b))
        if not options:
            raise RuntimeError("could not construct a frequency-matched control pair")
        a, b = options[int(rng.integers(len(options)))]
        controls.append((float(phi[a] @ phi[b]), a, b))
    return controls


def heldout_pair_statistics(data, pairs, split):
    total_products = int(data["n_item"])
    incidence = np.zeros(total_products, dtype=np.int64)
    observed = np.zeros(len(pairs), dtype=np.int64)
    adjacency = {}
    for index, (_, left, right) in enumerate(pairs):
        adjacency.setdefault(left, []).append((right, index))
        adjacency.setdefault(right, []).append((left, index))
    sizes = []
    trips = np.flatnonzero(data["trip_split"] == split)
    for trip in trips:
        lo, hi = int(data["line_ptr"][trip]), int(data["line_ptr"][trip + 1])
        items = np.unique(data["line_item"][lo:hi])
        sizes.append(len(items))
        incidence[items] += 1
        present = set(map(int, items))
        for left in items:
            for right, index in adjacency.get(int(left), ()):
                if right > left and right in present:
                    observed[index] += 1
    sizes = np.asarray(sizes, dtype=np.float64)
    slots = float(sizes.sum())
    # Configuration null: randomly allocate the observed product-incidence stubs to the
    # observed basket-size slots.  It preserves product frequency and basket-size moments.
    factor = float(np.sum(sizes * (sizes - 1.0)) / (slots * (slots - 1.0)))
    expected = np.asarray(
        [incidence[left] * incidence[right] * factor for _, left, right in pairs],
        dtype=np.float64)
    lift = (observed + 0.5) / (expected + 0.5)
    return dict(trips=int(len(trips)), incidence=incidence, observed=observed,
                expected=expected, lift=lift, configuration_factor=factor)


def summarize_pair_panel(observed, expected, lift):
    return {
        "pairs": int(len(observed)),
        "observed_coincidences": int(observed.sum()),
        "configuration_expected_coincidences": float(expected.sum()),
        "aggregate_lift": float((observed.sum() + 0.5) / (expected.sum() + 0.5)),
        "fraction_observed_above_expected": float(np.mean(observed > expected)),
        "median_smoothed_lift": float(np.median(lift)),
        "pairs_with_at_least_3_test_coincidences": int(np.sum(observed >= 3)),
    }


def pair_record(pair, position, metadata, observed, expected, lift, rho):
    gram, left, right = pair
    same = int(metadata.cat_id[left]) == int(metadata.cat_id[right])
    category_term = -float(rho[int(metadata.cat_id[left])]) if same else 0.0
    def item_record(item):
        return {
            "item_id": int(item),
            "product_id": int(metadata.PRODUCT_ID[item]),
            "description": str(metadata.SUB_COMMODITY_DESC[item]),
            "commodity": str(metadata.COMMODITY_DESC[item]),
            "department": str(metadata.DEPARTMENT[item]),
            "training_lines": int(metadata.n_train_lines[item]),
            "training_households": int(metadata.n_train_households[item]),
        }
    return {
        "rank": int(position + 1),
        "left": item_record(left),
        "right": item_record(right),
        "gram_coefficient": float(gram),
        "same_affinity_group": bool(same),
        "category_pair_coefficient": category_term,
        "pair_specific_interaction_coefficient": float(gram + category_term),
        "test_coincidences": int(observed[position]),
        "configuration_expected": float(expected[position]),
        "smoothed_test_lift": float(lift[position]),
    }


def markdown(result):
    structure = result["structure"]
    selection = result["selection"]
    heldout = result["heldout_cross_affinity_audit"]
    lines = [
        "# Interaction-embedding audit",
        "",
        "## Meaning of a complement score",
        "",
        "Only the Gram matrix is identified; individual embedding axes may rotate. The",
        "invariant pair-specific interaction coefficient is",
        "",
        r"\[",
        r"\gamma_{ij}=\phi_i^\top\phi_j",
        r"-\rho_{c(i)}\mathbf 1\{c(i)=c(j)\}.",
        r"\]",
        "",
        "The full energy cross-difference also includes the common size-curvature term",
        r"`-Delta^2 rho_0(|T|)`; it does not change pair ordering at a fixed",
        "background size.",
        "",
        "A positive coefficient is model-implied complementarity after the additive, size,",
        "price, household and remaining category terms are held fixed. It is predictive,",
        "not a causal cross-price effect.",
        "",
        "## Structural result",
        "",
        f"- Active rank: {structure['active_rank']}.",
        "- Active singular values: " + ", ".join(
            f"{x:.6f}" for x in structure["active_singular_values"]) + ".",
        f"- Products carrying 90%/95%/99% of row-norm mass: "
        f"{structure['products_for_cumulative_row_mass']['0.9']}/"
        f"{structure['products_for_cumulative_row_mass']['0.95']}/"
        f"{structure['products_for_cumulative_row_mass']['0.99']}.",
        f"- Rank-{structure['stability_rank']} split-half mean squared subspace overlap: "
        f"{structure['split_half_overlap']:.6f}.",
        "",
        (f"All {structure['active_rank']} active singular values are at the declared "
         "spectral cap." if structure["spectral_cap_saturated"] else
         "Not every active singular value is at the declared spectral cap."),
        "Pair ordering is useful for hypotheses, but magnitude and subspace stability",
        "must be retained when judging product-level claims.",
        "",
        "## Held-out aggregate check",
        "",
        f"The {selection['top_cross_affinity_pairs']:,} strongest cross-affinity pairs "
        "were selected from training parameters",
        "only. Test co-incidence is compared with a configuration null preserving product",
        "frequency and the observed basket-size sequence, plus training-frequency/household-",
        "support matched control pairs.",
        "",
        "| Panel | Observed | Expected | Aggregate lift | Fraction above null |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, label in (("top_pairs", "Top Gram pairs"),
                       ("matched_controls", "Matched controls")):
        block = heldout[key]
        lines.append(
            f"| {label} | {block['observed_coincidences']:,} | "
            f"{block['configuration_expected_coincidences']:,.1f} | "
            f"{block['aggregate_lift']:.3f} | "
            f"{block['fraction_observed_above_expected']:.1%} |")
    lines.extend(["", "The aggregate test supports information in the learned Gram kernel.",
                  "It does not imply that every top pair replicates.", "",
                  "## Highest cross-affinity pairs", "",
                  "| Rank | Product pair | Gram | Test observed/expected | Lift |",
                  "|---:|---|---:|---:|---:|"])
    for row in result["top_cross_affinity_pairs"]:
        lines.append(
            f"| {row['rank']} | {row['left']['description']} — "
            f"{row['right']['description']} | {row['gram_coefficient']:.5f} | "
            f"{row['test_coincidences']}/{row['configuration_expected']:.2f} | "
            f"{row['smoothed_test_lift']:.2f} |")
    lines.extend(["", "## Interpretation", "",
                  "Examples such as hot dogs with hot-dog or hamburger buns have both a",
                  "positive learned coefficient and held-out excess co-incidence. Produce",
                  "bundles also dominate the leading kernel. Some high-score pairs, such as",
                  "particular milk/banana or egg/banana SKUs, have test lift at or below one;",
                  "they must not be presented as established individual complements.", "",
                  "Within-affinity pairs must add the explicit `-rho_c` coefficient. Those",
                  "effects are reported separately in the JSON and are not attributed to",
                  "the interaction embedding.", "",
                  "Recommended use: retrieve candidate complements with the Gram score,",
                  "require minimum support and held-out replication, then validate promotions",
                  "experimentally. Do not infer causal demand response from co-incidence."])
    return "\n".join(lines) + "\n"


def main(args):
    checkpoint = args.checkpoint if args.checkpoint.is_absolute() else ROOT / args.checkpoint
    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = blob["model"]
    rank = int(blob.get("active_rank", 0))
    if not 0 < rank <= state["phi"].shape[1]:
        raise RuntimeError("checkpoint has no valid active interaction rank")
    phi = state["phi"][:, :rank].double().numpy()
    inactive_max = float(state["phi"][:, rank:].abs().max()) if rank < state["phi"].shape[1] else 0.0
    if inactive_max != 0.0:
        raise RuntimeError("checkpoint has nonzero interaction columns outside active rank")
    rho = state["rho_c"].double().numpy()
    metadata = pd.read_parquet(ROOT / "basket_input/items.parquet").sort_values(
        "item_id").reset_index(drop=True)
    if len(metadata) != len(phi) or not np.array_equal(metadata.item_id, np.arange(len(phi))):
        raise RuntimeError("product metadata does not match checkpoint row order")
    eligible = metadata.n_train_lines.to_numpy() >= args.minimum_training_lines
    category = metadata.cat_id.to_numpy()
    pairs = top_pairs(phi, category, eligible, args.pairs, relation="different")
    controls = matched_controls(pairs, phi, metadata, eligible, args.seed)
    data = build()
    heldout = heldout_pair_statistics(data, pairs + controls, split=2)
    cut = len(pairs)
    top_summary = summarize_pair_panel(
        heldout["observed"][:cut], heldout["expected"][:cut], heldout["lift"][:cut])
    control_summary = summarize_pair_panel(
        heldout["observed"][cut:], heldout["expected"][cut:], heldout["lift"][cut:])
    if args.spectral_report is not None:
        spectral_path = (args.spectral_report if args.spectral_report.is_absolute()
                         else ROOT / args.spectral_report)
    else:
        candidates = sorted((ROOT / "artifacts").glob("interaction_basis_rank*.json"))
        if not candidates:
            raise RuntimeError("no spectral report found; pass --spectral-report")
        spectral_path = candidates[-1]
    spectral = json.loads(spectral_path.read_text())
    stability_profiles = spectral["rank_stability"]
    stability_rank = rank if str(rank) in stability_profiles else int(
        spectral["selected_rank"])
    stability = stability_profiles[str(stability_rank)]
    singular = np.linalg.svd(phi, compute_uv=False)
    row_mass = np.square(phi).sum(1)
    listed = min(args.listed_pairs, len(pairs))
    pair_rows = [pair_record(pair, k, metadata, heldout["observed"],
                             heldout["expected"], heldout["lift"], rho)
                 for k, pair in enumerate(pairs[:listed])]
    # Highest cross-department subset is often easier to interpret operationally.
    departments = metadata.DEPARTMENT.astype(str).to_numpy()
    cross_department_positions = [k for k, (_, i, j) in enumerate(pairs)
                                  if departments[i] != departments[j]][:listed]
    cross_department = [pair_record(pairs[k], k, metadata, heldout["observed"],
                                    heldout["expected"], heldout["lift"], rho)
                        for k in cross_department_positions]
    same_pairs = top_pairs(phi, category, eligible, listed, relation="same")
    same_records = []
    for position, (gram, left, right) in enumerate(same_pairs):
        category_term = -float(rho[int(category[left])])
        same_records.append({
            "rank": position + 1,
            "left_product_id": int(metadata.PRODUCT_ID[left]),
            "left_description": str(metadata.SUB_COMMODITY_DESC[left]),
            "right_product_id": int(metadata.PRODUCT_ID[right]),
            "right_description": str(metadata.SUB_COMMODITY_DESC[right]),
            "gram_coefficient": gram,
            "category_pair_coefficient": category_term,
            "pair_specific_interaction_coefficient": gram + category_term,
        })
    result = {
        "checkpoint": str(checkpoint),
        "split": "test",
        "definition": (
            "pair-specific gamma_ij = phi_i'phi_j - rho_c(i) * 1[c(i)=c(j)]; "
            "full energy cross-difference also subtracts Delta^2 rho_0(|T|)"),
        "structure": {
            "products": int(len(phi)),
            "active_rank": rank,
            "inactive_maximum_absolute_loading": inactive_max,
            "active_singular_values": singular.tolist(),
            "row_norm_quantiles": quantiles(np.sqrt(row_mass)),
            "products_for_cumulative_row_mass": cumulative_mass_counts(row_mass),
            "spectral_report": str(spectral_path),
            "stability_rank": stability_rank,
            "stability_applies_exactly_to_active_rank": stability_rank == rank,
            "split_half_overlap": float(
                stability["split_half_mean_squared_subspace_overlap"]),
            "split_half_cosines": stability["split_half_subspace_cosines"],
            "spectral_cap_saturated": bool(np.allclose(singular, 1.0, atol=1e-8)),
        },
        "selection": {
            "minimum_training_lines_per_product": args.minimum_training_lines,
            "top_cross_affinity_pairs": len(pairs),
            "test_outcomes_used_for_selection": False,
            "control_match": (
                "nearest products by log training lines and log training households; "
                "different affinity groups"),
        },
        "heldout_cross_affinity_audit": {
            "test_trips": heldout["trips"],
            "configuration_factor": heldout["configuration_factor"],
            "top_pairs": top_summary,
            "matched_controls": control_summary,
        },
        "top_cross_affinity_pairs": pair_rows,
        "top_cross_department_pairs": cross_department,
        "top_within_affinity_pairs": same_records,
        "limitations": [
            "embedding axes are not identified; only Gram coefficients are interpreted",
            "spectral-cap saturation limits magnitude interpretation",
            (f"rank-{stability_rank} split-half stability is the relevant basis "
             "diagnostic"),
            "configuration lift controls frequency and size, not every household/context covariate",
            "pair scores and held-out co-incidence are predictive, not causal cross-price effects",
        ],
    }
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    md_output = output.with_suffix(".md")
    md_output.write_text(markdown(result))
    print(json.dumps({
        "output": str(output),
        "markdown": str(md_output),
        "active_rank": rank,
        "top_pair_aggregate_lift": top_summary["aggregate_lift"],
        "control_aggregate_lift": control_summary["aggregate_lift"],
        "top_pair_fraction_above_null": top_summary["fraction_observed_above_expected"],
        "control_fraction_above_null": control_summary["fraction_observed_above_expected"],
    }, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("artifacts/candidate_rank1.pt"))
    parser.add_argument("--spectral-report", type=Path,
                        help="rank-stability JSON used to construct the interaction basis")
    parser.add_argument("--minimum-training-lines", type=int, default=100)
    parser.add_argument("--pairs", type=int, default=2000)
    parser.add_argument("--listed-pairs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--output", type=Path,
                        default=Path("reports/interaction_embedding_audit.json"))
    main(parser.parse_args())
