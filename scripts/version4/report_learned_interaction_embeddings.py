#!/usr/bin/env python3
"""Create a reproducible, plot-backed audit of the learned Version-4 Gram embedding."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("V3_AFFINITY", "1")

import matplotlib.pyplot as plt
from matplotlib.text import Text
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from audit_interaction_embeddings import (
    heldout_pair_statistics,
    matched_controls,
    top_pairs,
)
from checkpoint_io import ROOT, load_checkpoint
from data import build
from provenance import file_sha256, strict_json_dumps


torch.set_default_dtype(torch.float64)


def _style():
    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 180,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def _save(fig, path: Path):
    # Keep source strings warning-free while emitting the single backslashes expected by
    # Matplotlib mathtext and Markdown TeX renderers.
    for label in fig.findobj(match=Text):
        label.set_text(label.get_text().replace(chr(92) * 2, chr(92)))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _item(metadata: pd.DataFrame, index: int, norm: np.ndarray) -> dict:
    row = metadata.iloc[index]
    return {
        "item_id": int(index),
        "product_id": int(row.PRODUCT_ID),
        "description": str(row.SUB_COMMODITY_DESC),
        "commodity": str(row.COMMODITY_DESC),
        "department": str(row.DEPARTMENT),
        "training_lines": int(row.n_train_lines),
        "training_households": int(row.n_train_households),
        "embedding_norm": float(norm[index]),
    }


def _pair_row(pair, position, metadata, heldout):
    score, left, right = pair
    return {
        "rank": int(position + 1),
        "left_product_id": int(metadata.PRODUCT_ID.iloc[left]),
        "left": str(metadata.SUB_COMMODITY_DESC.iloc[left]),
        "right_product_id": int(metadata.PRODUCT_ID.iloc[right]),
        "right": str(metadata.SUB_COMMODITY_DESC.iloc[right]),
        "left_department": str(metadata.DEPARTMENT.iloc[left]),
        "right_department": str(metadata.DEPARTMENT.iloc[right]),
        "gram": float(score),
        "test_observed": int(heldout["observed"][position]),
        "test_expected": float(heldout["expected"][position]),
        "test_lift": float(heldout["lift"][position]),
    }


def _markdown_link(output: Path, figure: Path) -> str:
    return Path(os.path.relpath(figure, output.parent)).as_posix()


def create_plots(*, figure_dir: Path, phi: np.ndarray, metadata: pd.DataFrame,
                 singular: np.ndarray, stability_cosines: np.ndarray,
                 pairs, controls, heldout, deciles, departments) -> dict[str, Path]:
    _style()
    paths = {
        "rank": figure_dir / "01-rank-and-stability.png",
        "support": figure_dir / "02-loading-versus-support.png",
        "concentration": figure_dir / "03-interaction-mass-concentration.png",
        "department": figure_dir / "04-department-mass.png",
        "heldout": figure_dir / "05-heldout-pair-validation.png",
    }
    blue, orange, green, grey = "#2f6f9f", "#dd8452", "#4c956c", "#777777"

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
    rank_axis = np.arange(1, len(singular) + 1)
    axes[0].bar(rank_axis, singular, color=blue, width=0.65)
    axes[0].axhline(1.0, color=orange, linewidth=1.2, linestyle="--",
                    label="spectral cap")
    axes[0].set(xticks=rank_axis, xlabel="Active interaction direction",
                ylabel="Singular value", title="Fitted interaction spectrum")
    axes[0].set_ylim(0, max(1.08, float(singular.max()) * 1.08))
    axes[0].legend(frameon=False, loc="lower right")
    cosine_axis = np.arange(1, len(stability_cosines) + 1)
    axes[1].bar(cosine_axis, stability_cosines, color=green, width=0.65)
    axes[1].set(xticks=cosine_axis, xlabel="Matched split-half direction",
                ylabel="Subspace cosine", title="Training split-half stability")
    axes[1].set_ylim(0, 1.0)
    _save(fig, paths["rank"])

    norm = np.linalg.norm(phi, axis=1)
    x = np.log10(metadata.n_train_lines.to_numpy() + 1.0)
    y = np.log10(norm)
    correlation = float(spearmanr(norm, metadata.n_train_lines.to_numpy()).statistic)
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    hb = ax.hexbin(x, y, gridsize=46, mincnt=1, bins="log", cmap="Blues")
    fig.colorbar(hb, ax=ax, label="Products per hexagon (log scale)")
    # Label only distinct high-leverage examples and stagger them deliberately; the two
    # leading milk SKUs and the two bun SKUs otherwise occupy almost identical positions.
    distinct = []
    seen = set()
    for index in np.argsort(norm)[::-1]:
        label = str(metadata.SUB_COMMODITY_DESC.iloc[index])[:24]
        if label not in seen:
            distinct.append((int(index), label))
            seen.add(label)
        if len(distinct) == 3:
            break
    offsets = ((-105, -4), (-42, -25), (-94, 16))
    for (index, label), offset in zip(distinct, offsets):
        ax.annotate(label, (x[index], y[index]), xytext=offset,
                    textcoords="offset points", fontsize=8,
                    arrowprops={"arrowstyle": "-", "color": "#777777", "lw": 0.7})
    ax.text(0.03, 0.96, f"Spearman $r_s$ = {correlation:.3f}",
            transform=ax.transAxes, va="top")
    ax.set(xlabel=r"$\log_{10}$(training basket-lines + 1)",
           ylabel=r"$\log_{10}(\|\phi_j\|_2)$",
           title="Products with more evidence receive larger interaction loadings")
    _save(fig, paths["support"])

    mass = norm ** 2
    ordered = np.sort(mass)[::-1]
    cumulative = np.cumsum(ordered) / ordered.sum()
    fraction = np.arange(1, len(ordered) + 1) / len(ordered)
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.plot(fraction, cumulative, color=blue, linewidth=2)
    ax.plot([0, 1], [0, 1], color=grey, linewidth=1, linestyle="--",
            label="uniform mass")
    for level in (0.90, 0.95, 0.99):
        where = int(np.searchsorted(cumulative, level))
        ax.scatter(fraction[where], cumulative[where], color=orange, s=28, zorder=3)
        ax.annotate(f"{level:.0%}: {where + 1:,} products",
                    (fraction[where], cumulative[where]), xytext=(5, -13),
                    textcoords="offset points", fontsize=8)
    ax.set(xlim=(0, 1), ylim=(0, 1.01), xlabel="Fraction of products, largest first",
           ylabel=r"Cumulative share of $\sum_j\|\phi_j\|_2^2$",
           title="Interaction mass is concentrated but not confined to a tiny SKU set")
    ax.legend(frameon=False, loc="lower right")
    _save(fig, paths["concentration"])

    shown = departments.head(9).sort_values("embedding_mass_share")
    y_pos = np.arange(len(shown))
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    width = 0.36
    ax.barh(y_pos - width / 2, shown.embedding_mass_share * 100, height=width,
            color=blue, label="embedding mass")
    ax.barh(y_pos + width / 2, shown.training_line_share * 100, height=width,
            color=orange, label="training lines")
    ax.set(yticks=y_pos, yticklabels=shown.index, xlabel="Share of total (%)",
           title="Department interaction mass compared with observed training support")
    ax.legend(frameon=False, loc="lower right")
    _save(fig, paths["department"])

    pair_count = len(pairs)
    top_lift = np.log2(heldout["lift"][:pair_count])
    control_lift = np.log2(heldout["lift"][pair_count:])
    lo, hi = np.quantile(np.concatenate((top_lift, control_lift)), [0.01, 0.99])
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    bins = np.linspace(lo, hi, 35)
    axes[0].hist(np.clip(top_lift, lo, hi), bins=bins, density=True,
                 alpha=0.60, color=blue, label="top Gram pairs")
    axes[0].hist(np.clip(control_lift, lo, hi), bins=bins, density=True,
                 alpha=0.55, color=orange, label="matched controls")
    axes[0].axvline(0, color=grey, linewidth=1, linestyle="--")
    axes[0].set(xlabel=r"$\log_2$ smoothed test co-incidence lift",
                ylabel="Density", title="Held-out lift distribution")
    axes[0].legend(frameon=False)
    axes[1].plot([row["median_score"] for row in deciles],
                 [row["aggregate_lift"] for row in deciles], marker="o",
                 color=green, linewidth=1.8)
    axes[1].axhline(1.0, color=grey, linewidth=1, linestyle="--")
    axes[1].set(xlabel="Median Gram score within decile",
                ylabel="Aggregate held-out lift",
                title="Higher scores are directionally, not perfectly, stronger")
    _save(fig, paths["heldout"])
    return paths


def write_report(*, output: Path, figures: dict[str, Path], summary: dict,
                 top_products: list[dict], top_pair_rows: list[dict]):
    structure = summary["structure"]
    support = summary["support_relationship"]
    validation = summary["heldout_validation"]
    departments = summary["departments"]
    lines = [
        "# Learned interaction embeddings: audit against the transaction data",
        "",
        "## Conclusion first",
        "",
        "The rank-5 interaction embedding contains reproducible aggregate co-purchase",
        "information. Product pairs selected only from the fitted training parameters have",
        f"held-out aggregate lift **{validation['top_aggregate_lift']:.3f}**, while matched",
        f"controls have lift **{validation['control_aggregate_lift']:.3f}**. This supports",
        "using the embedding to retrieve candidate complements.",
        "",
        "The result is not a license to interpret every high-scoring pair as a causal",
        "bundle. Loading magnitude is strongly related to product support, the five fitted",
        "singular values reach their imposed cap, and the score-to-lift relationship is",
        "positive but weak. The defensible use is therefore: retrieve with the model,",
        "filter on support and held-out replication, and validate commercial actions in an",
        "experiment.",
        "",
        "## 1. What is being audited",
        "",
        r"For product $j$, the model learns a vector $\phi_j\in\mathbb{R}^5$. The".replace("\\\\", "\\"),
        "interaction contribution to the basket log-score is",
        "",
        "$$",
        r"\sum_{i<j,\ i,j\in S}\phi_i^\top\phi_j.".replace("\\\\", "\\"),
        "$$",
        "",
        "Thus the orientation-invariant pair score is",
        "",
        "$$",
        r"g_{ij}=\phi_i^\top\phi_j.".replace("\\\\", "\\"),
        "$$",
        "",
        "A positive $g_{ij}$ raises the model score when the two products occur together,",
        "after the additive household, product, price, promotion, store, season, category",
        "and basket-size terms have been accounted for. For two products in the same",
        "affinity group, the complete pair-specific coefficient also contains the fitted",
        "category term $-\rho_{c(i)}$. Basket-size curvature contributes a common",
        "background-size term to a full cross-difference.",
        "",
        r"Individual coordinates of $\phi_j$ are not uniquely identified: replacing".replace("\\\\", "\\"),
        r"$\Phi$ by $\Phi Q$ for an orthogonal matrix $Q$ leaves every $g_{ij}$ unchanged.".replace("\\\\", "\\"),
        "Consequently this report interprets norms, subspaces and Gram products—not named",
        "embedding axes.",
        "",
        "## 2. Audit design and leakage control",
        "",
        f"The accepted checkpoint contains {structure['products']:,} products and active",
        f"rank {structure['active_rank']}. Geometry and candidate pair selection use the",
        "trained parameters and training-only product counts. Only after the 2,000 strongest",
        "eligible cross-affinity pairs are fixed do we inspect all 23,340 test baskets.",
        "Products must have at least 100 training basket-lines. Controls are matched using",
        "training frequency and training-household support. Test outcomes never select a",
        "pair or control.",
        "",
        "The held-out reference is a configuration null. If $f_i$ is the test incidence of",
        "product $i$ and $n_t$ is test basket size, its expected co-incidence is",
        "",
        "$$",
        r"E_{ij}^{\mathrm{null}}=f_i f_j".replace("\\\\", "\\"),
        r"\frac{\sum_t n_t(n_t-1)}{(\sum_t n_t)(\sum_t n_t-1)}.".replace("\\\\", "\\"),
        "$$",
        "",
        "This preserves product frequencies and the test basket-size sequence. It does not",
        "control every household, store or seasonal covariate, so the test is predictive",
        "rather than causal.",
        "",
        "## 3. Rank and stability",
        "",
        f"![Fitted rank and split-half stability]({_markdown_link(output, figures['rank'])})",
        "",
        "All five active singular values equal the spectral cap of 1. This means the rank-5",
        "model uses all permitted interaction directions, but their absolute scale is",
        "partly determined by the safety constraint. Relative pair ordering is more",
        "defensible than interpreting coefficient magnitude as an unconstrained optimum.",
        "",
        "The split-half subspace cosines are " + ", ".join(
            f"{value:.3f}" for value in structure["split_half_cosines"]) + ". Their mean",
        f"squared overlap is {structure['split_half_overlap']:.3f}. The leading directions",
        "are reproducible, while the fifth is noticeably less stable. This agrees with the",
        "pipeline's decision to use rank 5 rather than force rank 8.",
        "",
        "## 4. Which products carry the interaction signal",
        "",
        f"![Interaction norm versus product support]({_markdown_link(output, figures['support'])})",
        "",
        rf"The Spearman correlation between $\|\phi_j\|_2$ and training basket-lines is".replace("\\\\", "\\"),
        f"**{support['spearman_norm_training_lines']:.3f}**; with the number of training",
        f"households it is **{support['spearman_norm_training_households']:.3f}**. The model",
        "therefore assigns most interaction leverage where the input data can estimate it.",
        "This is statistically sensible, but it means a low norm for a rare product should",
        "not be read as evidence that the product has no complements.",
        "",
        f"![Cumulative interaction mass]({_markdown_link(output, figures['concentration'])})",
        "",
        f"The largest {structure['products_for_mass_90']:,}/"
        f"{structure['products_for_mass_95']:,}/"
        f"{structure['products_for_mass_99']:,} products carry 90%/95%/99% of total squared",
        "embedding norm. Signal is concentrated, but it is not a tiny sparse lookup table:",
        "roughly one third of the catalogue is needed for 95% of the mass.",
        "",
        "### Products with the largest norms",
        "",
        r"| Product | Department | Training lines | Training households | $\|\phi_j\|_2$ |".replace("\\\\", "\\"),
        "|---|---|---:|---:|---:|",
    ]
    for row in top_products[:15]:
        lines.append(
            f"| {row['description']} ({row['product_id']}) | {row['department']} | "
            f"{row['training_lines']:,} | {row['training_households']:,} | "
            f"{row['embedding_norm']:.4f} |")
    lines.extend([
        "",
        "These are high-leverage products, not automatically complements with every other",
        "high-norm product. A specific bundle hypothesis still requires $g_{ij}>0$ and",
        "held-out support for that pair.",
        "",
        "## 5. Department-level allocation",
        "",
        f"![Department interaction mass]({_markdown_link(output, figures['department'])})",
        "",
        f"Grocery supplies {departments['GROCERY']['embedding_mass_share']:.1%} of embedding",
        f"mass and {departments['GROCERY']['training_line_share']:.1%} of training lines.",
        f"Produce supplies {departments['PRODUCE']['embedding_mass_share']:.1%} of embedding",
        f"mass despite only {departments['PRODUCE']['training_line_share']:.1%} of lines.",
        "Produce is therefore interaction-rich relative to its observed volume, consistent",
        "with recurring fruit and vegetable combinations. Very small departments should not",
        "be compared by percentage alone because one product can dominate them.",
        "",
        "## 6. Do high-scoring pairs appear together in held-out baskets?",
        "",
        f"![Held-out pair validation]({_markdown_link(output, figures['heldout'])})",
        "",
        "| Panel | Pairs | Observed test co-incidences | Null expected | Aggregate lift | Above null |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Top Gram pairs | 2,000 | {validation['top_observed']:,} | "
        f"{validation['top_expected']:,.1f} | {validation['top_aggregate_lift']:.3f} | "
        f"{validation['top_fraction_above_null']:.1%} |",
        f"| Matched controls | 2,000 | {validation['control_observed']:,} | "
        f"{validation['control_expected']:,.1f} | {validation['control_aggregate_lift']:.3f} | "
        f"{validation['control_fraction_above_null']:.1%} |",
        "",
        f"Across the selected pairs, Gram score and smoothed held-out lift have Spearman",
        f"correlation **{validation['spearman_score_lift']:.3f}**. This is positive but weak:",
        "the embedding is informative in aggregate, while individual-pair uncertainty and",
        "uncontrolled context remain substantial.",
        "",
        "### Highest cross-affinity pair scores",
        "",
        "| Rank | Pair | Gram score | Test observed / expected | Smoothed lift | Reading |",
        "|---:|---|---:|---:|---:|---|",
    ])
    for row in top_pair_rows[:20]:
        reading = ("replicates" if row["test_observed"] >= 3 and row["test_lift"] > 1
                   else "does not replicate individually")
        lines.append(
            f"| {row['rank']} | {row['left']} — {row['right']} | {row['gram']:.4f} | "
            f"{row['test_observed']} / {row['test_expected']:.1f} | "
            f"{row['test_lift']:.2f} | {reading} |")
    lines.extend([
        "",
        "The table deliberately retains failures. A high model score can identify a useful",
        "population-level family of pairs without guaranteeing that every SKU pair survives",
        "a finite held-out sample. Removing contradictory examples would overstate the model.",
        "",
        "## 7. Operational interpretation",
        "",
        "A retailer can use the embedding as a retrieval layer:",
        "",
        "1. For an anchor SKU $i$, rank eligible products by $g_{ij}$.",
        "2. Add the category coefficient when $i$ and $j$ share an affinity group.",
        "3. Require minimum transaction and household support.",
        "4. Check held-out or recent-period co-incidence against a frequency-aware null.",
        "5. Pass surviving candidates to the basket model for customer/context-specific",
        "   scoring, then to an A/B promotion test before making a causal claim.",
        "",
        "The embedding is most defensible for candidate generation, related-item retrieval",
        "and interpretable basket hypotheses. It should not be sold as proof that discounting",
        "one product causes demand for another: observational co-purchase and cross-price",
        "causality are different estimands.",
        "",
        "## 8. Final assessment",
        "",
        "- **Information learned:** yes. Selected pairs beat frequency/size-matched controls",
        "  on untouched test baskets.",
        "- **Rank justified:** rank 5 is the largest stable audited subspace; rank 8 is not",
        "  supported by the split-half audit.",
        "- **Magnitude fully identified by data:** no. Every singular value reaches the",
        "  spectral cap.",
        "- **Every top pair trustworthy:** no. Aggregate enrichment is clear, but some",
        "  individual pairs contradict the held-out data.",
        "- **Commercial use:** use the kernel for retrieval and hypothesis formation, with",
        "  support filters and experimental confirmation.",
        "",
        "## Reproducibility",
        "",
        f"- Checkpoint SHA-256: `{summary['checkpoint_sha256']}`",
        f"- Data fingerprint: `{summary['data_fingerprint_sha256']}`",
        f"- Checkpoint: `{summary['checkpoint']}`",
        f"- Test baskets: {validation['test_trips']:,}",
        "- Pair selection uses no test outcomes.",
        "",
        "Regenerate this report from the repository root with:",
        "",
        "```bash",
        "python -u scripts/version4/report_learned_interaction_embeddings.py",
        "```",
        "",
    ])
    output.parent.mkdir(parents=True, exist_ok=True)
    rendered = "\n".join(lines).replace(chr(92) * 2, chr(92))
    output.write_text(rendered)


def main(args):
    checkpoint = args.checkpoint if args.checkpoint.is_absolute() else ROOT / args.checkpoint
    spectral_path = (args.spectral_report if args.spectral_report.is_absolute()
                     else ROOT / args.spectral_report)
    output = args.output if args.output.is_absolute() else ROOT / args.output
    figure_dir = args.figure_dir if args.figure_dir.is_absolute() else ROOT / args.figure_dir
    analysis_json = (args.analysis_json if args.analysis_json.is_absolute()
                     else ROOT / args.analysis_json)

    data = build()
    model, blob, _meta = load_checkpoint(
        checkpoint, data,
        required_capabilities=("conditional_nonempty_incidence", "gram_interactions"))
    rank = int(blob["active_rank"])
    phi = model.phi[:, :rank].detach().cpu().numpy()
    metadata = pd.read_parquet(ROOT / "basket_input" / "items.parquet").sort_values(
        "item_id").reset_index(drop=True)
    if len(metadata) != len(phi):
        raise RuntimeError("product metadata and interaction embedding have different rows")
    norm = np.linalg.norm(phi, axis=1)
    mass = norm ** 2
    singular = np.linalg.svd(phi, compute_uv=False)
    spectral = json.loads(spectral_path.read_text())
    stability = spectral["rank_stability"][str(rank)]
    stability_cosines = np.asarray(stability["split_half_subspace_cosines"])

    eligible = metadata.n_train_lines.to_numpy() >= args.minimum_training_lines
    category = metadata.cat_id.to_numpy()
    pairs = top_pairs(phi, category, eligible, args.pairs, relation="different")
    controls = matched_controls(pairs, phi, metadata, eligible, args.seed)
    heldout = heldout_pair_statistics(data, pairs + controls, split=2)
    pair_count = len(pairs)
    top_observed = heldout["observed"][:pair_count]
    top_expected = heldout["expected"][:pair_count]
    top_lift = heldout["lift"][:pair_count]
    control_observed = heldout["observed"][pair_count:]
    control_expected = heldout["expected"][pair_count:]
    control_lift = heldout["lift"][pair_count:]
    scores = np.asarray([pair[0] for pair in pairs])
    order = np.argsort(scores)
    deciles = []
    for index, positions in enumerate(np.array_split(order, 10), start=1):
        observed = int(top_observed[positions].sum())
        expected = float(top_expected[positions].sum())
        deciles.append({
            "decile_low_to_high": index,
            "pairs": int(len(positions)),
            "minimum_score": float(scores[positions].min()),
            "median_score": float(np.median(scores[positions])),
            "maximum_score": float(scores[positions].max()),
            "observed": observed,
            "expected": expected,
            "aggregate_lift": float((observed + 0.5) / (expected + 0.5)),
        })

    frame = metadata.assign(embedding_norm=norm, embedding_mass=mass)
    department_frame = frame.groupby("DEPARTMENT").agg(
        products=("item_id", "size"),
        training_lines=("n_train_lines", "sum"),
        training_households=("n_train_households", "sum"),
        embedding_mass=("embedding_mass", "sum"),
        median_norm=("embedding_norm", "median"),
        mean_norm=("embedding_norm", "mean"),
    )
    department_frame["embedding_mass_share"] = (
        department_frame.embedding_mass / department_frame.embedding_mass.sum())
    department_frame["training_line_share"] = (
        department_frame.training_lines / department_frame.training_lines.sum())
    department_frame["catalogue_share"] = department_frame.products / len(frame)
    department_frame = department_frame.sort_values("embedding_mass_share", ascending=False)

    cumulative = np.cumsum(np.sort(mass)[::-1]) / mass.sum()
    products_for_mass = {
        str(level): int(np.searchsorted(cumulative, level) + 1)
        for level in (0.90, 0.95, 0.99)
    }
    top_product_indices = np.argsort(norm)[::-1][:25]
    top_products = [_item(metadata, int(index), norm) for index in top_product_indices]
    top_pair_rows = [
        _pair_row(pair, index, metadata, heldout)
        for index, pair in enumerate(pairs[:50])
    ]
    summary = {
        "method": "orientation-invariant Version-4 interaction-embedding data audit",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "data_fingerprint_sha256": blob["data_fingerprint_sha256"],
        "structure": {
            "products": int(len(phi)),
            "active_rank": rank,
            "singular_values": singular.tolist(),
            "spectral_cap_saturated": bool(np.allclose(singular, 1.0, atol=1e-8)),
            "split_half_cosines": stability_cosines.tolist(),
            "split_half_overlap": float(
                stability["split_half_mean_squared_subspace_overlap"]),
            "products_for_mass_90": products_for_mass["0.9"],
            "products_for_mass_95": products_for_mass["0.95"],
            "products_for_mass_99": products_for_mass["0.99"],
        },
        "support_relationship": {
            "spearman_norm_training_lines": float(spearmanr(
                norm, metadata.n_train_lines.to_numpy()).statistic),
            "spearman_norm_training_households": float(spearmanr(
                norm, metadata.n_train_households.to_numpy()).statistic),
        },
        "heldout_validation": {
            "test_trips": int(heldout["trips"]),
            "selected_pairs": pair_count,
            "minimum_training_lines": args.minimum_training_lines,
            "top_observed": int(top_observed.sum()),
            "top_expected": float(top_expected.sum()),
            "top_aggregate_lift": float(
                (top_observed.sum() + 0.5) / (top_expected.sum() + 0.5)),
            "top_fraction_above_null": float(np.mean(top_observed > top_expected)),
            "control_observed": int(control_observed.sum()),
            "control_expected": float(control_expected.sum()),
            "control_aggregate_lift": float(
                (control_observed.sum() + 0.5) / (control_expected.sum() + 0.5)),
            "control_fraction_above_null": float(
                np.mean(control_observed > control_expected)),
            "spearman_score_lift": float(spearmanr(scores, top_lift).statistic),
            "pairs_with_at_least_3_test_coincidences": int(np.sum(top_observed >= 3)),
            "score_deciles": deciles,
        },
        "departments": {
            str(name): {
                key: (int(value) if key in {"products", "training_lines",
                                             "training_households"} else float(value))
                for key, value in row.items()
            }
            for name, row in department_frame.iterrows()
        },
        "top_products_by_norm": top_products,
        "top_cross_affinity_pairs": top_pair_rows,
        "selection_uses_test_outcomes": False,
    }
    figures = create_plots(
        figure_dir=figure_dir, phi=phi, metadata=metadata, singular=singular,
        stability_cosines=stability_cosines, pairs=pairs, controls=controls,
        heldout=heldout, deciles=deciles, departments=department_frame)
    summary["figures"] = {key: str(value) for key, value in figures.items()}
    analysis_json.parent.mkdir(parents=True, exist_ok=True)
    analysis_json.write_text(strict_json_dumps(summary))
    write_report(output=output, figures=figures, summary=summary,
                 top_products=top_products, top_pair_rows=top_pair_rows)
    print(strict_json_dumps({
        "report": str(output),
        "analysis_json": str(analysis_json),
        "figures": {key: str(value) for key, value in figures.items()},
        "top_pair_aggregate_lift": summary["heldout_validation"]["top_aggregate_lift"],
        "control_aggregate_lift": summary["heldout_validation"]["control_aggregate_lift"],
    }), end="")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("artifacts/candidate_rank1.pt"))
    parser.add_argument("--spectral-report", type=Path,
                        default=Path("artifacts/interaction_basis_rank8.json"))
    parser.add_argument("--minimum-training-lines", type=int, default=100)
    parser.add_argument("--pairs", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--output", type=Path,
                        default=Path("paper/LEARNED_INTERACTION_EMBEDDINGS.md"))
    parser.add_argument("--figure-dir", type=Path,
                        default=Path("paper/figures/interaction_embeddings"))
    parser.add_argument("--analysis-json", type=Path,
                        default=Path("reports/learned_interaction_embeddings.json"))
    main(parser.parse_args())
