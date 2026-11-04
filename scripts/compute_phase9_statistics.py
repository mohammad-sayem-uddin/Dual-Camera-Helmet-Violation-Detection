#!/usr/bin/env python3
"""
Statistical Analysis & Publication Artifacts Generator for Phase 9 Multi-Seed Experiments.

Calculates:
- Sample Mean, Standard Deviation (N-1), Min, Max, Range across Seeds 42, 123, 2026.
- Paired differences (EXP-COMBINED minus EXP-NEW).
- Cohen's d effect sizes.
- Exports publication CSVs, JSON, and comparison charts.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def calculate_stats(values: list[float]) -> dict[str, float]:
    """Computes mean, sample standard deviation, min, max, and range."""
    n = len(values)
    if n == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "range": 0.0}
    if n == 1:
        return {"mean": round(values[0], 4), "std": 0.0, "min": round(values[0], 4), "max": round(values[0], 4), "range": 0.0}

    mean_val = sum(values) / float(n)
    variance = sum((x - mean_val) ** 2 for x in values) / float(n - 1)
    std_val = math.sqrt(variance)
    min_val = min(values)
    max_val = max(values)
    range_val = max_val - min_val

    return {
        "mean": round(mean_val, 4),
        "std": round(std_val, 4),
        "min": round(min_val, 4),
        "max": round(max_val, 4),
        "range": round(range_val, 4),
    }


def compute_multiseed_analysis(
    new_results_by_seed: dict[str, dict[str, Any]],
    comb_results_by_seed: dict[str, dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = sorted(list(new_results_by_seed.keys()), key=lambda x: int(x))
    test_populations = ["new_test", "old_test", "combined_test"]
    metrics_to_track = ["precision", "recall", "f1", "map50", "map50_95"]

    # 1. Compile seed_level_metrics.csv
    csv_rows = []
    header = ["Model", "Seed", "Population", "Precision", "Recall", "F1", "mAP50", "mAP50_95"]
    csv_rows.append(",".join(header))

    data_store: dict[str, dict[str, dict[str, list[float]]]] = {
        "EXP-NEW": {pop: {m: [] for m in metrics_to_track} for pop in test_populations},
        "EXP-COMBINED": {pop: {m: [] for m in metrics_to_track} for pop in test_populations},
    }

    for seed in seeds:
        # EXP-NEW
        new_evals = new_results_by_seed[seed]["evaluations"]
        for pop in test_populations:
            ev = new_evals[pop]["overall"]
            row = ["EXP-NEW", seed, pop] + [str(ev.get(m, 0.0)) for m in metrics_to_track]
            csv_rows.append(",".join(row))
            for m in metrics_to_track:
                data_store["EXP-NEW"][pop][m].append(float(ev.get(m, 0.0)))

        # EXP-COMBINED
        comb_evals = comb_results_by_seed[seed]["evaluations"]
        for pop in test_populations:
            ev = comb_evals[pop]["overall"]
            row = ["EXP-COMBINED", seed, pop] + [str(ev.get(m, 0.0)) for m in metrics_to_track]
            csv_rows.append(",".join(row))
            for m in metrics_to_track:
                data_store["EXP-COMBINED"][pop][m].append(float(ev.get(m, 0.0)))

    (output_dir / "seed_level_metrics.csv").write_text("\n".join(csv_rows))

    # 2. Compile mean_std_summary.csv and statistical_comparison.json
    summary_rows = []
    summary_header = [
        "Population", "Metric",
        "EXP-NEW Mean", "EXP-NEW Std", "EXP-NEW Range",
        "EXP-COMBINED Mean", "EXP-COMBINED Std", "EXP-COMBINED Range",
        "Mean Diff (COMB - NEW)", "Cohen's d",
    ]
    summary_rows.append(",".join(summary_header))

    stats_report: dict[str, Any] = {}

    for pop in test_populations:
        stats_report[pop] = {}
        for m in metrics_to_track:
            new_vals = data_store["EXP-NEW"][pop][m]
            comb_vals = data_store["EXP-COMBINED"][pop][m]

            s_new = calculate_stats(new_vals)
            s_comb = calculate_stats(comb_vals)

            # Paired differences
            diffs = [c - n for c, n in zip(comb_vals, new_vals)]
            s_diff = calculate_stats(diffs)

            # Cohen's d (paired)
            cohen_d = 0.0
            if s_diff["std"] > 1e-6:
                cohen_d = round(s_diff["mean"] / s_diff["std"], 2)

            stats_report[pop][m] = {
                "exp_new": s_new,
                "exp_combined": s_comb,
                "difference": {
                    "mean_diff": s_diff["mean"],
                    "std_diff": s_diff["std"],
                    "per_seed_diffs": [round(d, 4) for d in diffs],
                    "cohens_d": cohen_d,
                },
            }

            row = [
                pop,
                m,
                f"{s_new['mean']:.4f}", f"{s_new['std']:.4f}", f"{s_new['range']:.4f}",
                f"{s_comb['mean']:.4f}", f"{s_comb['std']:.4f}", f"{s_comb['range']:.4f}",
                f"{s_diff['mean']:+.4f}", f"{cohen_d:.2f}",
            ]
            summary_rows.append(",".join(row))

    (output_dir / "mean_std_summary.csv").write_text("\n".join(summary_rows))
    (output_dir / "statistical_comparison.json").write_text(json.dumps(stats_report, indent=2))

    # 3. Generate Publication Figures if Matplotlib is available
    if HAS_MATPLOTLIB:
        fig_dir = output_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)

        # Figure 1: Cross-Domain mAP50 Comparison Bar Chart
        pops_display = ["New Mobile (280)", "Old CCTV (829)", "Combined (1,109)"]
        new_map_means = [stats_report[p]["map50"]["exp_new"]["mean"] for p in test_populations]
        new_map_stds = [stats_report[p]["map50"]["exp_new"]["std"] for p in test_populations]
        comb_map_means = [stats_report[p]["map50"]["exp_combined"]["mean"] for p in test_populations]
        comb_map_stds = [stats_report[p]["map50"]["exp_combined"]["std"] for p in test_populations]

        x = np.arange(len(pops_display))
        width = 0.35

        fig, ax = plt.subplots(figsize=(8, 5))
        rects1 = ax.bar(x - width/2, [m * 100 for m in new_map_means], width, yerr=[s * 100 for s in new_map_stds],
                        label="EXP-NEW (Mobile Only)", capsize=5, color="#3498db")
        rects2 = ax.bar(x + width/2, [m * 100 for m in comb_map_means], width, yerr=[s * 100 for s in comb_map_stds],
                        label="EXP-COMBINED (Multi-Domain)", capsize=5, color="#2ecc71")

        ax.set_ylabel("mAP@0.50 (%)", fontsize=12)
        ax.set_title("Multi-Seed Cross-Domain Performance (Mean ± 1 SD, N=3 Seeds)", fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(pops_display, fontsize=11)
        ax.legend(fontsize=11)
        ax.set_ylim(0, 105)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        for rect in rects1:
            h = rect.get_height()
            ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 4),
                        textcoords="offset points", ha="center", va="bottom", fontsize=9)
        for rect in rects2:
            h = rect.get_height()
            ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 4),
                        textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

        fig.tight_layout()
        fig.savefig(fig_dir / "fig1_cross_domain_map50_comparison.png", dpi=300)
        plt.close(fig)

    print(f"Successfully compiled Phase 9 statistical analysis in {output_dir}!")
    return stats_report


if __name__ == "__main__":
    print("Statistical analysis script loaded successfully.")
