#!/usr/bin/env python3
"""Compute rigorous statistical comparison across EXP-001, EXP-003, and EXP-002-FIXED for Phase 4."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.paths import ensure_parent, research_path


def load_metrics(metrics_file: Path) -> dict | None:
    p = research_path(metrics_file)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def median(vals: list[float]) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2.0


def stdev(vals: list[float]) -> float:
    if len(vals) < 2:
        return 0.0
    m = mean(vals)
    return math.sqrt(sum((x - m) ** 2 for x in vals) / (len(vals) - 1))


def student_t_cdf(t: float, df: int) -> float:
    """Exact Student-t CDF."""
    if df <= 0:
        return 0.5
    if df == 4:
        val = t / math.sqrt(4.0 + t * t)
        cdf = 0.5 + (3.0 / 8.0) * val * (1.0 - (1.0 / 3.0) * val**2)
        return max(0.0, min(1.0, cdf))
    n_steps = 1000
    x_max = abs(t)
    if x_max == 0:
        return 0.5
    dt = x_max / n_steps
    gamma_ratio = math.gamma((df + 1) / 2) / (math.sqrt(df * math.pi) * math.gamma(df / 2))
    integral = 0.0
    for step in range(n_steps):
        u0 = step * dt
        u1 = (step + 1) * dt
        u_mid = (u0 + u1) / 2.0
        f0 = (1.0 + (u0**2) / df) ** (-(df + 1) / 2.0)
        f1 = (1.0 + (u1**2) / df) ** (-(df + 1) / 2.0)
        f_mid = (1.0 + (u_mid**2) / df) ** (-(df + 1) / 2.0)
        integral += (dt / 6.0) * (f0 + 4.0 * f_mid + f1)
    cdf = 0.5 + (1.0 if t > 0 else -1.0) * gamma_ratio * integral
    return max(0.0, min(1.0, cdf))


def paired_t_test(a: list[float], b: list[float]) -> dict:
    """Compute paired Student's t-test, 95% CI, and Cohen's d."""
    if len(a) != len(b) or len(a) < 2:
        return {"t_stat": 0.0, "df": len(a)-1 if a else 0, "p_value": 1.0, "mean_diff": 0.0, "ci95": (0.0, 0.0), "cohens_d": 0.0}
    diffs = [x - y for x, y in zip(a, b)]
    n = len(diffs)
    df = n - 1
    m_diff = mean(diffs)
    s_diff = stdev(diffs)
    if s_diff == 0:
        return {"t_stat": float("inf") if m_diff != 0 else 0.0, "df": df, "p_value": 0.0 if m_diff != 0 else 1.0, "mean_diff": m_diff, "ci95": (m_diff, m_diff), "cohens_d": float("inf") if m_diff != 0 else 0.0}
    
    t_stat = m_diff / (s_diff / math.sqrt(n))
    cdf_val = student_t_cdf(abs(t_stat), df)
    p_val = 2.0 * (1.0 - cdf_val)
    
    t_crit_table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.7764, 5: 2.571, 9: 2.262}
    t_crit = t_crit_table.get(df, 1.96)
    margin = t_crit * (s_diff / math.sqrt(n))
    ci95 = (m_diff - margin, m_diff + margin)
    cohens_d = m_diff / s_diff
    
    return {
        "t_stat": round(t_stat, 4),
        "df": df,
        "p_value": round(p_val, 6),
        "mean_diff": round(m_diff, 6),
        "stdev_diff": round(s_diff, 6),
        "ci95": (round(ci95[0], 6), round(ci95[1], 6)),
        "cohens_d": round(cohens_d, 4)
    }


def wilcoxon_signed_rank_exact(a: list[float], b: list[float]) -> dict:
    """Compute exact Wilcoxon signed-rank test for small samples (n <= 10)."""
    diffs = [x - y for x, y in zip(a, b)]
    non_zero = [(abs(d), 1 if d > 0 else -1) for d in diffs if d != 0]
    n = len(non_zero)
    if n == 0:
        return {"w_stat": 0.0, "p_value": 1.0, "n_pairs": 0}
    
    non_zero.sort(key=lambda x: x[0])
    ranks = []
    i = 0
    while i < n:
        j = i
        while j < n and non_zero[j][0] == non_zero[i][0]:
            j += 1
        avg_rank = sum(range(i + 1, j + 1)) / (j - i)
        for k in range(i, j):
            ranks.append((avg_rank, non_zero[k][1]))
        i = j
        
    w_plus = sum(r for r, s in ranks if s > 0)
    w_minus = sum(r for r, s in ranks if s < 0)
    w_stat = min(w_plus, w_minus)
    
    total_permutations = 2 ** n
    all_ranks = [r for r, _ in ranks]
    count_less_or_equal = 0
    for mask in range(total_permutations):
        current_w = sum(all_ranks[bit] for bit in range(n) if (mask & (1 << bit)))
        if current_w <= w_stat:
            count_less_or_equal += 1
    p_val = min(1.0, 2.0 * (count_less_or_equal / total_permutations))
    
    return {
        "w_stat": round(w_stat, 2),
        "w_plus": round(w_plus, 2),
        "w_minus": round(w_minus, 2),
        "p_value": round(p_val, 6),
        "n_pairs": n
    }


def bootstrap_ci(a: list[float], b: list[float], n_boot: int = 10000, seed: int = 42, alpha: float = 0.05) -> tuple[float, float]:
    """Compute deterministic bootstrap confidence interval for mean paired difference."""
    if len(a) != len(b) or len(a) < 2:
        return 0.0, 0.0
    diffs = [x - y for x, y in zip(a, b)]
    n = len(diffs)
    rng = random.Random(seed)
    boot_means = []
    for _ in range(n_boot):
        sample = [rng.choice(diffs) for _ in range(n)]
        boot_means.append(mean(sample))
    boot_means.sort()
    low_idx = int(math.floor(n_boot * (alpha / 2.0)))
    high_idx = int(math.ceil(n_boot * (1.0 - alpha / 2.0))) - 1
    return round(boot_means[low_idx], 6), round(boot_means[high_idx], 6)


def format_stat(vals: list[float], is_percent: bool = True) -> str:
    m = mean(vals)
    s = stdev(vals)
    if is_percent:
        return f"{m*100:.2f}% ± {s*100:.2f}%"
    return f"{m:.4f} ± {s:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Phase 4 scientific multi-seed statistical validation.")
    parser.add_argument("--exp001-prefix", default="EXP-001")
    parser.add_argument("--exp003-prefix", default="EXP-003")
    parser.add_argument("--exp002-prefix", default="EXP-002-FIXED")
    parser.add_argument("--seeds", nargs="+", default=["42", "123", "2026", "7", "99"])
    parser.add_argument("--output", default="results/tables/THREE_WAY_SCIENTIFIC_COMPARISON.md")
    args = parser.parse_args()

    metrics_dir = research_path("results/metrics")
    print("=" * 80)
    print(f"COMPUTING MULTI-SEED STATISTICAL VALIDATION (Seeds: {args.seeds})")
    print("=" * 80)

    raw_results = {"seeds": args.seeds, "experiments": {}}
    metric_keys = ["map50", "map50_95", "precision", "recall", "f1"]
    
    # Ingest data
    data = {"EXP-001": {k: [] for k in metric_keys}, "EXP-003": {k: [] for k in metric_keys}, "EXP-002-FIXED": {k: [] for k in metric_keys}}
    seed_table_rows = []

    for seed in args.seeds:
        row = {"seed": seed}
        for exp_key, prefix in [("EXP-001", args.exp001_prefix), ("EXP-003", args.exp003_prefix), ("EXP-002-FIXED", args.exp002_prefix)]:
            candidates = [
                metrics_dir / f"{prefix}_seed_{seed}-test.json",
                metrics_dir / f"{prefix}_seed_{seed}_test.json",
                metrics_dir / f"{prefix}-seed_{seed}-test.json",
                metrics_dir / f"{prefix}_{seed}_test.json",
                metrics_dir / f"{prefix}_seed_{seed}.json",
            ]
            if seed == "42":
                candidates.extend([
                    metrics_dir / f"{prefix}-test.json",
                    metrics_dir / f"{prefix}_test.json",
                    metrics_dir / f"{prefix}_test_metrics.json",
                ])

            loaded = None
            found_path = None
            for cand in candidates:
                if cand.exists():
                    loaded = load_metrics(cand)
                    if loaded and "overall_metrics" in loaded:
                        found_path = cand
                        break

            if loaded and "overall_metrics" in loaded:
                for k in metric_keys:
                    val = loaded["overall_metrics"].get(k, 0.0)
                    data[exp_key][k].append(val)
                    row[f"{exp_key}_{k}"] = val
            else:
                for k in metric_keys:
                    row[f"{exp_key}_{k}"] = None
                print(f"[WARN] Metrics not found for {exp_key} (seed {seed}) in {[c.name for c in candidates]}")
        seed_table_rows.append(row)

    # 1. Save results/metrics/phase4_multiseed_raw_results.json
    raw_json_out = research_path("results/metrics/phase4_multiseed_raw_results.json")
    ensure_parent(raw_json_out).write_text(json.dumps({"seeds": args.seeds, "seed_data": seed_table_rows}, indent=2), encoding="utf-8")
    print(f"Saved raw seed metrics JSON: {raw_json_out}")

    # 2. Save results/tables/phase4_seed_level_results.csv
    csv_seed_out = research_path("results/tables/phase4_seed_level_results.csv")
    ensure_parent(csv_seed_out)
    with csv_seed_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Seed", "EXP-003 mAP50", "EXP-002-FIXED mAP50", "Paired Diff mAP50", "EXP-003 F1", "EXP-002-FIXED F1", "Paired Diff F1", "EXP-003 mAP50-95", "EXP-002-FIXED mAP50-95", "Paired Diff mAP50-95"])
        for r in seed_table_rows:
            m3 = r.get("EXP-003_map50")
            m2 = r.get("EXP-002-FIXED_map50")
            diff_m = (m3 - m2) if (m3 is not None and m2 is not None) else ""
            
            f3 = r.get("EXP-003_f1")
            f2 = r.get("EXP-002-FIXED_f1")
            diff_f = (f3 - f2) if (f3 is not None and f2 is not None) else ""
            
            mm3 = r.get("EXP-003_map50_95")
            mm2 = r.get("EXP-002-FIXED_map50_95")
            diff_mm = (mm3 - mm2) if (mm3 is not None and mm2 is not None) else ""
            
            writer.writerow([
                r["seed"],
                f"{m3:.4f}" if m3 is not None else "",
                f"{m2:.4f}" if m2 is not None else "",
                f"{diff_m:.4f}" if diff_m != "" else "",
                f"{f3:.4f}" if f3 is not None else "",
                f"{f2:.4f}" if f2 is not None else "",
                f"{diff_f:.4f}" if diff_f != "" else "",
                f"{mm3:.4f}" if mm3 is not None else "",
                f"{mm2:.4f}" if mm2 is not None else "",
                f"{diff_mm:.4f}" if diff_mm != "" else ""
            ])
    print(f"Saved seed-level table CSV: {csv_seed_out}")

    # Compute Statistical Summaries
    stats_dict = {}
    summary_rows = []
    
    names = {
        "map50": "mAP@0.50",
        "map50_95": "mAP@0.50:0.95",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1-Score",
    }

    for k in metric_keys:
        v3 = data["EXP-003"][k]
        v2 = data["EXP-002-FIXED"][k]
        v1 = data["EXP-001"][k]
        
        stat_item = {
            "metric": names[k],
            "exp003": {
                "mean": round(mean(v3), 6),
                "std": round(stdev(v3), 6),
                "median": round(median(v3), 6),
                "min": round(min(v3), 6) if v3 else None,
                "max": round(max(v3), 6) if v3 else None,
            },
            "exp002_fixed": {
                "mean": round(mean(v2), 6),
                "std": round(stdev(v2), 6),
                "median": round(median(v2), 6),
                "min": round(min(v2), 6) if v2 else None,
                "max": round(max(v2), 6) if v2 else None,
            }
        }
        
        if len(v3) >= 2 and len(v2) >= 2 and len(v3) == len(v2):
            t_res = paired_t_test(v3, v2)
            w_res = wilcoxon_signed_rank_exact(v3, v2)
            boot_ci_low, boot_ci_high = bootstrap_ci(v3, v2, n_boot=10000, seed=42)
            
            stat_item["paired_comparison"] = {
                "mean_paired_difference": t_res["mean_diff"],
                "std_paired_difference": t_res["stdev_diff"],
                "student_t_test": {
                    "t_statistic": t_res["t_stat"],
                    "df": t_res["df"],
                    "p_value": t_res["p_value"],
                    "ci95_student_t": t_res["ci95"]
                },
                "wilcoxon_signed_rank_test": {
                    "w_statistic": w_res["w_stat"],
                    "p_value": w_res["p_value"],
                    "n_pairs": w_res["n_pairs"]
                },
                "bootstrap_ci95": (boot_ci_low, boot_ci_high),
                "cohens_d": t_res["cohens_d"]
            }
            summary_rows.append([
                names[k],
                f"{stat_item['exp003']['mean']:.4f} ± {stat_item['exp003']['std']:.4f}",
                f"{stat_item['exp002_fixed']['mean']:.4f} ± {stat_item['exp002_fixed']['std']:.4f}",
                f"{t_res['mean_diff']:+.4f} ({t_res['mean_diff']*100:+.2f} pp)",
                f"[{t_res['ci95'][0]:+.4f}, {t_res['ci95'][1]:+.4f}]",
                f"[{boot_ci_low:+.4f}, {boot_ci_high:+.4f}]",
                f"t={t_res['t_stat']:.3f}, p={t_res['p_value']:.4f}",
                f"W={w_res['w_stat']:.1f}, p={w_res['p_value']:.4f}",
                f"{t_res['cohens_d']:.2f}"
            ])
        else:
            summary_rows.append([
                names[k],
                f"{mean(v3):.4f}",
                f"{mean(v2):.4f}",
                f"{mean(v3) - mean(v2):+.4f}" if v3 and v2 else "N/A",
                "N/A", "N/A", "N/A", "N/A", "N/A"
            ])
            
        stats_dict[k] = stat_item

    # 3. Save results/analysis/phase4_statistical_results.json
    analysis_json_out = research_path("results/analysis/phase4_statistical_results.json")
    ensure_parent(analysis_json_out).write_text(json.dumps({"seeds": args.seeds, "statistics": stats_dict}, indent=2), encoding="utf-8")
    print(f"Saved statistical analysis JSON: {analysis_json_out}")

    # 4. Save results/tables/phase4_statistical_summary.csv
    csv_stats_out = research_path("results/tables/phase4_statistical_summary.csv")
    ensure_parent(csv_stats_out)
    with csv_stats_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "EXP-003 (Mean ± Std)", "EXP-002-FIXED (Mean ± Std)", "Mean Paired Diff", "95% Student-t CI", "95% Bootstrap CI", "Paired t-test", "Wilcoxon Signed-Rank", "Cohen's d"])
        writer.writerows(summary_rows)
    print(f"Saved statistical summary CSV: {csv_stats_out}")

    # 5. Save results/tables/phase4_publication_table.tex
    tex_out = research_path("results/tables/phase4_publication_table.tex")
    ensure_parent(tex_out)
    tex_lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Multi-Seed Statistical Validation ($n=5$ seeds: 42, 123, 2026, 7, 99) comparing size-matched random-frame control (EXP-003) versus sequence-exclusive group split (EXP-002-FIXED).}",
        "\\label{tab:multiseed_validation}",
        "\\begin{tabular}{lcccccc}",
        "\\hline",
        "\\textbf{Metric} & \\textbf{EXP-003 (Random)} & \\textbf{EXP-002-FIXED (Group)} & \\textbf{Mean $\\Delta$ (pp)} & \\textbf{95\\% Student-t CI} & \\textbf{Paired $t$-test} & \\textbf{Cohen's $d$} \\\\",
        "\\hline"
    ]
    for row in summary_rows:
        m_name = row[0]
        c_rand = row[1]
        c_grp = row[2]
        c_diff = row[3]
        c_ci = row[4]
        c_t = row[6]
        c_d = row[8]
        tex_lines.append(f"{m_name} & {c_rand} & {c_grp} & {c_diff} & {c_ci} & {c_t} & {c_d} \\\\")
    tex_lines.extend([
        "\\hline",
        "\\end{tabular}",
        "\\end{table*}"
    ])
    tex_out.write_text("\n".join(tex_lines), encoding="utf-8")
    print(f"Saved LaTeX publication table: {tex_out}")

    # 6. Generate Markdown Report
    lines = []
    lines.append("# Controlled Multi-Seed Benchmark: Disentangling Dataset Size vs. Temporal Data Leakage\n")
    lines.append(f"**Seeds Evaluated**: {', '.join(args.seeds)}\n")
    lines.append("## 1. Descriptive Summary\n")
    lines.append("| Metric | EXP-003 (Size-Matched Random) | EXP-002-FIXED (Sequence-Isolated) | Mean Paired Diff | 95% Student-t CI | 95% Bootstrap CI | Paired t-test | Wilcoxon Signed-Rank | Cohen's $d$ |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for r in summary_rows:
        lines.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} | {r[6]} | {r[7]} | {r[8]} |")
        
    lines.append("\n## 2. Raw Seed-by-Seed Results\n")
    lines.append("| Seed | EXP-003 mAP@50 | EXP-002-FIXED mAP@50 | Paired Diff ($\Delta$) | EXP-003 F1 | EXP-002-FIXED F1 | Paired Diff ($\Delta$) |")
    lines.append("| :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for r in seed_table_rows:
        m3 = r.get("EXP-003_map50")
        m2 = r.get("EXP-002-FIXED_map50")
        diff_m = f"{(m3 - m2):+.4f} ({(m3 - m2)*100:+.2f} pp)" if (m3 is not None and m2 is not None) else "N/A"
        f3 = r.get("EXP-003_f1")
        f2 = r.get("EXP-002-FIXED_f1")
        diff_f = f"{(f3 - f2):+.4f} ({(f3 - f2)*100:+.2f} pp)" if (f3 is not None and f2 is not None) else "N/A"
        
        s_m3 = f"{m3:.4f}" if m3 is not None else "N/A"
        s_m2 = f"{m2:.4f}" if m2 is not None else "N/A"
        s_f3 = f"{f3:.4f}" if f3 is not None else "N/A"
        s_f2 = f"{f2:.4f}" if f2 is not None else "N/A"
        lines.append(f"| {r['seed']} | {s_m3} | {s_m2} | {diff_m} | {s_f3} | {s_f2} | {diff_f} |")

    lines.append("\n## 3. Scientific Attribution Breakdown\n")
    lines.append("1. **Sequence Isolation Effect**: Quantifies the true generalization drop when moving from random frame-splitting to sequence-exclusive video splitting at identical training set sizes (3,866 images).")
    lines.append("2. **Statistical Significance**: Multi-seed analysis isolates training-seed variability across matched random seeds.\n")

    out_file = ensure_parent(research_path(args.output))
    out_file.write_text("\n".join(lines), encoding="utf-8")
    print(f"Generated comparison report: {out_file}")


if __name__ == "__main__":
    main()
