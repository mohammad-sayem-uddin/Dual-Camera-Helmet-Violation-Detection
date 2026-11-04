#!/usr/bin/env python3
"""Format synthetic benchmark results into Markdown and CSV tables."""
import json
from pathlib import Path

def main():
    bench_file = Path("results/phase10_temporal/synthetic_benchmark_results.json")
    data = json.loads(bench_file.read_text())
    
    rows = []
    header = ["Aggregator", "Scenario", "Precision", "Recall", "False Alerts", "Delay (frames)", "Delay (sec)", "Stability", "Abstention Rate"]
    
    for method, scenarios in data.items():
        for s_name, res in scenarios.items():
            m = res["metrics"]
            rows.append([
                method,
                s_name,
                f"{m['alert_precision']:.2f}",
                f"{m['alert_recall']:.2f}",
                str(m['false_alerts']),
                f"{m['avg_confirmation_delay_frames']:.1f}",
                f"{m['avg_confirmation_delay_seconds']:.3f}s",
                f"{m['state_stability_score']:.3f}",
                f"{m['abstention_rate']:.3f}"
            ])
            
    # Markdown
    md_lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for r in rows:
        md_lines.append("| " + " | ".join(r) + " |")
    
    table_md = "\n".join(md_lines)
    Path("results/phase10_temporal/benchmark_comparison_table.md").write_text(table_md)
    
    # CSV
    csv_lines = [",".join(header)]
    for r in rows:
        csv_lines.append(",".join(r))
    Path("results/phase10_temporal/benchmark_comparison_table.csv").write_text("\n".join(csv_lines))
    print("Exported benchmark_comparison_table.md and .csv!")

if __name__ == "__main__":
    main()
