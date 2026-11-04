#!/usr/bin/env python3
"""
Generate publication-quality SVG figures for Phase 9 Multi-Seed Reproducibility.
No external dependencies required (pure standard Python).
"""
import json
from pathlib import Path

def generate_svg_bar_chart(stats: dict, output_path: Path):
    """Generate Fig 1: Grouped bar chart of mAP50 (Mean ± 1 SD) across test sets."""
    pops = [
        ("new_test", "New Mobile (280)", stats["new_test"]["map50"]),
        ("old_test", "Old CCTV (829)", stats["old_test"]["map50"]),
        ("combined_test", "Combined (1,109)", stats["combined_test"]["map50"]),
    ]
    
    # SVG canvas dimensions
    w, h = 800, 520
    margin = {"top": 80, "right": 50, "bottom": 80, "left": 90}
    chart_w = w - margin["left"] - margin["right"]
    chart_h = h - margin["top"] - margin["bottom"]
    
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" style="background-color: #ffffff; font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif;">',
        f'<!-- Title -->',
        f'<text x="{w/2}" y="35" text-anchor="middle" font-size="18" font-weight="700" fill="#1e293b">Cross-Domain Generalization Across Independent Seeds (N=3)</text>',
        f'<text x="{w/2}" y="58" text-anchor="middle" font-size="13" fill="#64748b">mAP@0.50 Mean ± 1 Sample Standard Deviation (Seeds 42, 123, 2026)</text>',
    ]
    
    # Grid lines and Y-axis labels (0 to 100%)
    for y_pct in range(0, 101, 20):
        y_pos = margin["top"] + chart_h * (1.0 - y_pct / 100.0)
        svg.append(f'<line x1="{margin["left"]}" y1="{y_pos:.1f}" x2="{w - margin["right"]}" y2="{y_pos:.1f}" stroke="#e2e8f0" stroke-width="1" stroke-dasharray="{"0" if y_pct==0 else "4,4"}"/>')
        svg.append(f'<text x="{margin["left"] - 12}" y="{y_pos + 4:.1f}" text-anchor="end" font-size="12" fill="#64748b">{y_pct}%</text>')
        
    # Y-axis Title
    svg.append(f'<text x="25" y="{margin["top"] + chart_h/2}" text-anchor="middle" font-size="13" font-weight="600" fill="#334155" transform="rotate(-90, 25, {margin["top"] + chart_h/2})">mAP @ IoU 0.50 (%)</text>')
    
    # Draw bars for each population
    n_groups = len(pops)
    group_width = chart_w / n_groups
    bar_width = 54
    gap = 14
    
    for i, (pop_key, pop_label, m) in enumerate(pops):
        group_center = margin["left"] + group_width * (i + 0.5)
        
        # EXP-NEW bar
        val_new = m["exp_new"]["mean"] * 100
        std_new = m["exp_new"]["std"] * 100
        bar_h_new = (val_new / 100.0) * chart_h
        x_new = group_center - bar_width - gap/2
        y_new = margin["top"] + chart_h - bar_h_new
        
        svg.append(f'<!-- EXP-NEW bar for {pop_key} -->')
        svg.append(f'<rect x="{x_new:.1f}" y="{y_new:.1f}" width="{bar_width}" height="{bar_h_new:.1f}" fill="#3b82f6" rx="4" />')
        svg.append(f'<text x="{x_new + bar_width/2:.1f}" y="{y_new - 12:.1f}" text-anchor="middle" font-size="12" font-weight="600" fill="#1d4ed8">{val_new:.1f}%</text>')
        
        # EXP-NEW error bar
        y_top_new = margin["top"] + chart_h - ((val_new + std_new) / 100.0) * chart_h
        y_bot_new = margin["top"] + chart_h - (max(0, val_new - std_new) / 100.0) * chart_h
        svg.append(f'<line x1="{x_new + bar_width/2:.1f}" y1="{y_top_new:.1f}" x2="{x_new + bar_width/2:.1f}" y2="{y_bot_new:.1f}" stroke="#1e3a8a" stroke-width="2"/>')
        svg.append(f'<line x1="{x_new + bar_width/2 - 6:.1f}" y1="{y_top_new:.1f}" x2="{x_new + bar_width/2 + 6:.1f}" y2="{y_top_new:.1f}" stroke="#1e3a8a" stroke-width="2"/>')
        svg.append(f'<line x1="{x_new + bar_width/2 - 6:.1f}" y1="{y_bot_new:.1f}" x2="{x_new + bar_width/2 + 6:.1f}" y2="{y_bot_new:.1f}" stroke="#1e3a8a" stroke-width="2"/>')

        # EXP-COMBINED bar
        val_comb = m["exp_combined"]["mean"] * 100
        std_comb = m["exp_combined"]["std"] * 100
        bar_h_comb = (val_comb / 100.0) * chart_h
        x_comb = group_center + gap/2
        y_comb = margin["top"] + chart_h - bar_h_comb
        
        svg.append(f'<!-- EXP-COMBINED bar for {pop_key} -->')
        svg.append(f'<rect x="{x_comb:.1f}" y="{y_comb:.1f}" width="{bar_width}" height="{bar_h_comb:.1f}" fill="#10b981" rx="4" />')
        svg.append(f'<text x="{x_comb + bar_width/2:.1f}" y="{y_comb - 12:.1f}" text-anchor="middle" font-size="12" font-weight="700" fill="#047857">{val_comb:.1f}%</text>')
        
        # EXP-COMBINED error bar
        y_top_comb = margin["top"] + chart_h - (min(100, val_comb + std_comb) / 100.0) * chart_h
        y_bot_comb = margin["top"] + chart_h - (max(0, val_comb - std_comb) / 100.0) * chart_h
        svg.append(f'<line x1="{x_comb + bar_width/2:.1f}" y1="{y_top_comb:.1f}" x2="{x_comb + bar_width/2:.1f}" y2="{y_bot_comb:.1f}" stroke="#064e3b" stroke-width="2"/>')
        svg.append(f'<line x1="{x_comb + bar_width/2 - 6:.1f}" y1="{y_top_comb:.1f}" x2="{x_comb + bar_width/2 + 6:.1f}" y2="{y_top_comb:.1f}" stroke="#064e3b" stroke-width="2"/>')
        svg.append(f'<line x1="{x_comb + bar_width/2 - 6:.1f}" y1="{y_bot_comb:.1f}" x2="{x_comb + bar_width/2 + 6:.1f}" y2="{y_bot_comb:.1f}" stroke="#064e3b" stroke-width="2"/>')

        # X-axis label
        svg.append(f'<text x="{group_center:.1f}" y="{h - margin["bottom"] + 24}" text-anchor="middle" font-size="13" font-weight="600" fill="#1e293b">{pop_label}</text>')
        
        # Difference tag
        diff = val_comb - val_new
        color = "#10b981" if diff >= 0 else "#ef4444"
        sign = "+" if diff >= 0 else ""
        svg.append(f'<rect x="{group_center - 38:.1f}" y="{h - margin["bottom"] + 34}" width="76" height="20" rx="10" fill="{color}" fill-opacity="0.15"/>')
        svg.append(f'<text x="{group_center:.1f}" y="{h - margin["bottom"] + 48}" text-anchor="middle" font-size="11" font-weight="700" fill="{color}">{sign}{diff:.1f}%</text>')

    # Legend
    leg_x = w - margin["right"] - 290
    leg_y = margin["top"] + 15
    svg.append(f'<rect x="{leg_x}" y="{leg_y}" width="280" height="42" rx="6" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/>')
    svg.append(f'<rect x="{leg_x + 14}" y="{leg_y + 14}" width="14" height="14" rx="3" fill="#3b82f6"/>')
    svg.append(f'<text x="{leg_x + 34}" y="{leg_y + 26}" font-size="11" font-weight="600" fill="#334155">EXP-NEW (Mobile)</text>')
    svg.append(f'<rect x="{leg_x + 145}" y="{leg_y + 14}" width="14" height="14" rx="3" fill="#10b981"/>')
    svg.append(f'<text x="{leg_x + 165}" y="{leg_y + 26}" font-size="11" font-weight="600" fill="#334155">EXP-COMBINED</text>')
    
    svg.append('</svg>')
    output_path.write_text("\n".join(svg))
    print(f"Generated {output_path}")

def generate_svg_seed_strip_plot(seeds_csv: Path, output_path: Path):
    """Generate Fig 2: Individual seed data points showing variance and clustering."""
    lines = seeds_csv.read_text().strip().split("\n")[1:]
    data = []
    for l in lines:
        parts = l.split(",")
        data.append({
            "model": parts[0],
            "seed": parts[1],
            "pop": parts[2],
            "map50": float(parts[6]) * 100
        })
        
    w, h = 800, 480
    margin = {"top": 70, "right": 50, "bottom": 80, "left": 90}
    chart_w = w - margin["left"] - margin["right"]
    chart_h = h - margin["top"] - margin["bottom"]
    
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" style="background-color: #ffffff; font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif;">',
        f'<text x="{w/2}" y="32" text-anchor="middle" font-size="18" font-weight="700" fill="#1e293b">Per-Seed Stability Distribution (Seeds 42, 123, 2026)</text>',
        f'<text x="{w/2}" y="52" text-anchor="middle" font-size="13" fill="#64748b">mAP@0.50 (%) per individual run showing low variance in EXP-COMBINED</text>',
    ]
    
    for y_pct in range(0, 101, 20):
        y_pos = margin["top"] + chart_h * (1.0 - y_pct / 100.0)
        svg.append(f'<line x1="{margin["left"]}" y1="{y_pos:.1f}" x2="{w - margin["right"]}" y2="{y_pos:.1f}" stroke="#e2e8f0" stroke-width="1" stroke-dasharray="{"0" if y_pct==0 else "4,4"}"/>')
        svg.append(f'<text x="{margin["left"] - 12}" y="{y_pos + 4:.1f}" text-anchor="end" font-size="12" fill="#64748b">{y_pct}%</text>')
        
    svg.append(f'<text x="25" y="{margin["top"] + chart_h/2}" text-anchor="middle" font-size="13" font-weight="600" fill="#334155" transform="rotate(-90, 25, {margin["top"] + chart_h/2})">mAP @ IoU 0.50 (%)</text>')
    
    pops = [("new_test", "New Mobile (280)"), ("old_test", "Old CCTV (829)"), ("combined_test", "Combined (1,109)")]
    n_groups = len(pops)
    group_width = chart_w / n_groups
    
    seed_offsets = {"42": -18, "123": 0, "2026": 18}
    seed_symbols = {"42": "●", "123": "▲", "2026": "■"}
    
    for i, (pop_key, pop_label) in enumerate(pops):
        group_center = margin["left"] + group_width * (i + 0.5)
        
        # Center line for each group
        svg.append(f'<line x1="{group_center}" y1="{margin["top"]}" x2="{group_center}" y2="{h - margin["bottom"]}" stroke="#f1f5f9" stroke-width="1"/>')
        svg.append(f'<text x="{group_center:.1f}" y="{h - margin["bottom"] + 30}" text-anchor="middle" font-size="13" font-weight="600" fill="#1e293b">{pop_label}</text>')
        
        # EXP-NEW column (offset -50)
        col_new = group_center - 50
        # EXP-COMBINED column (offset +50)
        col_comb = group_center + 50
        
        svg.append(f'<text x="{col_new}" y="{h - margin["bottom"] + 52}" text-anchor="middle" font-size="11" font-weight="600" fill="#3b82f6">EXP-NEW</text>')
        svg.append(f'<text x="{col_comb}" y="{h - margin["bottom"] + 52}" text-anchor="middle" font-size="11" font-weight="600" fill="#10b981">EXP-COMB</text>')
        
        # Plot points
        for row in data:
            if row["pop"] == pop_key:
                val = row["map50"]
                y_pos = margin["top"] + chart_h * (1.0 - val / 100.0)
                is_comb = row["model"] == "EXP-COMBINED"
                x_base = col_comb if is_comb else col_new
                x_pos = x_base + seed_offsets[row["seed"]] * 0.7
                color = "#059669" if is_comb else "#2563eb"
                
                svg.append(f'<circle cx="{x_pos:.1f}" cy="{y_pos:.1f}" r="6" fill="{color}" stroke="#ffffff" stroke-width="1.5" />')
                svg.append(f'<text x="{x_pos:.1f}" y="{y_pos - 9:.1f}" text-anchor="middle" font-size="10" font-weight="600" fill="{color}">{val:.1f}%</text>')

    # Legend
    leg_x = w - margin["right"] - 220
    leg_y = margin["top"] + 15
    svg.append(f'<rect x="{leg_x}" y="{leg_y}" width="210" height="60" rx="6" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/>')
    svg.append(f'<circle cx="{leg_x + 20}" cy="{leg_y + 20}" r="5" fill="#3b82f6"/>')
    svg.append(f'<text x="{leg_x + 35}" y="{leg_y + 24}" font-size="11" fill="#334155">EXP-NEW Seeds</text>')
    svg.append(f'<circle cx="{leg_x + 20}" cy="{leg_y + 42}" r="5" fill="#10b981"/>')
    svg.append(f'<text x="{leg_x + 35}" y="{leg_y + 46}" font-size="11" fill="#334155">EXP-COMBINED Seeds</text>')
    
    svg.append('</svg>')
    output_path.write_text("\n".join(svg))
    print(f"Generated {output_path}")

def main():
    root = Path(".")
    p9_dir = root / "results/phase9_multiseed"
    fig_dir = p9_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    stats_json = p9_dir / "statistical_comparison.json"
    seeds_csv = p9_dir / "seed_level_metrics.csv"
    
    if stats_json.exists():
        stats = json.loads(stats_json.read_text())
        generate_svg_bar_chart(stats, fig_dir / "fig1_cross_domain_map50_comparison.svg")
        
    if seeds_csv.exists():
        generate_svg_seed_strip_plot(seeds_csv, fig_dir / "fig2_seed_variance_strip_plot.svg")

if __name__ == "__main__":
    main()
