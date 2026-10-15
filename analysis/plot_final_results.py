#!/usr/bin/env python3
"""Render the final-report F1 comparison as a dependency-free SVG."""

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "final_report_metrics.csv"
OUTPUT = ROOT / "docs" / "final-results-summary.svg"

WIDTH, HEIGHT = 1440, 720
COLORS = {
    "background": "#FFFFFF",
    "baseline": "#94A3B8",
    "look": "#2563EB",
    "positive": "#0F766E",
    "negative": "#C2410C",
    "text": "#172033",
    "muted": "#64748B",
    "grid": "#E2E8F0",
    "panel": "#F8FAFC",
}


def load_results():
    values = {}
    with SOURCE.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            values[(row["task"], row["temporal_encoder"], row["look_feature"])] = float(row["f1"])
    return values


def line(x1, y1, x2, y2, **attrs):
    settings = " ".join(f'{key.replace("_", "-")}="{value}"' for key, value in attrs.items())
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" {settings}/>'


def text(x, y, value, **attrs):
    settings = " ".join(f'{key.replace("_", "-")}="{value}"' for key, value in attrs.items())
    return f'<text x="{x}" y="{y}" {settings}>{value}</text>'


def panel(values, *, x, title, task, domain):
    parts = [f'<rect x="{x}" y="205" width="585" height="390" rx="22" fill="{COLORS["panel"]}"/>']
    parts.append(text(x + 36, 253, title, fill=COLORS["text"], font_size="23", font_weight="700"))
    parts.append(text(x + 36, 283, "F1 score", fill=COLORS["muted"], font_size="14"))

    plot_left, plot_right = x + 165, x + 500
    low, high = domain

    def scale(value):
        return plot_left + (value - low) / (high - low) * (plot_right - plot_left)

    for i in range(4):
        tick = low + i * (high - low) / 3
        tick_x = scale(tick)
        parts.append(line(tick_x, 318, tick_x, 530, stroke=COLORS["grid"], stroke_width="1"))
        parts.append(text(tick_x, 558, f"{tick:.2f}", fill=COLORS["muted"], font_size="13", text_anchor="middle"))

    for model, y in (("Temporal GAT", 370), ("GRU", 475)):
        baseline = values[(task, model, "no")]
        with_look = values[(task, model, "yes")]
        delta = with_look - baseline
        baseline_x, look_x = scale(baseline), scale(with_look)

        parts.append(text(x + 36, y + 5, model, fill=COLORS["text"], font_size="16", font_weight="600"))
        parts.append(line(baseline_x, y, look_x, y, stroke=COLORS["grid"], stroke_width="10", stroke_linecap="round"))
        parts.append(f'<circle cx="{baseline_x:.1f}" cy="{y}" r="10" fill="{COLORS["baseline"]}" stroke="#FFFFFF" stroke-width="3"/>')
        parts.append(f'<circle cx="{look_x:.1f}" cy="{y}" r="11" fill="{COLORS["look"]}" stroke="#FFFFFF" stroke-width="3"/>')
        parts.append(text(baseline_x, y + 32, f"{baseline:.4f}", fill=COLORS["muted"], font_size="13", text_anchor="middle"))
        parts.append(text(look_x, y - 20, f"{with_look:.4f}", fill=COLORS["look"], font_size="14", font_weight="700", text_anchor="middle"))
        parts.append(
            text(
                x + 548,
                y + 5,
                f"{delta:+.4f}",
                fill=COLORS["positive"] if delta >= 0 else COLORS["negative"],
                font_size="15",
                font_weight="700",
                text_anchor="middle",
            )
        )
    return parts


def main():
    values = load_results()
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" role="img">',
        "<title>Final report F1 results</title>",
        "<desc>Comparison of Temporal GAT and GRU with and without the look feature for crossing intention and current action prediction.</desc>",
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="{COLORS["background"]}"/>',
        '<g font-family="Inter, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif">',
        text(80, 82, "Final Report Results", fill=COLORS["text"], font_size="34", font_weight="750"),
        text(80, 120, "Effect of adding the per-frame look feature · higher F1 is better", fill=COLORS["muted"], font_size="17"),
        f'<circle cx="1040" cy="89" r="8" fill="{COLORS["baseline"]}"/>',
        text(1057, 95, "Without look", fill=COLORS["muted"], font_size="15"),
        f'<circle cx="1200" cy="89" r="8" fill="{COLORS["look"]}"/>',
        text(1217, 95, "With look", fill=COLORS["muted"], font_size="15"),
    ]
    svg.extend(panel(values, x=80, title="Crossing intention", task="intention", domain=(0.78, 0.84)))
    svg.extend(panel(values, x=775, title="Current action", task="action", domain=(0.72, 0.765)))
    svg.extend(
        [
            text(
                80,
                656,
                "Source: results/final_report_metrics.csv · Exact F1 labels and signed changes are shown.",
                fill=COLORS["muted"],
                font_size="14",
            ),
            "</g>",
            "</svg>",
        ]
    )
    OUTPUT.write_text("\n".join(svg), encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
