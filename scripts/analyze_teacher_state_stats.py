#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List


SECTION_ORDER = ["activation", "teleport", "teacher_phase"]
SECTION_COLORS = {
    "activation": "#1f77b4",
    "teleport": "#d62728",
    "teacher_phase": "#2ca02c",
}

DIM_LABELS = {
    "base_rel_table_xyz": ["x", "y", "z"],
    "base_rel_object_xyz": ["x", "y", "z"],
    "eef_rel_object_xyz": ["x", "y", "z"],
    "eef_target_quat_deg": ["deg"],
    "q_arm": [f"j{i}" for i in range(7)],
    "q_hand": [f"j{i}" for i in range(16)],
}


def _load_stats(path: Path) -> Dict:
    with path.open("r") as f:
        return json.load(f)


def _metric_names(stats: Dict) -> List[str]:
    metrics = set()
    for section in SECTION_ORDER:
        if section not in stats:
            continue
        metrics.update(k for k in stats[section].keys() if k != "count")
    return sorted(metrics)


def _dim_names(metric: str, n: int) -> List[str]:
    labels = DIM_LABELS.get(metric)
    if labels is not None and len(labels) == n:
        return labels
    return [f"d{i}" for i in range(n)]


def _flatten_rows(stats: Dict) -> List[Dict]:
    rows: List[Dict] = []
    for section in SECTION_ORDER:
        if section not in stats:
            continue
        section_stats = stats[section]
        count = int(section_stats.get("count", 0))
        for metric, metric_stats in section_stats.items():
            if metric == "count":
                continue
            means = metric_stats.get("mean", [])
            mins = metric_stats.get("min", [])
            maxs = metric_stats.get("max", [])
            dim_names = _dim_names(metric, len(means))
            for idx, dim_name in enumerate(dim_names):
                rows.append(
                    {
                        "section": section,
                        "count": count,
                        "metric": metric,
                        "dim": dim_name,
                        "mean": means[idx],
                        "min": mins[idx],
                        "max": maxs[idx],
                        "range": maxs[idx] - mins[idx],
                    }
                )
    return rows


def _write_csv(rows: Iterable[Dict], path: Path) -> None:
    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["section", "count", "metric", "dim", "mean", "min", "max", "range"],
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(input_path: Path, stats: Dict, rows: List[Dict], path: Path) -> None:
    lines: List[str] = []
    lines.append(f"Input: {input_path.name}")
    lines.append(f"sim_steps: {stats.get('sim_steps', 'n/a')}")
    lines.append("")
    lines.append("Section counts:")
    for section in SECTION_ORDER:
        if section in stats:
            lines.append(f"- {section}: {int(stats[section].get('count', 0))}")

    lines.append("")
    lines.append("Metric summaries:")
    metrics = _metric_names(stats)
    for metric in metrics:
        lines.append(f"- {metric}:")
        metric_rows = [r for r in rows if r["metric"] == metric]
        for section in SECTION_ORDER:
            section_rows = [r for r in metric_rows if r["section"] == section]
            if not section_rows:
                continue
            parts = []
            for row in section_rows:
                parts.append(
                    f"{row['dim']}: mean={row['mean']:.4f}, min={row['min']:.4f}, max={row['max']:.4f}"
                )
            lines.append(f"  {section}: " + "; ".join(parts))
        lines.append("")

    path.write_text("\n".join(lines))


def _svg_header(width: int, height: int) -> List[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]


def _svg_footer(lines: List[str], path: Path) -> None:
    lines.append("</svg>")
    path.write_text("\n".join(lines))


def _plot_counts(stats: Dict, path: Path) -> None:
    sections = [s for s in SECTION_ORDER if s in stats]
    counts = [int(stats[s].get("count", 0)) for s in sections]
    width, height = 900, 450
    margin_left, margin_right, margin_top, margin_bottom = 80, 30, 40, 80
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    max_count = max(counts) if counts else 1
    bar_w = plot_w / max(len(sections), 1) * 0.6
    gap = plot_w / max(len(sections), 1)

    lines = _svg_header(width, height)
    lines.append(f'<text x="{width/2}" y="24" text-anchor="middle" font-size="18" font-family="sans-serif">Teacher State Stats Counts</text>')
    lines.append(f'<line x1="{margin_left}" y1="{height-margin_bottom}" x2="{width-margin_right}" y2="{height-margin_bottom}" stroke="black"/>')
    lines.append(f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height-margin_bottom}" stroke="black"/>')

    for i in range(5):
        frac = i / 4.0
        y = margin_top + plot_h * (1 - frac)
        value = int(round(max_count * frac))
        lines.append(f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width-margin_right}" y2="{y:.1f}" stroke="#dddddd"/>')
        lines.append(f'<text x="{margin_left-8}" y="{y+4:.1f}" text-anchor="end" font-size="11" font-family="sans-serif">{value}</text>')

    for idx, (section, count) in enumerate(zip(sections, counts)):
        cx = margin_left + gap * (idx + 0.5)
        bar_h = 0 if max_count == 0 else plot_h * (count / max_count)
        x = cx - bar_w / 2
        y = height - margin_bottom - bar_h
        color = SECTION_COLORS.get(section, "#444444")
        lines.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{color}"/>')
        lines.append(f'<text x="{cx:.1f}" y="{y-6:.1f}" text-anchor="middle" font-size="11" font-family="sans-serif">{count}</text>')
        lines.append(f'<text x="{cx:.1f}" y="{height-margin_bottom+20}" text-anchor="middle" font-size="12" font-family="sans-serif">{section}</text>')

    _svg_footer(lines, path)


def _plot_metric(stats: Dict, metric: str, path: Path) -> None:
    sections = [s for s in SECTION_ORDER if s in stats and metric in stats[s]]
    if not sections:
        return

    n_dims = len(stats[sections[0]][metric]["mean"])
    dim_names = _dim_names(metric, n_dims)
    width = max(900, int(120 * n_dims + 220))
    row_h = 180
    panel_gap = 26
    margin_left, margin_right, margin_top, margin_bottom = 80, 30, 50, 60
    height = margin_top + margin_bottom + row_h * len(sections) + panel_gap * max(len(sections) - 1, 0)
    plot_w = width - margin_left - margin_right
    all_vals = []
    for section in sections:
        metric_stats = stats[section][metric]
        all_vals.extend(metric_stats["min"])
        all_vals.extend(metric_stats["max"])

    vmin = min(all_vals) if all_vals else 0.0
    vmax = max(all_vals) if all_vals else 1.0
    if abs(vmax - vmin) < 1e-9:
        vmax = vmin + 1.0

    def x_pos(idx: int) -> float:
        if n_dims == 1:
            return margin_left + plot_w / 2
        return margin_left + plot_w * idx / (n_dims - 1)

    def y_pos(val: float, top: float) -> float:
        frac = (val - vmin) / (vmax - vmin)
        return top + row_h * (1 - frac)

    lines = _svg_header(width, height)
    lines.append(f'<text x="{width/2}" y="24" text-anchor="middle" font-size="18" font-family="sans-serif">{metric}</text>')

    for idx, section in enumerate(sections):
        metric_stats = stats[section][metric]
        means = metric_stats["mean"]
        mins = metric_stats["min"]
        maxs = metric_stats["max"]
        color = SECTION_COLORS.get(section, "#444444")
        count = int(stats[section].get("count", 0))
        panel_top = margin_top + idx * (row_h + panel_gap)
        panel_bottom = panel_top + row_h

        lines.append(f'<rect x="{margin_left}" y="{panel_top}" width="{plot_w}" height="{row_h}" fill="none" stroke="#cccccc"/>')
        lines.append(
            f'<text x="{margin_left}" y="{panel_top-8}" font-size="13" font-family="sans-serif">{section} (count={count})</text>'
        )
        lines.append(f'<line x1="{margin_left}" y1="{panel_bottom}" x2="{width-margin_right}" y2="{panel_bottom}" stroke="black"/>')
        lines.append(f'<line x1="{margin_left}" y1="{panel_top}" x2="{margin_left}" y2="{panel_bottom}" stroke="black"/>')

        for tick_idx in range(5):
            frac = tick_idx / 4.0
            value = vmin + (vmax - vmin) * frac
            y = y_pos(value, panel_top)
            lines.append(f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width-margin_right}" y2="{y:.1f}" stroke="#dddddd"/>')
            lines.append(f'<text x="{margin_left-8}" y="{y+4:.1f}" text-anchor="end" font-size="11" font-family="sans-serif">{value:.3f}</text>')

        for dim_idx, label in enumerate(dim_names):
            x = x_pos(dim_idx)
            lines.append(f'<line x1="{x:.1f}" y1="{panel_top}" x2="{x:.1f}" y2="{panel_bottom}" stroke="#f0f0f0"/>')
            if idx == len(sections) - 1:
                lines.append(f'<text x="{x:.1f}" y="{height-margin_bottom+20}" text-anchor="middle" font-size="12" font-family="sans-serif">{label}</text>')

        for dim_idx in range(n_dims):
            x = x_pos(dim_idx)
            ymin = y_pos(mins[dim_idx], panel_top)
            ymax = y_pos(maxs[dim_idx], panel_top)
            lines.append(f'<line x1="{x:.1f}" y1="{ymin:.1f}" x2="{x:.1f}" y2="{ymax:.1f}" stroke="{color}" stroke-width="2" opacity="0.45"/>')

        pts = " ".join(f"{x_pos(i):.1f},{y_pos(means[i], panel_top):.1f}" for i in range(n_dims))
        lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{pts}"/>')
        for i in range(n_dims):
            lines.append(f'<circle cx="{x_pos(i):.1f}" cy="{y_pos(means[i], panel_top):.1f}" r="3.5" fill="{color}"/>')

    _svg_footer(lines, path)


def analyze(input_path: Path, output_dir: Path) -> None:
    stats = _load_stats(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _flatten_rows(stats)
    summary_txt = output_dir / "summary.txt"
    summary_csv = output_dir / "summary.csv"
    summary_json = output_dir / "summary.json"

    _write_summary(input_path, stats, rows, summary_txt)
    _write_csv(rows, summary_csv)
    summary_json.write_text(json.dumps(stats, indent=2))

    _plot_counts(stats, output_dir / "counts.svg")
    for metric in _metric_names(stats):
        _plot_metric(stats, metric, output_dir / f"{metric}.svg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze aggregated teacher state stats JSON and save plots.")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to debug_teacher_state_stats_*.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save summary files and plots. Defaults to <input_stem>_analysis next to the input file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else input_path.with_name(f"{input_path.stem}_analysis")
    analyze(input_path, output_dir)
    print(f"Saved analysis to {output_dir}")


if __name__ == "__main__":
    main()
