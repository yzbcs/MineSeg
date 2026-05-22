"""
Summarize MUSeg experiment JSON files into bilingual tables and SVG charts.

Outputs:
  - results/summary/summary_metrics.csv
  - results/summary/summary_ablation.csv
  - results/summary/per_class_comparison.csv
  - results/summary/summary_report.md
  - results/summary/*.svg
"""

import csv
import json
import math
import os
from typing import Dict, List, Optional, Tuple


RESULTS_DIR = "./results"
OUTPUT_DIR = "./results/summary"

EXPERIMENTS = [
    {
        "exp": "A1",
        "file": "A1_merged.json",
        "label_en": "A1 Baseline",
        "label_zh": "A1 基线",
        "group": "main",
        "setting_en": "LISA-7B zero-shot",
        "setting_zh": "LISA-7B 零样本",
        "sort_order": 1,
    },
    {
        "exp": "A3",
        "file": "A3_merged.json",
        "label_en": "A3 LoRA",
        "label_zh": "A3 微调",
        "group": "main",
        "setting_en": "LISA-7B + LoRA (r=8, ep=10, lr=3e-4)",
        "setting_zh": "LISA-7B + LoRA (r=8, ep=10, lr=3e-4)",
        "sort_order": 2,
    },
    {
        "exp": "B1_r4",
        "file": "B1_r4_merged.json",
        "label_en": "B1 rank=4",
        "label_zh": "B1 rank=4",
        "group": "rank",
        "setting_en": "LoRA rank r=4",
        "setting_zh": "LoRA 秩 r=4",
        "sort_order": 3,
    },
    {
        "exp": "B1_r16",
        "file": "B1_r16_merged.json",
        "label_en": "B1 rank=16",
        "label_zh": "B1 rank=16",
        "group": "rank",
        "setting_en": "LoRA rank r=16",
        "setting_zh": "LoRA 秩 r=16",
        "sort_order": 4,
    },
    {
        "exp": "B2_ep5",
        "file": "B2_ep5_merged.json",
        "label_en": "B2 epoch=5",
        "label_zh": "B2 epoch=5",
        "group": "epoch",
        "setting_en": "Training epochs = 5",
        "setting_zh": "训练轮数 = 5",
        "sort_order": 5,
    },
    {
        "exp": "B2_ep15",
        "file": "B2_ep15_merged.json",
        "label_en": "B2 epoch=15",
        "label_zh": "B2 epoch=15",
        "group": "epoch",
        "setting_en": "Training epochs = 15",
        "setting_zh": "训练轮数 = 15",
        "sort_order": 6,
    },
    {
        "exp": "B2_ep20",
        "file": "B2_ep20.json",
        "label_en": "B2 epoch=20",
        "label_zh": "B2 epoch=20",
        "group": "epoch",
        "setting_en": "Training epochs = 20",
        "setting_zh": "训练轮数 = 20",
        "sort_order": 7,
    },
    {
        "exp": "B3_lr1e4",
        "file": "B3_lr1e4.json",
        "label_en": "B3 lr=1e-4",
        "label_zh": "B3 lr=1e-4",
        "group": "lr",
        "setting_en": "Learning rate = 1e-4",
        "setting_zh": "学习率 = 1e-4",
        "sort_order": 8,
    },
    {
        "exp": "B3_lr5e4",
        "file": "B3_lr5e4.json",
        "label_en": "B3 lr=5e-4",
        "label_zh": "B3 lr=5e-4",
        "group": "lr",
        "setting_en": "Learning rate = 5e-4",
        "setting_zh": "学习率 = 5e-4",
        "sort_order": 9,
    },
]

CLASS_NAME_ZH = {
    "person": "人员",
    "cable": "线缆",
    "tube": "管道",
    "indicator": "指示设备",
    "electrical equipment": "电气设备",
    "electronic equipment": "电子设备",
    "mining equipment": "采矿设备",
    "rail area": "轨道区域",
    "support equipment": "支护设备",
    "door": "门",
    "tools and materials": "工具与材料",
    "rescue equipment": "救援设备",
    "container": "容器",
    "metal fixture": "金属固定装置",
    "anchoring equipment": "锚固设备",
}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_json(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def fmt(v: Optional[float], digits: int = 4) -> str:
    if v is None:
        return "-"
    return f"{v:.{digits}f}"


def pct(v: Optional[float], digits: int = 1) -> str:
    if v is None:
        return "-"
    return f"{v * 100:.{digits}f}%"


def sanitize_filename(name: str) -> str:
    safe = []
    for ch in name:
        if ch.isalnum() or ch in ("-", "_"):
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe).strip("_")


def write_csv(path: str, header: List[str], rows: List[List[object]]) -> None:
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def rank_rows(rows: List[dict], key: str) -> List[dict]:
    return sorted(rows, key=lambda row: row[key], reverse=True)


def svg_header(width: int, height: int) -> List[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>',
        'text { font-family: "Segoe UI", "Microsoft YaHei", sans-serif; fill: #1f2937; }',
        '.title { font-size: 20px; font-weight: 700; }',
        '.subtitle { font-size: 12px; fill: #6b7280; }',
        '.axis { font-size: 11px; fill: #374151; }',
        '.label { font-size: 12px; }',
        '.value { font-size: 11px; font-weight: 600; }',
        '</style>',
    ]


def svg_footer() -> List[str]:
    return ["</svg>"]


def save_svg(path: str, lines: List[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def draw_bar_chart(
    path: str,
    title: str,
    subtitle: str,
    labels: List[str],
    values: List[float],
    colors: List[str],
    y_max: Optional[float] = None,
) -> None:
    width = 1200
    height = 640
    left = 80
    right = 40
    top = 110
    bottom = 170
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_val = y_max if y_max is not None else max(values) * 1.15
    max_val = max(max_val, 0.01)
    n = len(values)
    step = plot_w / max(n, 1)
    bar_w = step * 0.62
    lines = svg_header(width, height)
    lines.append(f'<text x="{left}" y="40" class="title">{title}</text>')
    lines.append(f'<text x="{left}" y="64" class="subtitle">{subtitle}</text>')
    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#111827" stroke-width="1.4"/>')
    lines.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#111827" stroke-width="1.4"/>')

    for i in range(6):
        v = max_val * i / 5
        y = top + plot_h - plot_h * i / 5
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#e5e7eb" stroke-width="1"/>')
        lines.append(f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" class="axis">{v:.2f}</text>')

    for idx, (label, value, color) in enumerate(zip(labels, values, colors)):
        x = left + step * idx + (step - bar_w) / 2
        h = 0 if max_val == 0 else (value / max_val) * plot_h
        y = top + plot_h - h
        lines.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" rx="6" fill="{color}"/>')
        lines.append(f'<text x="{x + bar_w / 2:.1f}" y="{y - 8:.1f}" text-anchor="middle" class="value">{value:.4f}</text>')
        lines.append(f'<text x="{x + bar_w / 2:.1f}" y="{top + plot_h + 24:.1f}" text-anchor="middle" class="label">{label}</text>')

    lines.extend(svg_footer())
    save_svg(path, lines)


def draw_horizontal_delta_chart(
    path: str,
    title: str,
    subtitle: str,
    labels: List[str],
    values: List[float],
) -> None:
    width = 1300
    row_h = 32
    top = 110
    bottom = 40
    left = 320
    right = 80
    height = top + bottom + len(values) * row_h
    plot_w = width - left - right
    max_abs = max(max(abs(v) for v in values), 0.01)
    lines = svg_header(width, height)
    lines.append(f'<text x="40" y="40" class="title">{title}</text>')
    lines.append(f'<text x="40" y="64" class="subtitle">{subtitle}</text>')
    zero_x = left + plot_w / 2
    lines.append(f'<line x1="{zero_x:.1f}" y1="{top - 10}" x2="{zero_x:.1f}" y2="{height - bottom + 8}" stroke="#111827" stroke-width="1.4"/>')

    for i in range(5):
        ratio = -1 + i * 0.5
        x = zero_x + ratio * (plot_w / 2)
        val = ratio * max_abs
        lines.append(f'<line x1="{x:.1f}" y1="{top - 10}" x2="{x:.1f}" y2="{height - bottom + 8}" stroke="#e5e7eb" stroke-width="1"/>')
        lines.append(f'<text x="{x:.1f}" y="{top - 18}" text-anchor="middle" class="axis">{val:+.2f}</text>')

    for idx, (label, value) in enumerate(zip(labels, values)):
        y = top + idx * row_h
        bar_len = (abs(value) / max_abs) * (plot_w / 2)
        if value >= 0:
            x = zero_x
            color = "#0f766e"
            anchor = "start"
            tx = x + bar_len + 8
        else:
            x = zero_x - bar_len
            color = "#b91c1c"
            anchor = "end"
            tx = x - 8
        lines.append(f'<text x="{left - 12}" y="{y + 18}" text-anchor="end" class="label">{label}</text>')
        lines.append(f'<rect x="{x:.1f}" y="{y + 6:.1f}" width="{bar_len:.1f}" height="18" rx="4" fill="{color}"/>')
        lines.append(f'<text x="{tx:.1f}" y="{y + 20:.1f}" text-anchor="{anchor}" class="value">{value:+.4f}</text>')

    lines.extend(svg_footer())
    save_svg(path, lines)


def draw_line_chart(
    path: str,
    title: str,
    subtitle: str,
    series: List[Tuple[str, List[Tuple[float, float]], str]],
    x_label: str,
    y_label: str,
) -> None:
    width = 1100
    height = 620
    left = 90
    right = 50
    top = 110
    bottom = 100
    plot_w = width - left - right
    plot_h = height - top - bottom
    xs = [x for _, points, _ in series for x, _ in points]
    ys = [y for _, points, _ in series for _, y in points]
    x_min, x_max = min(xs), max(xs)
    y_min = min(ys) - 0.01
    y_max = max(ys) + 0.01
    if math.isclose(x_min, x_max):
        x_max = x_min + 1
    if math.isclose(y_min, y_max):
        y_max = y_min + 1

    def px_x(xv: float) -> float:
        return left + (xv - x_min) / (x_max - x_min) * plot_w

    def px_y(yv: float) -> float:
        return top + plot_h - (yv - y_min) / (y_max - y_min) * plot_h

    lines = svg_header(width, height)
    lines.append(f'<text x="{left}" y="40" class="title">{title}</text>')
    lines.append(f'<text x="{left}" y="64" class="subtitle">{subtitle}</text>')
    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#111827" stroke-width="1.4"/>')
    lines.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#111827" stroke-width="1.4"/>')
    for i in range(6):
        ratio = i / 5
        yv = y_min + (y_max - y_min) * ratio
        py = px_y(yv)
        lines.append(f'<line x1="{left}" y1="{py:.1f}" x2="{left + plot_w}" y2="{py:.1f}" stroke="#e5e7eb" stroke-width="1"/>')
        lines.append(f'<text x="{left - 10}" y="{py + 4:.1f}" text-anchor="end" class="axis">{yv:.3f}</text>')
    for xv in sorted(set(xs)):
        px = px_x(xv)
        lines.append(f'<line x1="{px:.1f}" y1="{top}" x2="{px:.1f}" y2="{top + plot_h}" stroke="#f3f4f6" stroke-width="1"/>')
        lines.append(f'<text x="{px:.1f}" y="{top + plot_h + 24}" text-anchor="middle" class="axis">{xv:g}</text>')

    legend_x = left
    for name, points, color in series:
        path_cmd = []
        for idx, (xv, yv) in enumerate(points):
            cmd = "M" if idx == 0 else "L"
            path_cmd.append(f"{cmd} {px_x(xv):.1f} {px_y(yv):.1f}")
        lines.append(f'<path d="{" ".join(path_cmd)}" fill="none" stroke="{color}" stroke-width="3"/>')
        for xv, yv in points:
            cx, cy = px_x(xv), px_y(yv)
            lines.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="4.5" fill="{color}"/>')
            lines.append(f'<text x="{cx:.1f}" y="{cy - 10:.1f}" text-anchor="middle" class="value">{yv:.4f}</text>')
        lines.append(f'<rect x="{legend_x}" y="{height - 40}" width="20" height="8" rx="4" fill="{color}"/>')
        lines.append(f'<text x="{legend_x + 28}" y="{height - 32}" class="label">{name}</text>')
        legend_x += 140

    lines.append(f'<text x="{left + plot_w / 2:.1f}" y="{height - 10}" text-anchor="middle" class="label">{x_label}</text>')
    lines.append(f'<text x="24" y="{top + plot_h / 2:.1f}" transform="rotate(-90 24 {top + plot_h / 2:.1f})" text-anchor="middle" class="label">{y_label}</text>')
    lines.extend(svg_footer())
    save_svg(path, lines)


def build_summary() -> None:
    ensure_dir(OUTPUT_DIR)
    loaded = []
    for spec in EXPERIMENTS:
        data = load_json(os.path.join(RESULTS_DIR, spec["file"]))
        if data is None:
            continue
        row = dict(spec)
        row.update(data)
        loaded.append(row)

    loaded.sort(key=lambda row: row["sort_order"])
    if not loaded:
        raise RuntimeError("No experiment JSON files were loaded.")

    base = next(row for row in loaded if row["exp"] == "A1")
    best = max(loaded, key=lambda row: row["giou"])

    metrics_rows = []
    for row in rank_rows(loaded, "giou"):
        delta_g = row["giou"] - base["giou"]
        rel_g = (row["giou"] / base["giou"] - 1.0) if base["giou"] else None
        delta_m = row["miou"] - base["miou"]
        metrics_rows.append([
            row["exp"],
            row["label_zh"],
            row["label_en"],
            row["setting_zh"],
            row["setting_en"],
            fmt(row["giou"]),
            fmt(row["ciou"]),
            fmt(row["miou"]),
            fmt(delta_g),
            pct(rel_g),
            fmt(delta_m),
            row.get("total", ""),
        ])

    write_csv(
        os.path.join(OUTPUT_DIR, "summary_metrics.csv"),
        [
            "exp",
            "label_zh",
            "label_en",
            "setting_zh",
            "setting_en",
            "giou",
            "ciou",
            "miou",
            "delta_giou_vs_A1",
            "relative_giou_vs_A1",
            "delta_miou_vs_A1",
            "total_samples",
        ],
        metrics_rows,
    )

    ablation_rows = []
    a3_row = next(row for row in loaded if row["exp"] == "A3")
    ablation_rows.extend([
        ["rank", "LoRA 秩 / LoRA rank", "4", fmt(next(row for row in loaded if row["exp"] == "B1_r4")["giou"]), fmt(next(row for row in loaded if row["exp"] == "B1_r4")["miou"])],
        ["rank", "LoRA 秩 / LoRA rank", "8", fmt(a3_row["giou"]), fmt(a3_row["miou"])],
        ["rank", "LoRA 秩 / LoRA rank", "16", fmt(next(row for row in loaded if row["exp"] == "B1_r16")["giou"]), fmt(next(row for row in loaded if row["exp"] == "B1_r16")["miou"])],
        ["epoch", "训练轮数 / Epoch", "5", fmt(next(row for row in loaded if row["exp"] == "B2_ep5")["giou"]), fmt(next(row for row in loaded if row["exp"] == "B2_ep5")["miou"])],
        ["epoch", "训练轮数 / Epoch", "10", fmt(a3_row["giou"]), fmt(a3_row["miou"])],
        ["epoch", "训练轮数 / Epoch", "15", fmt(next(row for row in loaded if row["exp"] == "B2_ep15")["giou"]), fmt(next(row for row in loaded if row["exp"] == "B2_ep15")["miou"])],
        ["epoch", "训练轮数 / Epoch", "20", fmt(next(row for row in loaded if row["exp"] == "B2_ep20")["giou"]), fmt(next(row for row in loaded if row["exp"] == "B2_ep20")["miou"])],
        ["lr", "学习率 / Learning Rate", "1e-4", fmt(next(row for row in loaded if row["exp"] == "B3_lr1e4")["giou"]), fmt(next(row for row in loaded if row["exp"] == "B3_lr1e4")["miou"])],
        ["lr", "学习率 / Learning Rate", "3e-4", fmt(a3_row["giou"]), fmt(a3_row["miou"])],
        ["lr", "学习率 / Learning Rate", "5e-4", fmt(next(row for row in loaded if row["exp"] == "B3_lr5e4")["giou"]), fmt(next(row for row in loaded if row["exp"] == "B3_lr5e4")["miou"])],
    ])
    write_csv(
        os.path.join(OUTPUT_DIR, "summary_ablation.csv"),
        ["factor", "factor_label", "value", "giou", "miou"],
        ablation_rows,
    )

    classes = sorted(set(base.get("per_class_iou", {}).keys()) | set(a3_row.get("per_class_iou", {}).keys()))
    per_class_rows = []
    delta_items = []
    total_count = sum(base.get("per_class_count", {}).values())
    for cls in classes:
        a1 = base["per_class_iou"].get(cls)
        a3 = a3_row["per_class_iou"].get(cls)
        count = a3_row.get("per_class_count", {}).get(cls, base.get("per_class_count", {}).get(cls, 0))
        delta = None if a1 is None or a3 is None else a3 - a1
        weighted_signal = None if delta is None else delta * count
        share = None if total_count == 0 else count / total_count
        per_class_rows.append([
            cls,
            CLASS_NAME_ZH.get(cls, cls),
            count,
            pct(share),
            fmt(a1),
            fmt(a3),
            fmt(delta),
            fmt(weighted_signal),
        ])
        if delta is not None:
            delta_items.append((cls, delta))

    write_csv(
        os.path.join(OUTPUT_DIR, "per_class_comparison.csv"),
        [
            "class_en",
            "class_zh",
            "count",
            "share_of_test_pairs",
            "A1_iou",
            "A3_iou",
            "delta_A3_minus_A1",
            "count_weighted_delta",
        ],
        per_class_rows,
    )

    draw_bar_chart(
        os.path.join(OUTPUT_DIR, "overall_giou_ranking.svg"),
        "Overall gIoU Ranking / 总体 gIoU 排名",
        "Higher is better. A3 is the best current setting.",
        [row["exp"] for row in rank_rows(loaded, "giou")],
        [row["giou"] for row in rank_rows(loaded, "giou")],
        ["#0f766e" if row["exp"] == best["exp"] else "#2563eb" if row["exp"] == "A1" else "#64748b" for row in rank_rows(loaded, "giou")],
        y_max=max(row["giou"] for row in loaded) * 1.2,
    )

    draw_bar_chart(
        os.path.join(OUTPUT_DIR, "overall_miou_ranking.svg"),
        "Overall mIoU Ranking / 总体 mIoU 排名",
        "Higher is better. mIoU reflects class-balanced performance.",
        [row["exp"] for row in rank_rows(loaded, "miou")],
        [row["miou"] for row in rank_rows(loaded, "miou")],
        ["#0f766e" if row["exp"] == best["exp"] else "#2563eb" if row["exp"] == "A1" else "#64748b" for row in rank_rows(loaded, "miou")],
        y_max=max(row["miou"] for row in loaded) * 1.2,
    )

    delta_items.sort(key=lambda item: item[1], reverse=True)
    delta_labels = [f"{item[0]} / {CLASS_NAME_ZH.get(item[0], item[0])}" for item in delta_items]
    delta_values = [item[1] for item in delta_items]
    draw_horizontal_delta_chart(
        os.path.join(OUTPUT_DIR, "per_class_delta_a3_vs_a1.svg"),
        "Per-class IoU Delta: A3 vs A1 / 各类别 IoU 变化",
        "Positive means A3 LoRA is better than the zero-shot baseline.",
        delta_labels,
        delta_values,
    )

    draw_line_chart(
        os.path.join(OUTPUT_DIR, "ablation_rank.svg"),
        "Rank Ablation / Rank 消融",
        "Compare gIoU and mIoU across LoRA rank.",
        [
            ("gIoU", [(4, next(row for row in loaded if row["exp"] == "B1_r4")["giou"]), (8, a3_row["giou"]), (16, next(row for row in loaded if row["exp"] == "B1_r16")["giou"])], "#0f766e"),
            ("mIoU", [(4, next(row for row in loaded if row["exp"] == "B1_r4")["miou"]), (8, a3_row["miou"]), (16, next(row for row in loaded if row["exp"] == "B1_r16")["miou"])], "#2563eb"),
        ],
        "LoRA rank",
        "Metric value",
    )

    draw_line_chart(
        os.path.join(OUTPUT_DIR, "ablation_epoch.svg"),
        "Epoch Ablation / Epoch 消融",
        "Compare gIoU and mIoU across training epochs.",
        [
            ("gIoU", [(5, next(row for row in loaded if row["exp"] == "B2_ep5")["giou"]), (10, a3_row["giou"]), (15, next(row for row in loaded if row["exp"] == "B2_ep15")["giou"]), (20, next(row for row in loaded if row["exp"] == "B2_ep20")["giou"])], "#0f766e"),
            ("mIoU", [(5, next(row for row in loaded if row["exp"] == "B2_ep5")["miou"]), (10, a3_row["miou"]), (15, next(row for row in loaded if row["exp"] == "B2_ep15")["miou"]), (20, next(row for row in loaded if row["exp"] == "B2_ep20")["miou"])], "#2563eb"),
        ],
        "Epoch",
        "Metric value",
    )

    draw_line_chart(
        os.path.join(OUTPUT_DIR, "ablation_lr.svg"),
        "Learning Rate Ablation / 学习率消融",
        "Compare gIoU and mIoU across learning rates.",
        [
            ("gIoU", [(1, next(row for row in loaded if row["exp"] == "B3_lr1e4")["giou"]), (3, a3_row["giou"]), (5, next(row for row in loaded if row["exp"] == "B3_lr5e4")["giou"])], "#0f766e"),
            ("mIoU", [(1, next(row for row in loaded if row["exp"] == "B3_lr1e4")["miou"]), (3, a3_row["miou"]), (5, next(row for row in loaded if row["exp"] == "B3_lr5e4")["miou"])], "#2563eb"),
        ],
        "Learning rate (1=1e-4, 3=3e-4, 5=5e-4)",
        "Metric value",
    )

    summary_lines = []
    summary_lines.append("# MUSeg LoRA Experiment Summary / MUSeg LoRA 实验汇总")
    summary_lines.append("")
    summary_lines.append("## 1. Overall Ranking / 总体排名")
    summary_lines.append("")
    summary_lines.append("| Rank | Exp | 中文名称 | English Label | gIoU | cIoU | mIoU | ΔgIoU vs A1 | Relative ΔgIoU |")
    summary_lines.append("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for idx, row in enumerate(rank_rows(loaded, "giou"), start=1):
        delta_g = row["giou"] - base["giou"]
        rel_g = (row["giou"] / base["giou"] - 1.0) if base["giou"] else None
        summary_lines.append(
            f"| {idx} | {row['exp']} | {row['label_zh']} | {row['label_en']} | "
            f"{fmt(row['giou'])} | {fmt(row['ciou'])} | {fmt(row['miou'])} | {fmt(delta_g)} | {pct(rel_g)} |"
        )
    summary_lines.append("")
    summary_lines.append("## 2. Key Findings / 核心结论")
    summary_lines.append("")
    summary_lines.append(f"- Best setting / 最优配置: `{best['exp']}` with `gIoU={fmt(best['giou'])}`, `mIoU={fmt(best['miou'])}`.")
    summary_lines.append(f"- Baseline to best / 基线到最优: `gIoU {fmt(base['giou'])} -> {fmt(best['giou'])}`, relative gain `+{(best['giou'] / base['giou'] - 1.0) * 100:.1f}%`.")
    summary_lines.append("- `gIoU` is the most stable headline metric here because it reflects average instance-level performance.")
    summary_lines.append("- `cIoU` in B3 uses exact shard merge, while A1-B2 use approximate merge, so cross-group `cIoU` comparison should be treated cautiously.")
    summary_lines.append("")
    summary_lines.append("## 3. Per-class A3 vs A1 / A3 与 A1 类别对比")
    summary_lines.append("")
    summary_lines.append("| Class (EN) | 类别(中文) | Count | Share | A1 IoU | A3 IoU | Delta |")
    summary_lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for cls, _, count, share, a1, a3, delta, _ in per_class_rows:
        summary_lines.append(f"| {cls} | {CLASS_NAME_ZH.get(cls, cls)} | {count} | {share} | {a1} | {a3} | {delta} |")
    summary_lines.append("")
    top_gain = max(delta_items, key=lambda item: item[1])
    top_drop = min(delta_items, key=lambda item: item[1])
    summary_lines.append(f"- Largest improvement / 最大提升: `{top_gain[0]}` (`{CLASS_NAME_ZH.get(top_gain[0], top_gain[0])}`), delta `{top_gain[1]:+.4f}`.")
    summary_lines.append(f"- Largest drop / 最大下降: `{top_drop[0]}` (`{CLASS_NAME_ZH.get(top_drop[0], top_drop[0])}`), delta `{top_drop[1]:+.4f}`.")
    summary_lines.append("")
    summary_lines.append("## 4. Output Files / 输出文件")
    summary_lines.append("")
    summary_lines.append("- `summary_metrics.csv`: overall experiment ranking")
    summary_lines.append("- `summary_ablation.csv`: ablation-ready metric table")
    summary_lines.append("- `per_class_comparison.csv`: class-level A1 vs A3 comparison")
    summary_lines.append("- `overall_giou_ranking.svg`, `overall_miou_ranking.svg`: overall ranking charts")
    summary_lines.append("- `per_class_delta_a3_vs_a1.svg`: class delta chart")
    summary_lines.append("- `ablation_rank.svg`, `ablation_epoch.svg`, `ablation_lr.svg`: ablation charts")
    summary_lines.append("")

    with open(os.path.join(OUTPUT_DIR, "summary_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))


if __name__ == "__main__":
    build_summary()
