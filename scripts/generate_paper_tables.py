"""
读取 results/ 目录下的 JSON, 生成论文所需 LaTeX 表格

用法:
  python scripts/generate_paper_tables.py --results_dir ./results
"""

import argparse
import json
import os

CLASS_ORDER = [
    "person", "cable", "tube", "indicator",
    "electrical equipment", "electronic equipment",
    "mining equipment", "rail area", "support equipment",
    "door", "tools and materials", "rescue equipment",
    "container", "metal fixture", "anchoring equipment",
]
CLASS_ABBREV = {
    "person": "Pers.", "cable": "Cable", "tube": "Tube",
    "indicator": "Indic.", "electrical equipment": "Elec.",
    "electronic equipment": "Eltn.", "mining equipment": "Mine.",
    "rail area": "Rail", "support equipment": "Supp.",
    "door": "Door", "tools and materials": "T\\&M",
    "rescue equipment": "Resc.", "container": "Cont.",
    "metal fixture": "M.Fix", "anchoring equipment": "Anch.",
}


def load(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def f(v, d=4):
    return f"{v:.{d}f}" if v is not None else "-"


def table_main(rd):
    rows = [
        ("A1_baseline_7b.json",  "LISA-7B (Original)"),
        ("A3_ours_7b.json",      "LISA-7B + LoRA (Ours)"),
        ("A2_baseline_13b.json", "LISA-13B (Original)"),
        ("A4_ours_13b.json",     "LISA-13B + LoRA (Ours)"),
    ]
    print("% ========== Table 1: Main Comparison ==========")
    print("\\begin{table}[htbp]\\centering")
    print("\\caption{Baseline vs fine-tuned LISA on MUSeg test set.}")
    print("\\label{tab:main}")
    print("\\begin{tabular}{lccc}\\toprule")
    print("Method & gIoU & cIoU & mIoU \\\\\\midrule")
    for fn, label in rows:
        r = load(os.path.join(rd, fn))
        if r:
            print(f"{label} & {f(r['giou'])} & {f(r['ciou'])} & {f(r['miou'])} \\\\")
        if "Ours" in label and "7B" in label:
            print("\\midrule")
    print("\\bottomrule\\end{tabular}\\end{table}")


def table_perclass(rd):
    rows = [
        ("A1_baseline_7b.json",  "LISA-7B"),
        ("A3_ours_7b.json",      "Ours-7B"),
        ("A2_baseline_13b.json", "LISA-13B"),
        ("A4_ours_13b.json",     "Ours-13B"),
    ]
    cols = " & ".join(CLASS_ABBREV[c] for c in CLASS_ORDER)
    ncols = len(CLASS_ORDER)
    print(f"\n% ========== Table 2: Per-class IoU ==========")
    print("\\begin{table*}[htbp]\\centering")
    print("\\caption{Per-class IoU (\\%) on MUSeg test set.}")
    print("\\label{tab:perclass}")
    print(f"\\resizebox{{\\textwidth}}{{!}}{{\\begin{{tabular}}{{l{'c'*ncols}c}}\\toprule")
    print(f"Method & {cols} & mIoU \\\\\\midrule")
    for fn, label in rows:
        r = load(os.path.join(rd, fn))
        if r and "per_class_iou" in r:
            vals = " & ".join(
                f(r["per_class_iou"].get(c), 2) for c in CLASS_ORDER
            )
            print(f"{label} & {vals} & {f(r['miou'],2)} \\\\")
    print("\\bottomrule\\end{tabular}}\\end{table*}")


def table_ablation(rd):
    print(f"\n% ========== Table 3: Ablation ==========")
    print("\\begin{table}[htbp]\\centering")
    print("\\caption{Ablation study results on MUSeg test set.}")
    print("\\label{tab:ablation}")
    print("\\begin{tabular}{llccc}\\toprule")
    print("Factor & Setting & gIoU & cIoU & mIoU \\\\\\midrule")

    for r_val in [4, 8, 16]:
        r = load(os.path.join(rd, f"B1_lora_r{r_val}.json"))
        if r:
            print(f"LoRA rank & $r={r_val}$ & {f(r['giou'])} & {f(r['ciou'])} & {f(r['miou'])} \\\\")
    print("\\midrule")

    for ep in [5, 10, 15, 20]:
        r = load(os.path.join(rd, f"B2_epoch_{ep}.json"))
        if r:
            print(f"Epochs & {ep} & {f(r['giou'])} & {f(r['ciou'])} & {f(r['miou'])} \\\\")
    print("\\midrule")

    for lr in ["1e-4", "3e-4", "5e-4"]:
        fn = f"B3_lr_{lr.replace('-0','-')}.json"
        r = load(os.path.join(rd, fn))
        if not r:
            r = load(os.path.join(rd, f"B3_lr_{lr}.json"))
        if r:
            print(f"Learning rate & {lr} & {f(r['giou'])} & {f(r['ciou'])} & {f(r['miou'])} \\\\")

    print("\\bottomrule\\end{tabular}\\end{table}")


def table_crossmine(rd):
    sp = os.path.join(rd, "cross_mine", "summary.json")
    if not os.path.exists(sp):
        print("\n% Cross-mine results not found")
        return
    with open(sp, "r") as fh:
        s = json.load(fh)

    print(f"\n% ========== Table 4: Cross-mine ==========")
    print("\\begin{table}[htbp]\\centering")
    print("\\caption{Leave-one-mine-out generalization results.}")
    print("\\label{tab:crossmine}")
    print("\\begin{tabular}{lccc}\\toprule")
    print("Test Mine & gIoU & cIoU & mIoU \\\\\\midrule")
    for k in sorted(s.get("experiments", {})):
        r = s["experiments"][k]
        print(f"Mine-{k} & {f(r['giou'])} & {f(r['ciou'])} & {f(r['miou'])} \\\\")
    print("\\midrule")
    print(f"Average & {f(s['avg_giou'])} & {f(s['avg_ciou'])} & {f(s['avg_miou'])} \\\\")
    print("\\bottomrule\\end{tabular}\\end{table}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="./results")
    args = parser.parse_args()

    table_main(args.results_dir)
    table_perclass(args.results_dir)
    table_ablation(args.results_dir)
    table_crossmine(args.results_dir)


if __name__ == "__main__":
    main()
