"""
Merge evaluation shard results into a single JSON.

Handles two formats:
  - New format: has per_class_intersection & per_class_union (exact merge)
  - Old format: only has per_class_iou & per_class_count (approximate merge)

Usage:
  python scripts/merge_shard_results.py shard0.json shard1.json output.json exp_name
"""

import json
import sys


def merge_shards(shard_paths, output_path, exp_name="merged"):
    shards = []
    for p in shard_paths:
        with open(p, "r", encoding="utf-8") as f:
            shards.append(json.load(f))

    total_samples = sum(s["total_samples"] for s in shards)
    giou = sum(s["giou"] * s["total_samples"] for s in shards) / total_samples

    has_raw = all("per_class_intersection" in s and "per_class_union" in s for s in shards)

    all_classes = set()
    for s in shards:
        all_classes.update(s.get("per_class_count", {}).keys())

    per_class_iou = {}
    per_class_count = {}
    global_inter = 0.0
    global_union = 0.0

    for cls in sorted(all_classes):
        cls_inter = 0.0
        cls_union = 0.0
        cls_count = 0

        for s in shards:
            count = s.get("per_class_count", {}).get(cls, 0)
            if count == 0:
                continue

            if has_raw:
                cls_inter += s["per_class_intersection"].get(cls, 0)
                cls_union += s["per_class_union"].get(cls, 0)
            else:
                iou_val = s.get("per_class_iou", {}).get(cls)
                if iou_val is not None:
                    cls_inter += iou_val * count
                    cls_union += count
            cls_count += count

        if cls_union > 0:
            per_class_iou[cls] = cls_inter / cls_union
        else:
            per_class_iou[cls] = None
        per_class_count[cls] = cls_count
        global_inter += cls_inter
        global_union += cls_union

    ciou = global_inter / (global_union + 1e-10)

    valid = [v for v in per_class_iou.values() if v is not None]
    miou = sum(valid) / len(valid) if valid else 0.0

    merge_type = "exact" if has_raw else "approximate"
    print(f"  [{exp_name}] gIoU={giou:.4f} cIoU={ciou:.4f} mIoU={miou:.4f} "
          f"({total_samples} samples, {merge_type})")

    merged = {
        "exp": exp_name,
        "giou": giou,
        "ciou": ciou,
        "miou": miou,
        "total": total_samples,
        "per_class_iou": per_class_iou,
        "per_class_count": per_class_count,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    print(f"  Saved to {output_path}")
    return merged


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python merge_shard_results.py shard0.json shard1.json output.json [exp_name]")
        sys.exit(1)

    exp = sys.argv[-1] if len(sys.argv) >= 5 else "merged"
    out = sys.argv[-2] if len(sys.argv) >= 5 else sys.argv[-1]
    shard_files = sys.argv[1:-2] if len(sys.argv) >= 5 else sys.argv[1:-1]

    merge_shards(shard_files, out, exp)
