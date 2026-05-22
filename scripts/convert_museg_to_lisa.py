"""
MUSeg 数据集索引构建脚本

扫描 MUSeg 原始目录，为每张图像统计包含的类别，
生成轻量 JSON 索引文件。训练时 Dataset 根据索引直接从原始文件读取。

不拷贝/链接任何文件，只生成索引。

用法:
  python scripts/convert_museg_to_lisa.py ^
      --museg_root ../MUSeg ^
      --output_dir ./dataset/museg ^
      --train_mines 01 03 06 ^
      --val_mines 02 ^
      --test_mines 04 05
"""

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np


CLASS_NAMES = {
    0: "background",
    1: "person",
    2: "cable",
    3: "tube",
    4: "indicator",
    5: "electrical equipment",
    6: "electronic equipment",
    7: "mining equipment",
    8: "rail area",
    9: "support equipment",
    10: "door",
    11: "tools and materials",
    12: "rescue equipment",
    13: "container",
    14: "metal fixture",
    15: "anchoring equipment",
}

NUM_CLASSES = 15  # 不含 background


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--museg_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--train_mines", nargs="+", default=["01", "03", "06"])
    parser.add_argument("--val_mines", nargs="+", default=["02"])
    parser.add_argument("--test_mines", nargs="+", default=["04", "05"])
    return parser.parse_args()


def scan_mine(museg_root, mine_num):
    """扫描一个矿区，返回样本列表"""
    mine_dir = os.path.join(museg_root, f"{mine_num}-Mine")
    image_dir = os.path.join(mine_dir, "Image")
    label_dir = os.path.join(mine_dir, "Label")

    if not os.path.isdir(image_dir) or not os.path.isdir(label_dir):
        print(f"[WARN] {mine_dir} 缺少 Image 或 Label 目录, 跳过")
        return []

    samples = []
    for img_file in sorted(os.listdir(image_dir)):
        if not img_file.lower().endswith((".jpg", ".png", ".bmp")):
            continue
        stem = os.path.splitext(img_file)[0]
        image_path = os.path.join(image_dir, img_file)
        label_path = os.path.join(label_dir, f"{stem}_label.png")

        if not os.path.exists(label_path):
            continue

        label_img = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
        if label_img is None:
            continue

        unique_ids = np.unique(label_img).tolist()
        class_ids = [cid for cid in unique_ids if 1 <= cid <= NUM_CLASSES]

        if not class_ids:
            continue

        samples.append({
            "image_path": os.path.abspath(image_path),
            "label_path": os.path.abspath(label_path),
            "mine": mine_num,
            "stem": stem,
            "class_ids": class_ids,
            "class_names": [CLASS_NAMES[cid] for cid in class_ids],
            "height": int(label_img.shape[0]),
            "width": int(label_img.shape[1]),
        })
    return samples


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    mine_to_split = {}
    for m in args.train_mines:
        mine_to_split[m] = "train"
    for m in args.val_mines:
        mine_to_split[m] = "val"
    for m in args.test_mines:
        mine_to_split[m] = "test"

    split_data = {"train": [], "val": [], "test": []}

    for mine_num in sorted(mine_to_split.keys()):
        split = mine_to_split[mine_num]
        samples = scan_mine(args.museg_root, mine_num)
        split_data[split].extend(samples)
        print(f"{mine_num}-Mine → {split}: {len(samples)} 张图像")

    # 统计每个 split 的单类别样本数（展开后）
    for split_name, samples in split_data.items():
        total_pairs = sum(len(s["class_ids"]) for s in samples)
        index_path = os.path.join(args.output_dir, f"{split_name}.json")
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)
        print(f"\n{split_name}: {len(samples)} 张图像, {total_pairs} 个单类别样本")
        print(f"  索引已保存: {index_path}")

    # 保存类别映射
    meta = {
        "class_names": {str(k): v for k, v in CLASS_NAMES.items()},
        "num_classes": NUM_CLASSES,
        "mine_split": mine_to_split,
    }
    meta_path = os.path.join(args.output_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"\n类别映射已保存: {meta_path}")


if __name__ == "__main__":
    main()
