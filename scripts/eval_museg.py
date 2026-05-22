"""
MUSeg 评估脚本

评估流程 (与 chat.py 一致):
  1. 构建指令 prompt
  2. 调用 model.evaluate() 让模型自由生成文本 (含 [SEG])
  3. 从生成的 [SEG] embedding 解码 mask
  4. 与 GT mask 计算指标

指标:
  gIoU  - 每样本 IoU 的均值
  cIoU  - 全局 intersection / union
  mIoU  - 各类 IoU 的均值
  per-class IoU

用法:
  python scripts/eval_museg.py ^
      --version xinlai/LISA-7B-v1 ^
      --precision bf16 ^
      --dataset_dir ./dataset ^
      --val_dataset museg|test ^
      --vis_save_path ./vis_output
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from transformers import AutoTokenizer, CLIPImageProcessor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.LISA import LISAForCausalLM
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)


CLASS_NAMES = {
    1: "person", 2: "cable", 3: "tube", 4: "indicator",
    5: "electrical equipment", 6: "electronic equipment",
    7: "mining equipment", 8: "rail area", 9: "support equipment",
    10: "door", 11: "tools and materials", 12: "rescue equipment",
    13: "container", 14: "metal fixture", 15: "anchoring equipment",
}


def parse_args():
    parser = argparse.ArgumentParser(description="MUSeg Evaluation")
    parser.add_argument("--version", required=True, type=str)
    parser.add_argument("--dataset_dir", default="./dataset", type=str)
    parser.add_argument("--val_dataset", default="museg|test", type=str)
    parser.add_argument("--precision", default="bf16", type=str,
                        choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--image_size", default=1024, type=int)
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--vision_tower", default="openai/clip-vit-large-patch14", type=str)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--conv_type", default="llava_v1", type=str,
                        choices=["llava_v1", "llava_llama_2"])
    parser.add_argument("--vis_save_path", default="./vis_output/museg_eval", type=str)
    parser.add_argument("--save_vis", action="store_true", default=False)
    parser.add_argument("--output_json", default=None, type=str)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--max_samples", default=0, type=int,
                        help="Max samples to evaluate (0=all, for quick testing)")
    parser.add_argument("--num_shards", default=1, type=int,
                        help="Total number of shards for parallel evaluation")
    parser.add_argument("--shard_id", default=0, type=int,
                        help="This shard's index (0-based)")
    return parser.parse_args()


def preprocess_image(image_np, transform, img_size=1024):
    """SAM 预处理: resize + normalize + pad"""
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)

    image = transform.apply_image(image_np)
    resize = image.shape[:2]
    image = torch.from_numpy(image).permute(2, 0, 1).contiguous().float()
    image = (image - pixel_mean) / pixel_std
    h, w = image.shape[-2:]
    image = F.pad(image, (0, img_size - w, 0, img_size - h))
    return image, resize


def load_eval_samples(dataset_dir, val_dataset):
    """从索引加载评估样本, 展开为 (image, class_id) 对"""
    ds_name, split = val_dataset.split("|")
    index_path = os.path.join(dataset_dir, ds_name, f"{split}.json")
    with open(index_path, "r", encoding="utf-8") as f:
        images = json.load(f)

    samples = []
    for img_info in images:
        for cid in img_info["class_ids"]:
            samples.append({
                "image_path": img_info["image_path"],
                "label_path": img_info["label_path"],
                "class_id": cid,
                "class_name": CLASS_NAMES[cid],
            })
    return samples


def main():
    args = parse_args()

    # ====== 加载模型 (与 chat.py 逻辑一致) ======
    tokenizer = AutoTokenizer.from_pretrained(
        args.version, model_max_length=args.model_max_length,
        padding_side="right", use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]

    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    kwargs = {"torch_dtype": torch_dtype}
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig
        kwargs.update({
            "torch_dtype": torch.half,
            "load_in_4bit": True,
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                llm_int8_skip_modules=["visual_model"],
            ),
        })
    elif args.load_in_8bit:
        from transformers import BitsAndBytesConfig
        kwargs.update({
            "torch_dtype": torch.half,
            "quantization_config": BitsAndBytesConfig(
                llm_int8_skip_modules=["visual_model"],
                load_in_8bit=True,
            ),
        })

    extra_kw = {}
    if args.load_in_4bit or args.load_in_8bit:
        extra_kw["device_map"] = "auto"
    model = LISAForCausalLM.from_pretrained(
        args.version, low_cpu_mem_usage=True,
        vision_tower=args.vision_tower,
        seg_token_idx=seg_token_idx,
        **kwargs, **extra_kw,
    )
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()

    if args.load_in_4bit or args.load_in_8bit:
        vision_tower.to(dtype=torch.half, device="cuda")
    elif args.precision == "bf16":
        vision_tower.to(dtype=torch_dtype)
        model = model.bfloat16().cuda()
    else:
        vision_tower.to(dtype=torch_dtype)
        model = model.float().cuda()
    model.eval()

    clip_processor = CLIPImageProcessor.from_pretrained(model.config.vision_tower)
    transform = ResizeLongestSide(args.image_size)
    conversation_lib.default_conversation = conversation_lib.conv_templates[args.conv_type]

    # ====== 加载评估数据 ======
    samples = load_eval_samples(args.dataset_dir, args.val_dataset)
    if args.max_samples > 0:
        samples = samples[:args.max_samples]
    if args.num_shards > 1:
        total = len(samples)
        shard_size = (total + args.num_shards - 1) // args.num_shards
        start = args.shard_id * shard_size
        end = min(start + shard_size, total)
        samples = samples[start:end]
        print(f"Shard {args.shard_id}/{args.num_shards}: samples [{start}:{end}] = {len(samples)}")
    print(f"Loaded {len(samples)} (image, class) evaluation pairs")

    if args.save_vis:
        os.makedirs(args.vis_save_path, exist_ok=True)

    # ====== 逐样本推理 ======
    per_class_intersection = {}
    per_class_union = {}
    per_class_iou_list = {}
    total_giou_sum = 0.0
    total_count = 0

    for sample in tqdm.tqdm(samples, desc="Evaluating"):
        image_path = sample["image_path"]
        label_path = sample["label_path"]
        class_id = sample["class_id"]
        class_name = sample["class_name"]

        image_np = cv2.imread(image_path)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        original_size = image_np.shape[:2]

        label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
        gt_mask = (label == class_id).astype(np.uint8)

        # --- 构建 prompt (与 chat.py 一致) ---
        conv = conversation_lib.default_conversation.copy()
        conv.messages = []
        prompt_text = (DEFAULT_IMAGE_TOKEN + "\n"
                       + f"What is {class_name} in this image? "
                       + "Please output segmentation mask.")
        if args.use_mm_start_end:
            prompt_text = prompt_text.replace(
                DEFAULT_IMAGE_TOKEN,
                DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN,
            )
        conv.append_message(conv.roles[0], prompt_text)
        conv.append_message(conv.roles[1], "")
        prompt = conv.get_prompt()

        # --- 图像预处理 ---
        # 4/8-bit: model loaded as fp16, so images must be fp16 too
        # Otherwise: use the specified precision
        if args.load_in_4bit or args.load_in_8bit:
            img_dtype = torch.half
        else:
            img_dtype = torch_dtype
        image_clip = clip_processor.preprocess(
            image_np, return_tensors="pt"
        )["pixel_values"][0].unsqueeze(0).cuda().to(img_dtype)
        image_sam, resize = preprocess_image(image_np, transform, args.image_size)
        image_sam = image_sam.unsqueeze(0).cuda().to(img_dtype)

        input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        input_ids = input_ids.unsqueeze(0).cuda()

        # --- 推理: 用 model.evaluate() 自由生成 ---
        with torch.no_grad():
            output_ids, pred_masks = model.evaluate(
                image_clip, image_sam, input_ids,
                [resize], [original_size],
                max_new_tokens=512, tokenizer=tokenizer,
            )

        # --- 提取预测 mask ---
        if len(pred_masks) == 0 or pred_masks[0].shape[0] == 0:
            pred_binary = np.zeros(original_size, dtype=np.uint8)
        else:
            pred_binary = (pred_masks[0][0].detach().cpu().numpy() > 0).astype(np.uint8)

        # --- 计算指标 ---
        intersection = np.logical_and(pred_binary, gt_mask).sum().item()
        union = np.logical_or(pred_binary, gt_mask).sum().item()
        sample_iou = intersection / (union + 1e-10)

        per_class_intersection[class_name] = per_class_intersection.get(class_name, 0) + intersection
        per_class_union[class_name] = per_class_union.get(class_name, 0) + union
        if class_name not in per_class_iou_list:
            per_class_iou_list[class_name] = []
        per_class_iou_list[class_name].append(sample_iou)

        total_giou_sum += sample_iou
        total_count += 1

        # --- 可视化 ---
        if args.save_vis:
            vis = image_np.copy()
            green_overlay = np.zeros_like(vis)
            green_overlay[gt_mask > 0] = [0, 255, 0]
            red_overlay = np.zeros_like(vis)
            red_overlay[pred_binary > 0] = [255, 0, 0]
            vis_gt = cv2.addWeighted(vis, 0.6, green_overlay, 0.4, 0)
            vis_pred = cv2.addWeighted(vis, 0.6, red_overlay, 0.4, 0)
            combined = np.concatenate([vis, vis_gt, vis_pred], axis=1)
            stem = os.path.splitext(os.path.basename(image_path))[0]
            out_name = f"{stem}_{class_name.replace(' ','_')}_iou{sample_iou:.2f}.jpg"
            cv2.imwrite(
                os.path.join(args.vis_save_path, out_name),
                cv2.cvtColor(combined, cv2.COLOR_RGB2BGR),
            )

    # ====== 汇总指标 ======
    giou = total_giou_sum / (total_count + 1e-10)

    total_inter = sum(per_class_intersection.values())
    total_union = sum(per_class_union.values())
    ciou = total_inter / (total_union + 1e-10)

    per_class_iou = {}
    for cls_name in sorted(CLASS_NAMES.values()):
        if per_class_union.get(cls_name, 0) > 0:
            per_class_iou[cls_name] = (
                per_class_intersection[cls_name] / (per_class_union[cls_name] + 1e-10)
            )
        else:
            per_class_iou[cls_name] = None

    valid_ious = [v for v in per_class_iou.values() if v is not None]
    miou = sum(valid_ious) / len(valid_ious) if valid_ious else 0.0

    # ====== 打印结果 ======
    print("\n" + "=" * 65)
    print(f"Model:   {args.version}")
    print(f"Dataset: {args.val_dataset}")
    print("=" * 65)
    print(f"  gIoU (avg per-sample IoU):    {giou:.4f}")
    print(f"  cIoU (global IoU):            {ciou:.4f}")
    print(f"  mIoU (avg per-class IoU):     {miou:.4f}")
    print(f"  Evaluated pairs:              {total_count}")
    print(f"  Valid classes:                {len(valid_ious)}/{len(CLASS_NAMES)}")
    print()
    print(f"  {'Class':<25} {'IoU':>8}  {'Count':>6}")
    print("  " + "-" * 43)
    for cls_name in sorted(CLASS_NAMES.values()):
        iou_val = per_class_iou.get(cls_name)
        count = len(per_class_iou_list.get(cls_name, []))
        if iou_val is not None:
            print(f"  {cls_name:<25} {iou_val:>8.4f}  {count:>6}")
        else:
            print(f"  {cls_name:<25} {'N/A':>8}  {count:>6}")
    print("=" * 65)

    # ====== 保存 JSON ======
    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        results = {
            "model": args.version,
            "val_dataset": args.val_dataset,
            "giou": giou,
            "ciou": ciou,
            "miou": miou,
            "total_samples": total_count,
            "per_class_iou": per_class_iou,
            "per_class_count": {k: len(v) for k, v in per_class_iou_list.items()},
            "per_class_intersection": {k: int(v) for k, v in per_class_intersection.items()},
            "per_class_union": {k: int(v) for k, v in per_class_union.items()},
        }
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Results saved to: {args.output_json}")


if __name__ == "__main__":
    main()
