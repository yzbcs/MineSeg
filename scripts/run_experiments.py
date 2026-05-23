"""
统一实验运行器

主实验 (Baseline vs Fine-tuned):
  - LISA-7B  原始 → 零样本评估
  - LISA-7B  + LoRA 微调 → 评估

消融实验 (基于 7B):
  - LoRA rank  = 4, 8, 16
  - 训练 epoch = 5, 10, 15, 20
  - 学习率     = 1e-4, 3e-4, 5e-4
  - 采样策略   = fixed-1, fixed-3, random-1-3, all

用法:
  python scripts/run_experiments.py --experiment main
  python scripts/run_experiments.py --experiment ablation
  python scripts/run_experiments.py --experiment all
"""

import argparse
import json
import os
import subprocess
import sys


LISA_MAIN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODELS_DIR = os.path.join(LISA_MAIN, "..", "models")

MODEL_CONFIGS = {
    "7b": {
        "hf_id": "xinlai/LISA-7B-v1",
        "conv_type": "llava_v1",
        "precision": "bf16",
    },
}


def resolve_model_path(hf_id):
    """Use local model if available, otherwise fall back to HF model ID."""
    local_name = hf_id.split("/")[-1]
    local_path = os.path.join(MODELS_DIR, local_name)
    if os.path.isdir(local_path):
        print(f"Using local model: {local_path}")
        return local_path
    print(f"Using HuggingFace model: {hf_id}")
    return hf_id


def run_cmd(cmd, cwd=LISA_MAIN):
    print(f"\n>>> {' '.join(cmd)}\n")
    subprocess.run(cmd, cwd=cwd, check=True)


def run_eval(version, conv_type, val_dataset, output_json, precision="bf16",
             dataset_dir="./dataset", save_vis=False, vis_path=None,
             load_in_4bit=True):
    cmd = [
        sys.executable, "scripts/eval_museg.py",
        f"--version={version}",
        f"--dataset_dir={dataset_dir}",
        f"--val_dataset={val_dataset}",
        f"--precision={precision}",
        f"--conv_type={conv_type}",
        f"--output_json={output_json}",
    ]
    if load_in_4bit:
        cmd.append("--load_in_4bit")
    if save_vis and vis_path:
        cmd += ["--save_vis", f"--vis_save_path={vis_path}"]
    run_cmd(cmd)


def run_train(model_size, exp_name, lora_r=8, lora_alpha=16, epochs=20,
              steps_per_epoch=500, lr=3e-4, batch_size=1,
              grad_accumulation_steps=20, dataset_dir="./dataset",
              museg_data="museg|train", val_dataset="museg|val",
              load_in_4bit=True, sample_strategy="random-1-3"):
    """Launch train_museg.py (single-GPU, no DeepSpeed)."""
    cfg = MODEL_CONFIGS[model_size]
    version = resolve_model_path(cfg["hf_id"])
    cmd = [
        sys.executable, "train_museg.py",
        f"--version={version}",
        f"--conv_type={cfg['conv_type']}",
        f"--precision={cfg['precision']}",
        f"--exp_name={exp_name}",
        f"--lora_r={lora_r}",
        f"--lora_alpha={lora_alpha}",
        f"--epochs={epochs}",
        f"--steps_per_epoch={steps_per_epoch}",
        f"--lr={lr}",
        f"--batch_size={batch_size}",
        f"--grad_accumulation_steps={grad_accumulation_steps}",
        f"--dataset_dir={dataset_dir}",
        f"--museg_data={museg_data}",
        f"--val_dataset={val_dataset}",
        f"--sample_strategy={sample_strategy}",
    ]
    if load_in_4bit:
        cmd.append("--load_in_4bit")
    run_cmd(cmd)


def merge_weights(model_size, exp_name, lora_r=8, lora_alpha=16):
    """Load base model + LoRA, load trained weights, merge, save as HF model."""
    cfg = MODEL_CONFIGS[model_size]
    version = resolve_model_path(cfg["hf_id"])
    weight_path = os.path.join(LISA_MAIN, "runs", exp_name, "best.pt")
    if not os.path.exists(weight_path):
        weight_path = os.path.join(LISA_MAIN, "runs", exp_name, "epoch_0.pt")
    save_path = os.path.join(MODELS_DIR, exp_name)

    cmd = [
        sys.executable, "merge_lora_weights_and_save_hf_model.py",
        f"--version={version}",
        f"--conv_type={cfg['conv_type']}",
        f"--lora_r={lora_r}",
        f"--lora_alpha={lora_alpha}",
        f"--weight={weight_path}",
        f"--save_path={save_path}",
    ]
    run_cmd(cmd)
    return save_path


# ====== 主实验: 零样本 vs LoRA 微调 ======
def experiment_main(args):
    os.makedirs(os.path.join(LISA_MAIN, "results"), exist_ok=True)

    print("\n" + "=" * 60)
    print("主实验: 零样本基线 vs LoRA 微调")
    print("=" * 60)

    # 零样本 LISA-7B
    print("\n--- 零样本 LISA-7B ---")
    run_eval(
        resolve_model_path(MODEL_CONFIGS["7b"]["hf_id"]),
        MODEL_CONFIGS["7b"]["conv_type"],
        "museg|test", "results/baseline_7b.json",
        save_vis=True, vis_path="vis_output/baseline_7b",
    )

    # LISA-7B + LoRA 微调
    print("\n--- LISA-7B + LoRA 微调 ---")
    run_train("7b", "lisa-7b-museg-lora-r8")
    merged_7b = merge_weights("7b", "lisa-7b-museg-lora-r8")
    run_eval(
        merged_7b, MODEL_CONFIGS["7b"]["conv_type"],
        "museg|test", "results/finetuned_7b.json",
        save_vis=True, vis_path="vis_output/finetuned_7b",
    )


# ====== 消融实验 ======
def experiment_ablation(args):
    os.makedirs(os.path.join(LISA_MAIN, "results"), exist_ok=True)

    print("\n" + "=" * 60)
    print("消融实验 (基于 LISA-7B)")
    print("=" * 60)

    # LoRA rank 消融
    for r in [4, 8, 16]:
        name = f"ablation-lora-r{r}"
        print(f"\n--- LoRA rank={r} ---")
        run_train("7b", name, lora_r=r, lora_alpha=r * 2)
        merged = merge_weights("7b", name, lora_r=r, lora_alpha=r * 2)
        run_eval(merged, "llava_v1", "museg|test", f"results/lora_r{r}.json")

    # 训练 epoch 消融
    for ep in [5, 10, 15, 20]:
        name = f"ablation-epoch-{ep}"
        print(f"\n--- Epochs={ep} ---")
        run_train("7b", name, epochs=ep)
        merged = merge_weights("7b", name)
        run_eval(merged, "llava_v1", "museg|test", f"results/epoch_{ep}.json")

    # 学习率消融
    for lr in [1e-4, 3e-4, 5e-4]:
        lr_str = f"{lr:.0e}".replace("+", "").replace("-0", "-")
        name = f"ablation-lr-{lr_str}"
        print(f"\n--- LR={lr} ---")
        run_train("7b", name, lr=lr)
        merged = merge_weights("7b", name)
        run_eval(merged, "llava_v1", "museg|test", f"results/lr_{lr_str}.json")

    # 采样策略消融
    for strategy in ["fixed-1", "fixed-3", "random-1-3", "all"]:
        name = f"ablation-sample-{strategy}"
        print(f"\n--- Sample strategy={strategy} ---")
        run_train("7b", name, sample_strategy=strategy)
        merged = merge_weights("7b", name)
        run_eval(merged, "llava_v1", "museg|test", f"results/sample_{strategy}.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="all",
                        choices=["main", "ablation", "all"])
    args = parser.parse_args()

    if args.experiment in ("main", "all"):
        experiment_main(args)
    if args.experiment in ("ablation", "all"):
        experiment_ablation(args)

    print("\n" + "=" * 60)
    print("实验完成! 运行以下命令生成论文表格:")
    print("  python scripts/generate_paper_tables.py --results_dir ./results")
    print("=" * 60)


if __name__ == "__main__":
    main()
