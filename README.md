# LISA-MUSeg: Fine-tuning LISA for Underground Coal Mine Segmentation

This repository contains the code for fine-tuning [LISA](https://github.com/XinLai/LISA) (Large Language Instructed Segmentation Assistant) on the MUSeg dataset for semantic segmentation in underground coal mine scenes.

## Overview

LISA is a reasoning segmentation model that combines Large Language Models (LLM) with Segment Anything Model (SAM). This project adapts LISA for the specific domain of underground coal mine segmentation using LoRA fine-tuning.

### Key Features

- **LoRA Fine-tuning**: Parameter-efficient fine-tuning using Low-Rank Adaptation
- **Multi-GPU Support**: Single-GPU and multi-GPU DDP training
- **Comprehensive Evaluation**: Automated evaluation with IoU metrics (gIoU, cIoU, mIoU)
- **Ablation Studies**: Automated hyperparameter search for LoRA rank, epochs, and learning rate

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/LISA-MUSeg.git
cd LISA-MUSeg

# Install dependencies
pip install -r requirements.txt
```

## Dataset Preparation

### MUSeg Dataset

The MUSeg dataset contains 15 semantic classes for underground coal mine scenes:

| Class ID | Class Name |
|----------|------------|
| 1 | person |
| 2 | cable |
| 3 | tube |
| 4 | indicator |
| 5 | electrical equipment |
| 6 | electronic equipment |
| 7 | mining equipment |
| 8 | rail area |
| 9 | support equipment |
| 10 | door |
| 11 | tools and materials |
| 12 | rescue equipment |
| 13 | container |
| 14 | metal fixture |
| 15 | anchoring equipment |

### Generate Data Indices

```bash
python scripts/convert_museg_to_lisa.py \
    --museg_root /path/to/MUSeg \
    --output_dir ./dataset/museg \
    --train_mines 01 03 06 \
    --val_mines 02 \
    --test_mines 04 05
```

## Training

### Single-GPU Training

```bash
python train_museg.py \
    --version xinlai/LISA-7B-v1 \
    --conv_type llava_v1 \
    --precision bf16 \
    --exp_name lisa-7b-museg-lora-r8 \
    --lora_r 8 \
    --lora_alpha 16 \
    --epochs 10 \
    --steps_per_epoch 500 \
    --batch_size 2 \
    --grad_accumulation_steps 2 \
    --lr 3e-4 \
    --workers 4
```

### Multi-GPU Training (DDP)

```bash
torchrun --nproc_per_node=2 train_museg.py \
    --version xinlai/LISA-7B-v1 \
    --conv_type llava_v1 \
    --precision bf16 \
    --exp_name lisa-7b-museg-lora-r8 \
    --lora_r 8 \
    --lora_alpha 16 \
    --epochs 10 \
    --steps_per_epoch 500 \
    --batch_size 2 \
    --grad_accumulation_steps 2 \
    --lr 3e-4 \
    --workers 4
```

### Key Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--lora_r` | 8 | LoRA rank |
| `--lora_alpha` | 16 | LoRA alpha parameter |
| `--epochs` | 10 | Number of training epochs |
| `--steps_per_epoch` | 500 | Training steps per epoch |
| `--lr` | 3e-4 | Learning rate |
| `--batch_size` | 2 | Batch size per GPU |
| `--grad_accumulation_steps` | 10 | Gradient accumulation steps |

## Evaluation

### Single Experiment Evaluation

```bash
python scripts/eval_museg.py \
    --version path/to/model \
    --precision bf16 \
    --dataset_dir ./dataset \
    --val_dataset "museg|test"
```

### Parallel Evaluation (Sharded)

```bash
# Run shards in parallel
python scripts/eval_museg.py --version ... --num_shards 2 --shard_id 0
python scripts/eval_museg.py --version ... --num_shards 2 --shard_id 1

# Merge results
python scripts/merge_shard_results.py results/A3_shard0.json results/A3_shard1.json results/A3_merged.json
```

## Experiments

### Run All Experiments

```bash
python scripts/run_experiments.py --experiment all
```

### Run Specific Experiments

```bash
# Experiment A: Main comparison (Baseline vs LoRA fine-tuned)
python scripts/run_experiments.py --experiment A

# Experiment B: Ablation study (LoRA rank, epochs, learning rate)
python scripts/run_experiments.py --experiment B

# Experiment C: Cross-mine generalization (Leave-one-out)
python scripts/run_experiments.py --experiment C
```

### Experiment Design

| Experiment | Description |
|------------|-------------|
| A1 | LISA-7B zero-shot baseline |
| A3 | LISA-7B + LoRA (r=8, ep=10, lr=3e-4) |
| B1 | LoRA rank ablation (r=4, 8, 16) |
| B2 | Epoch ablation (5, 10, 15, 20) |
| B3 | Learning rate ablation (1e-4, 3e-4, 5e-4) |
| C1-C4 | Sampling strategy ablation |

## Results Summary

### Main Results

| Experiment | gIoU | cIoU | mIoU |
|------------|------|------|------|
| A1 (Baseline) | 0.1813 | 0.1498 | 0.2225 |
| A3 (LoRA) | 0.2865 | 0.3854 | 0.3484 |

### Ablation Results

| Experiment | gIoU | cIoU | mIoU |
|------------|------|------|------|
| B1_r4 | 0.2751 | 0.3025 | 0.3351 |
| B1_r16 | 0.2744 | 0.2964 | 0.3206 |
| B2_ep20 | 0.2802 | 0.3729 | 0.3356 |
| B3_lr1e4 | 0.2645 | 0.3558 | 0.3314 |
| B3_lr5e4 | 0.2535 | 0.3505 | 0.3165 |

## Generate Paper Tables

```bash
python scripts/generate_paper_tables.py --results_dir ./results
```

## Merge LoRA Weights

```bash
python merge_lora_weights_and_save_hf_model.py \
    --version xinlai/LISA-7B-v1 \
    --weight runs/lisa-7b-museg-lora-r8/best.pt \
    --save_path ./models/lisa-7b-museg-merged
```

## Interactive Demo

### Command Line

```bash
python chat.py --version path/to/merged/model --precision bf16
```

### Web Demo

```bash
python app.py --version path/to/merged/model --precision bf16
```

## Project Structure

```
LISA-MUSeg/
├── train_museg.py                      # Main training script
├── merge_lora_weights_and_save_hf_model.py  # Merge LoRA weights
├── chat.py                             # Interactive CLI demo
├── app.py                              # Web demo
├── model/
│   ├── LISA.py                         # LISA model definition
│   ├── llava/                          # LLaVA base model
│   └── segment_anything/               # SAM model
├── utils/
│   ├── museg_dataset.py                # MUSeg dataset loader
│   ├── dataset.py                      # Dataset dispatcher
│   ├── conversation.py                 # Conversation templates
│   └── utils.py                        # Utility functions
├── scripts/
│   ├── convert_museg_to_lisa.py        # Generate data indices
│   ├── eval_museg.py                   # Evaluation script
│   ├── merge_shard_results.py          # Merge evaluation shards
│   ├── run_experiments.py              # Run ablation experiments
│   ├── summarize_experiment_results.py # Generate summary tables
│   └── generate_paper_tables.py        # Generate LaTeX tables
├── dataset/                            # Data indices (generated)
├── runs/                               # Training checkpoints
├── results/                            # Evaluation results
└── vis_output/                         # Visualization outputs
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [LISA](https://github.com/XinLai/LISA) - Original LISA model
- [LLaVA](https://github.com/haotian-liu/LLaVA) - Vision-language model
- [Segment Anything](https://github.com/facebookresearch/segment-anything) - SAM model

## Citation

If you find this code useful, please cite:

```bibtex
@article{lai2023lisa,
  title={LISA: Reasoning Segmentation via Large Language Model},
  author={Lai, Xin and Tian, Zhuotao and Chen, Yukang and Li, Yanwei and Yuan, Yuhui and Liu, Shu and Jia, Jiaya},
  journal={arXiv preprint arXiv:2308.00692},
  year={2023}
}
```
