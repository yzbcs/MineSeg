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

### Generate Data Indices

```bash
python3 scripts/convert_museg_to_lisa.py \
    --museg_root /path/to/MUSeg \
    --output_dir ./dataset/museg \
    --train_mines 01 03 06 \
    --val_mines 02 \
    --test_mines 04 05
```

## Training

### Single-GPU Training

```bash
python3 train_museg.py \
    --version xinlai/LISA-7B-v1 \
    --conv_type llava_v1 \
    --precision bf16 \
    --exp_name lisa-7b-museg-lora-r8 \
    --lora_r 8 \
    --lora_alpha 16 \
    --epochs 10 \
    --steps_per_epoch 500 \
    --batch_size 2 \
    --grad_accumulation_steps 10 \
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
    --grad_accumulation_steps 10 \
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

```bash
python3 scripts/eval_museg.py \
    --version path/to/model \
    --precision bf16 \
    --dataset_dir ./dataset \
    --val_dataset "museg|test"
```

## Merge LoRA Weights

```bash
python3 merge_lora_weights_and_save_hf_model.py \
    --version xinlai/LISA-7B-v1 \
    --weight runs/lisa-7b-museg-lora-r8/best.pt \
    --save_path ./models/lisa-7b-museg-merged
```

## Project Structure

```
LISA-MUSeg/
├── train_museg.py                         # Main training script
├── merge_lora_weights_and_save_hf_model.py # Merge LoRA weights
├── chat.py                                # Interactive CLI demo
├── app.py                                 # Web demo
├── model/
│   ├── LISA.py                            # LISA model definition
│   ├── llava/                             # LLaVA base model
│   └── segment_anything/                  # SAM model
├── utils/
│   ├── museg_dataset.py                   # MUSeg dataset loader
│   ├── dataset.py                         # Dataset dispatcher
│   ├── conversation.py                    # Conversation templates
│   └── utils.py                           # Utility functions
├── scripts/
│   ├── convert_museg_to_lisa.py           # Generate data indices
│   ├── eval_museg.py                      # Evaluation script
│   ├── merge_shard_results.py             # Merge evaluation shards
│   ├── run_experiments.py                 # Run experiments
│   ├── summarize_experiment_results.py    # Generate summary tables
├── dataset/                               # Data indices (generated)
├── runs/                                  # Training checkpoints
├── results/                               # Evaluation results
└── vis_output/                            # Visualization outputs
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

This project is based on [LISA](https://github.com/XinLai/LISA) by Xin Lai et al., also licensed under Apache License 2.0.

## Acknowledgments

- [LISA](https://github.com/XinLai/LISA) - Original LISA model
- [LLaVA](https://github.com/haotian-liu/LLaVA) - Vision-language model
- [Segment Anything](https://github.com/facebookresearch/segment-anything) - SAM model
- [MUSeg](https://www.nature.com/articles/s41597-025-05493-9) - Multimodal semantic segmentation dataset for underground mine scenes

