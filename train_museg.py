"""
MUSeg fine-tuning for LISA — supports single-GPU and multi-GPU DDP.

Uses native PyTorch AMP + gradient accumulation.
Saves LoRA + trainable weights as a standard .pt file.

Single-GPU:
    python train_museg.py --version ... --exp_name ...

Multi-GPU (DDP):
    torchrun --nproc_per_node=2 train_museg.py --version ... --exp_name ...
"""

import argparse
import os
import shutil
import sys
import time
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import tqdm
import transformers
from peft import LoraConfig, get_peft_model
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from model.LISA import LISAForCausalLM
from model.llava import conversation as conversation_lib
from utils.dataset import HybridDataset, ValDataset, collate_fn
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         AverageMeter, ProgressMeter, Summary,
                         intersectionAndUnionGPU)


def parse_args():
    p = argparse.ArgumentParser(description="LISA MUSeg Fine-Tuning")
    p.add_argument("--version", required=True, type=str,
                   help="HF model id or local path, e.g. xinlai/LISA-7B-v1")
    p.add_argument("--vision_tower", default="openai/clip-vit-large-patch14")
    p.add_argument("--vision_pretrained", default="", type=str,
                   help="SAM ViT-H checkpoint path (only for training from LLaVA base)")
    p.add_argument("--precision", default="bf16", choices=["fp32", "bf16", "fp16"])
    p.add_argument("--image_size", default=1024, type=int)
    p.add_argument("--model_max_length", default=512, type=int)
    p.add_argument("--load_in_4bit", action="store_true", default=False)
    p.add_argument("--load_in_8bit", action="store_true", default=False)

    # LoRA
    p.add_argument("--lora_r", default=8, type=int)
    p.add_argument("--lora_alpha", default=16, type=int)
    p.add_argument("--lora_dropout", default=0.05, type=float)
    p.add_argument("--lora_target_modules", default="q_proj,v_proj", type=str)

    # Data
    p.add_argument("--dataset", default="museg", type=str)
    p.add_argument("--sample_rates", default="1", type=str)
    p.add_argument("--museg_data", default="museg|train", type=str)
    p.add_argument("--val_dataset", default="museg|val", type=str)
    p.add_argument("--dataset_dir", default="./dataset", type=str)
    p.add_argument("--num_classes_per_sample", default=3, type=int)

    # Training
    p.add_argument("--epochs", default=10, type=int)
    p.add_argument("--steps_per_epoch", default=500, type=int)
    p.add_argument("--batch_size", default=2, type=int)
    p.add_argument("--grad_accumulation_steps", default=10, type=int)
    p.add_argument("--lr", default=3e-4, type=float)
    p.add_argument("--beta1", default=0.9, type=float)
    p.add_argument("--beta2", default=0.95, type=float)
    p.add_argument("--warmup_steps", default=100, type=int)
    p.add_argument("--max_grad_norm", default=1.0, type=float)

    # Loss weights
    p.add_argument("--ce_loss_weight", default=1.0, type=float)
    p.add_argument("--dice_loss_weight", default=0.5, type=float)
    p.add_argument("--bce_loss_weight", default=2.0, type=float)

    # Misc
    p.add_argument("--log_base_dir", default="./runs", type=str)
    p.add_argument("--exp_name", default="museg_7b", type=str)
    p.add_argument("--workers", default=4, type=int)
    p.add_argument("--print_freq", default=1, type=int)
    p.add_argument("--no_eval", action="store_true", default=False)
    p.add_argument("--out_dim", default=256, type=int)
    p.add_argument("--conv_type", default="llava_v1",
                   choices=["llava_v1", "llava_llama_2"])
    p.add_argument("--use_mm_start_end", action="store_true", default=True)
    p.add_argument("--gradient_checkpointing", action="store_true", default=True)
    p.add_argument("--train_mask_decoder", action="store_true", default=True)
    p.add_argument("--resume", default="", type=str)
    p.add_argument("--exclude_val", action="store_true", default=False)

    return p.parse_args()


def find_linear_layers(model, target_module_names):
    result = set()
    excluded = {"visual_model", "vision_tower", "mm_projector", "text_hidden_fcs"}
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if any(x in name for x in excluded):
            continue
        if any(x in name for x in target_module_names):
            result.add(name)
    return sorted(result)


def dict_to_cuda(d):
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            d[k] = v.cuda(non_blocking=True)
        elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], torch.Tensor):
            d[k] = [t.cuda(non_blocking=True) for t in v]
    return d


def get_lr_schedule(optimizer, warmup_steps, total_steps):
    """Linear warmup then cosine decay."""
    from torch.optim.lr_scheduler import LambdaLR
    import math

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def save_trainable_weights(model, save_path):
    """Save only LoRA and other trainable parameters."""
    state = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            state[name] = param.data.cpu()
    torch.save(state, save_path)
    print(f"Saved trainable weights to {save_path} ({len(state)} tensors)")


def is_dist():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    return dist.get_rank() if is_dist() else 0


def get_world_size():
    return dist.get_world_size() if is_dist() else 1


def is_main_process():
    return get_rank() == 0


def main():
    args = parse_args()

    use_ddp = int(os.environ.get("WORLD_SIZE", 1)) > 1
    if use_ddp:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
    else:
        local_rank = 0

    log_dir = os.path.join(args.log_base_dir, args.exp_name)
    if is_main_process():
        os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir) if is_main_process() else None

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    # ======================== Tokenizer ========================
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version, model_max_length=args.model_max_length,
        padding_side="right", use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    num_new_tokens = tokenizer.add_tokens("[SEG]")
    seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]

    if args.use_mm_start_end:
        num_new_tokens += tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
        )

    # ======================== Model ========================
    torch_dtype = {"fp32": torch.float32, "bf16": torch.bfloat16,
                   "fp16": torch.float16}[args.precision]

    model_kwargs = {
        "train_mask_decoder": args.train_mask_decoder,
        "out_dim": args.out_dim,
        "ce_loss_weight": args.ce_loss_weight,
        "dice_loss_weight": args.dice_loss_weight,
        "bce_loss_weight": args.bce_loss_weight,
        "seg_token_idx": seg_token_idx,
        "vision_tower": args.vision_tower,
        "use_mm_start_end": args.use_mm_start_end,
    }
    if args.vision_pretrained:
        model_kwargs["vision_pretrained"] = args.vision_pretrained

    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            llm_int8_skip_modules=["visual_model", "text_hidden_fcs"],
        )
        model_kwargs["torch_dtype"] = torch_dtype
        model_kwargs["device_map"] = "auto"
    elif args.load_in_8bit:
        model_kwargs["load_in_8bit"] = True
        model_kwargs["torch_dtype"] = torch_dtype
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["torch_dtype"] = torch_dtype

    model = LISAForCausalLM.from_pretrained(
        args.version, low_cpu_mem_usage=True, **model_kwargs,
    )
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.enable_input_require_grads()
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype)

    # Only call initialize_lisa_modules for LLaVA base models.
    # For LISA models, SAM and text_hidden_fcs are already loaded by from_pretrained.
    if not hasattr(model.get_model(), "visual_model"):
        model.get_model().initialize_lisa_modules(model.get_model().config)

    for p in vision_tower.parameters():
        p.requires_grad = False
    for p in model.get_model().mm_projector.parameters():
        p.requires_grad = False

    conversation_lib.default_conversation = conversation_lib.conv_templates[args.conv_type]

    # ======================== LoRA ========================
    if args.lora_r > 0:
        targets = find_linear_layers(model, args.lora_target_modules.split(","))
        lora_config = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha,
            target_modules=targets, lora_dropout=args.lora_dropout,
            bias="none", task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    if num_new_tokens > 0:
        model.resize_token_embeddings(len(tokenizer))

    for n, p in model.named_parameters():
        if any(x in n for x in ["lm_head", "embed_tokens", "mask_decoder",
                                 "text_hidden_fcs"]):
            if p.dtype.is_floating_point:
                p.requires_grad = True

    if not args.load_in_4bit and not args.load_in_8bit:
        model = model.to(dtype=torch_dtype).cuda()

    if use_ddp:
        model = DDP(model, device_ids=[local_rank],
                    find_unused_parameters=False, static_graph=True)

    # ======================== Dataset ========================
    world_size = get_world_size()
    samples_per_epoch = (args.batch_size * args.grad_accumulation_steps
                         * args.steps_per_epoch * world_size)
    train_dataset = HybridDataset(
        args.dataset_dir, tokenizer, args.vision_tower,
        samples_per_epoch=samples_per_epoch,
        precision=args.precision, image_size=args.image_size,
        num_classes_per_sample=args.num_classes_per_sample,
        exclude_val=args.exclude_val, dataset=args.dataset,
        sample_rate=[float(x) for x in args.sample_rates.split(",")],
        museg_data=args.museg_data,
    )

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if use_ddp else None
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.workers, pin_memory=True, drop_last=True,
        collate_fn=partial(collate_fn, tokenizer=tokenizer,
                           conv_type=args.conv_type,
                           use_mm_start_end=args.use_mm_start_end,
                           local_rank=local_rank),
    )

    val_dataset = None
    val_loader = None
    if not args.no_eval:
        val_dataset = ValDataset(
            args.dataset_dir, tokenizer, args.vision_tower,
            args.val_dataset, args.image_size,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=1, shuffle=False,
            num_workers=args.workers, pin_memory=True,
            collate_fn=partial(collate_fn, tokenizer=tokenizer,
                               conv_type=args.conv_type,
                               use_mm_start_end=args.use_mm_start_end,
                               local_rank=local_rank),
        )
        if is_main_process():
            print(f"Train: {len(train_dataset)} samples | Val: {len(val_dataset)} samples")
    else:
        if is_main_process():
            print(f"Train: {len(train_dataset)} samples | Val: disabled")

    # ======================== Optimizer & Scheduler ========================
    raw_model = model.module if use_ddp else model
    trainable_params = [p for p in raw_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params, lr=args.lr, weight_decay=0.0,
        betas=(args.beta1, args.beta2),
    )
    total_steps = args.epochs * args.steps_per_epoch
    scheduler = get_lr_schedule(optimizer, args.warmup_steps, total_steps)

    use_amp = args.precision in ("bf16", "fp16")
    amp_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
    scaler = GradScaler(enabled=(args.precision == "fp16"))

    # ======================== Resume ========================
    start_epoch = 0
    best_giou = 0.0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu")
        state_dict = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) else ckpt
        missing, unexpected = raw_model.load_state_dict(state_dict, strict=False)
        if is_main_process():
            print(f"Resumed from {args.resume} (missing={len(missing)}, "
                  f"unexpected={len(unexpected)})")
        if isinstance(ckpt, dict) and "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        if isinstance(ckpt, dict) and "epoch" in ckpt:
            start_epoch = ckpt["epoch"] + 1
        if isinstance(ckpt, dict) and "best_giou" in ckpt:
            best_giou = ckpt["best_giou"]

    # ======================== Training ========================
    global_step = start_epoch * args.steps_per_epoch
    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        train_iter = iter(train_loader)

        losses_meter = AverageMeter("Loss", ":.4f")
        ce_meter = AverageMeter("CE", ":.4f")
        bce_meter = AverageMeter("BCE", ":.4f")
        dice_meter = AverageMeter("Dice", ":.4f")
        batch_time = AverageMeter("Time", ":6.2f")

        progress = ProgressMeter(
            args.steps_per_epoch,
            [batch_time, losses_meter, ce_meter, bce_meter, dice_meter],
            prefix=f"Epoch [{epoch}/{args.epochs}]",
        )

        end = time.time()
        for step in range(args.steps_per_epoch):
            optimizer.zero_grad()

            for _micro in range(args.grad_accumulation_steps):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    batch = next(train_iter)

                batch = dict_to_cuda(batch)
                if args.precision == "bf16":
                    batch["images"] = batch["images"].bfloat16()
                    batch["images_clip"] = batch["images_clip"].bfloat16()
                elif args.precision == "fp16":
                    batch["images"] = batch["images"].half()
                    batch["images_clip"] = batch["images_clip"].half()

                with autocast(enabled=use_amp, dtype=amp_dtype):
                    out = model(**batch)
                    loss = out["loss"] / args.grad_accumulation_steps

                scaler.scale(loss).backward()

                bs = batch["images"].size(0)
                losses_meter.update(out["loss"].item(), bs)
                ce_meter.update(out["ce_loss"].item(), bs)
                bce_meter.update(out["mask_bce_loss"].item(), bs)
                dice_meter.update(out["mask_dice_loss"].item(), bs)

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1

            batch_time.update(time.time() - end)
            end = time.time()

            if (step + 1) % args.print_freq == 0 and is_main_process():
                progress.display(step + 1)
                writer.add_scalar("train/loss", losses_meter.avg, global_step)
                writer.add_scalar("train/ce_loss", ce_meter.avg, global_step)
                writer.add_scalar("train/mask_bce", bce_meter.avg, global_step)
                writer.add_scalar("train/mask_dice", dice_meter.avg, global_step)
                writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)
                losses_meter.reset()
                ce_meter.reset()
                bce_meter.reset()
                dice_meter.reset()
                batch_time.reset()

        # ======================== Validation (rank 0 only) ========================
        giou, ciou = 0.0, 0.0
        if val_loader is not None and is_main_process():
            giou, ciou = validate(val_loader, raw_model, args)
            writer.add_scalar("val/giou", giou, epoch)
            writer.add_scalar("val/ciou", ciou, epoch)
            print(f"  Epoch {epoch}: gIoU={giou:.4f}, cIoU={ciou:.4f}")

        if use_ddp:
            dist.barrier()

        # ======================== Save (rank 0 only) ========================
        if is_main_process():
            is_best = giou > best_giou or args.no_eval
            if is_best:
                best_giou = max(giou, best_giou)

            save_path = os.path.join(log_dir, f"epoch_{epoch}.pt")
            save_trainable_weights(raw_model, save_path)

            if is_best:
                best_path = os.path.join(log_dir, "best.pt")
                save_trainable_weights(raw_model, best_path)
                print(f"  New best! gIoU={best_giou:.4f}")

        if use_ddp:
            dist.barrier()

    if is_main_process():
        writer.close()
        print(f"\nTraining complete. Best gIoU: {best_giou:.4f}")
        print(f"Checkpoints saved to: {log_dir}")

    if use_ddp:
        dist.destroy_process_group()


def validate(val_loader, model, args):
    """Teacher-forcing validation."""
    model.eval()
    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)

    amp_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
    use_amp = args.precision in ("bf16", "fp16")

    for batch in tqdm.tqdm(val_loader, desc="Validating"):
        torch.cuda.empty_cache()
        batch = dict_to_cuda(batch)

        if args.precision == "bf16":
            batch["images"] = batch["images"].bfloat16()
            batch["images_clip"] = batch["images_clip"].bfloat16()
        elif args.precision == "fp16":
            batch["images"] = batch["images"].half()
            batch["images_clip"] = batch["images_clip"].half()

        with torch.no_grad(), autocast(enabled=use_amp, dtype=amp_dtype):
            out = model(**batch)

        pred_masks = out["pred_masks"]
        masks_list = out["gt_masks"][0].int()
        output_list = (pred_masks[0] > 0).int()

        intersection, union, acc_iou = 0.0, 0.0, 0.0
        for mask_i, output_i in zip(masks_list, output_list):
            intersection_i, union_i, _ = intersectionAndUnionGPU(
                output_i.contiguous().clone(), mask_i.contiguous(), 2,
                ignore_index=255,
            )
            intersection += intersection_i
            union += union_i
            acc_iou += intersection_i / (union_i + 1e-5)
            acc_iou[union_i == 0] += 1.0

        intersection = intersection.cpu().numpy()
        union = union.cpu().numpy()
        acc_iou = acc_iou.cpu().numpy() / masks_list.shape[0]
        intersection_meter.update(intersection)
        union_meter.update(union)
        acc_iou_meter.update(acc_iou, n=masks_list.shape[0])

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    ciou = iou_class[1]
    giou = acc_iou_meter.avg[1]

    model.train()
    return giou, ciou


if __name__ == "__main__":
    main()
