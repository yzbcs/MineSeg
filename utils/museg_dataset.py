"""
MUSeg Dataset for LISA

直接从 MUSeg 原始目录加载数据:
  - Image/{stem}.jpg → RGB 原图
  - Label/{stem}_label.png → 语义标签 (uint8, 值 0-15)

训练时: 根据采样策略选择类别 (fixed-1 / fixed-3 / random-1-3 / all)，提取二值 mask
验证时: 遍历每张图的每个类别, 逐一评估
"""

import json
import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor

from model.llava import conversation as conversation_lib
from model.segment_anything.utils.transforms import ResizeLongestSide

from .utils import ANSWER_LIST, DEFAULT_IMAGE_TOKEN


CLASS_NAMES = {
    1: "person", 2: "cable", 3: "tube", 4: "indicator",
    5: "electrical equipment", 6: "electronic equipment",
    7: "mining equipment", 8: "rail area", 9: "support equipment",
    10: "door", 11: "tools and materials", 12: "rescue equipment",
    13: "container", 14: "metal fixture", 15: "anchoring equipment",
}

SHORT_QUESTION_LIST = [
    DEFAULT_IMAGE_TOKEN + "\n" + "Can you segment the {class_name} in this image?",
    DEFAULT_IMAGE_TOKEN + "\n" + "Please segment the {class_name} in this image.",
    DEFAULT_IMAGE_TOKEN + "\n" + "What is {class_name} in this image? Please respond with segmentation mask.",
    DEFAULT_IMAGE_TOKEN + "\n" + "What is {class_name} in this image? Please output segmentation mask.",
]


class MUSegDataset(torch.utils.data.Dataset):
    """MUSeg 训练数据集: 支持 fixed-1 / fixed-3 / random-1-3 / all 采样策略"""

    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        samples_per_epoch=500 * 8 * 2 * 10,
        precision: str = "fp32",
        image_size: int = 224,
        num_classes_per_sample: int = 3,
        exclude_val=False,
        museg_data="museg|train",
        sample_strategy="random-1-3",
    ):
        self.samples_per_epoch = samples_per_epoch
        self.num_classes_per_sample = num_classes_per_sample
        self.sample_strategy = sample_strategy
        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)
        self.short_question_list = SHORT_QUESTION_LIST
        self.answer_list = ANSWER_LIST

        ds_name, split = museg_data.split("|")
        index_path = os.path.join(base_image_dir, ds_name, f"{split}.json")
        with open(index_path, "r", encoding="utf-8") as f:
            self.samples = json.load(f)

        print(f"MUSeg train [{museg_data}]: {len(self.samples)} images loaded")

    def __len__(self):
        return self.samples_per_epoch

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.pixel_mean) / self.pixel_std
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, idx):
        idx = random.randint(0, len(self.samples) - 1)
        sample = self.samples[idx]
        image_path = sample["image_path"]
        label_path = sample["label_path"]
        class_ids = sample["class_ids"]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
        ori_size = image.shape[:2]

        image_clip = self.clip_image_processor.preprocess(
            image, return_tensors="pt"
        )["pixel_values"][0]

        # 根据采样策略选择类别
        if self.sample_strategy == "all":
            sampled_ids = class_ids
        elif self.sample_strategy == "random-1-3":
            n = random.randint(1, min(3, len(class_ids)))
            sampled_ids = random.sample(class_ids, n)
        elif self.sample_strategy == "fixed-1":
            sampled_ids = random.sample(class_ids, min(1, len(class_ids)))
        elif self.sample_strategy == "fixed-3":
            if len(class_ids) >= 3:
                sampled_ids = random.sample(class_ids, 3)
            else:
                sampled_ids = class_ids
        else:
            if len(class_ids) >= self.num_classes_per_sample:
                sampled_ids = random.sample(class_ids, self.num_classes_per_sample)
            else:
                sampled_ids = class_ids

        sampled_sents = [CLASS_NAMES[cid] for cid in sampled_ids]
        sampled_masks = [(label == cid).astype(np.float32) for cid in sampled_ids]

        image = self.transform.apply_image(image)
        resize = image.shape[:2]

        questions = []
        answers = []
        for class_name in sampled_sents:
            q = random.choice(self.short_question_list).format(
                class_name=class_name
            )
            questions.append(q)
            answers.append(random.choice(self.answer_list))

        conversations = []
        conv = conversation_lib.default_conversation.copy()
        for i in range(len(questions)):
            conv.messages = []
            conv.append_message(conv.roles[0], questions[i])
            conv.append_message(conv.roles[1], answers[i])
            conversations.append(conv.get_prompt())

        image = self.preprocess(
            torch.from_numpy(image).permute(2, 0, 1).contiguous()
        )

        masks = np.stack(sampled_masks, axis=0)
        masks = torch.from_numpy(masks)
        label_out = torch.ones(ori_size) * self.ignore_label

        return (
            image_path,
            image,
            image_clip,
            conversations,
            masks,
            label_out,
            resize,
            questions,
            sampled_sents,
        )


class MUSegValDataset(torch.utils.data.Dataset):
    """MUSeg 验证/测试数据集: 展开每张图的每个类别为独立样本"""

    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        val_dataset="museg|val",
        image_size=1024,
    ):
        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

        ds_name, split = val_dataset.split("|")
        index_path = os.path.join(base_image_dir, ds_name, f"{split}.json")
        with open(index_path, "r", encoding="utf-8") as f:
            images = json.load(f)

        # 展开: 每张图 x 每个类 → 一条样本
        self.eval_samples = []
        for img_info in images:
            for cid in img_info["class_ids"]:
                self.eval_samples.append({
                    "image_path": img_info["image_path"],
                    "label_path": img_info["label_path"],
                    "class_id": cid,
                    "class_name": CLASS_NAMES[cid],
                })

        print(f"MUSeg val [{val_dataset}]: {len(images)} images, "
              f"{len(self.eval_samples)} (image, class) pairs")

    def __len__(self):
        return len(self.eval_samples)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.pixel_mean) / self.pixel_std
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, idx):
        s = self.eval_samples[idx]
        image_path = s["image_path"]
        label_path = s["label_path"]
        class_id = s["class_id"]
        class_name = s["class_name"]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)

        binary_mask = (label == class_id).astype(np.float32)

        conv = conversation_lib.default_conversation.copy()
        conv.messages = []
        conv.append_message(
            conv.roles[0],
            DEFAULT_IMAGE_TOKEN
            + "\nWhat is {} in this image? Please output segmentation mask.".format(
                class_name
            ),
        )
        conv.append_message(conv.roles[1], "[SEG].")
        conversations = [conv.get_prompt()]

        image_clip = self.clip_image_processor.preprocess(
            image, return_tensors="pt"
        )["pixel_values"][0]

        image = self.transform.apply_image(image)
        resize = image.shape[:2]
        image = self.preprocess(
            torch.from_numpy(image).permute(2, 0, 1).contiguous()
        )

        masks = torch.from_numpy(binary_mask).unsqueeze(0)  # [1, H, W]
        labels = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label
        inference = True

        return (
            image_path,
            image,
            image_clip,
            conversations,
            masks,
            labels,
            resize,
            None,
            None,
            inference,
        )
