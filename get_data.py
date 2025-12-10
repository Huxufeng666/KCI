import os
import csv
import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import cv2

import torchvision.transforms as transforms


def _load_mask_from_pil(pil_img: Image.Image) -> torch.Tensor:
    """把 PIL.Image 转成和原来 _load_mask 一样的 tensor"""
    mask = np.array(pil_img)
    mask = (mask > 127).astype(np.float32)
    return torch.from_numpy(mask).unsqueeze(0)

# ---------- 辅助函数 ----------
def _load_image(path: str) -> torch.Tensor:
    try:
        img = Image.open(path).convert("L")
        return torch.from_numpy(np.array(img)).float().unsqueeze(0) / 255.0
    except Exception as e:
        raise IOError(f"无法加载图像: {path} → {e}")


def _load_mask(path: str) -> torch.Tensor:
    try:
        mask = Image.open(path).convert("L")
        mask = np.array(mask)
        mask = (mask > 127).astype(np.float32)
        return torch.from_numpy(mask).unsqueeze(0)
    except Exception as e:
        raise IOError(f"无法加载掩码: {path} → {e}")


def _create_edge(mask_np: np.ndarray, kernel_size=3) -> np.ndarray:
    grad_x = cv2.Sobel(mask_np, cv2.CV_32F, 1, 0, ksize=kernel_size)
    grad_y = cv2.Sobel(mask_np, cv2.CV_32F, 0, 1, ksize=kernel_size)
    edge = np.sqrt(grad_x**2 + grad_y**2)
    return (edge > 0).astype(np.float32)


# ---------- 主 Dataset ----------
class BUSI_Data(Dataset):
    def __init__(self,
                 root_dir: str = "dataset",
                 split: str = "train",
                 img_size: int = 256,
                 seed: int = 2025):
        assert split in {"train", "val", "test"}, f"split 必须是 train/val/test"
        self.root = Path(root_dir)
        self.split = split
        self.img_size = img_size
        random.seed(seed)

        # 1. 读取 labels.csv
        label_csv = self.root / "labels.csv"
        if not label_csv.exists():
            raise FileNotFoundError(f"labels.csv 不存在: {label_csv}")
        self.label_dict = self._load_labels(label_csv)

        # 2. 收集图像
        img_dir = self.root / split / "images"
        mask_dir = self.root / split / "masks"
        if not img_dir.exists():
            raise FileNotFoundError(f"图像目录不存在: {img_dir}")
        if not mask_dir.exists():
            raise FileNotFoundError(f"掩码目录不存在: {mask_dir}")

        self.samples = self._collect_samples(img_dir, mask_dir)
        print(f"[{split.upper()}] 加载 {len(self.samples)} 张图像")

    # ------------------- 内部工具 -------------------
    def _load_labels(self, csv_path: Path) -> Dict[str, int]:
        """支持多种列名: class_id, label, cls, category"""
        d = {}
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            print(f"[DEBUG] CSV 列名: {reader.fieldnames}")

            # 自动找 class_id 列
            class_col = None
            for col in ["class_id", "label", "cls", "category", "class"]:
                if col in reader.fieldnames:
                    class_col = col
                    break
            if class_col is None:
                raise KeyError(f"CSV 中未找到类别列! 可用: {reader.fieldnames}")

            for i, row in enumerate(reader):
                img_path = row["image_path"].strip()
                try:
                    cls_id = int(row[class_col].strip())
                except ValueError:
                    raise ValueError(f"第 {i+2} 行 class_id 不是整数: {row[class_col]}")
                d[img_path] = cls_id
        print(f"[DEBUG] 成功加载 {len(d)} 条标签")
        return d

    def _collect_samples(self, img_dir: Path, mask_dir: Path) -> List[Dict]:
        samples = []
        for img_path in img_dir.glob("*.png"):
            # img_path_str = str(img_path).replace("\\", "/")
            filename = img_path.name  # 只用文件名

            if filename not in self.label_dict:
                print(f"警告: labels.csv 中找不到: {filename}")
                continue

            # 2. 匹配 mask（支持 xxx_mask.png）
            stem = img_path.stem
            possible_masks = [
                mask_dir / f"{stem}_mask.png",
                mask_dir / f"{stem}_mask_1.png",
                mask_dir / f"{stem}.png",
                mask_dir / img_path.name
            ]
            mask_path = None
            for mp in possible_masks:
                if mp.exists():
                    mask_path = str(mp).replace("\\", "/")
                    break
            if mask_path is None:
                print(f"警告: 找不到 mask: {img_path}")
                continue

            samples.append({
                "image_path": str(img_path),
                "mask_path": mask_path,
                "class_id": self.label_dict[filename]
            })

        random.shuffle(samples)
        return samples



    # ------------------- 核心 -------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        sample = self.samples[idx]

        # img = _load_image(sample["image_path"])
        # mask = _load_mask(sample["mask_path"])
        img = _load_image(sample["image_path"])
        
        base_name = Path(sample["image_path"]).stem
        mask_dir = Path(sample["image_path"]).parent.parent / "masks"  # 自动定位 masks 文件夹

        mask1_path = mask_dir / f"{base_name}_mask.png"
        mask2_path = mask_dir / f"{base_name}_mask_1.png"

        mask1 = cv2.imread(str(mask1_path), cv2.IMREAD_GRAYSCALE) if mask1_path.exists() else None
        mask2 = cv2.imread(str(mask2_path), cv2.IMREAD_GRAYSCALE) if mask2_path.exists() else None

        if mask1 is None and mask2 is None:
            raise FileNotFoundError(f"找不到任何 mask: {base_name}_mask.png 或 {base_name}_mask_1.png")

        # 合并：只要有一个像素是前景，就当前景（取并集）
        if mask1 is not None and mask2 is not None:
            final_mask = np.maximum(mask1, mask2)
        elif mask1 is not None:
            final_mask = mask1
        else:
            final_mask = mask2

        # 转回 PIL → 后面用你原来的 _load_mask 逻辑
        final_mask_pil = Image.fromarray(final_mask)
        mask = _load_mask_from_pil(final_mask_pil)  # 下面会补这个函数
    
        

        # Resize
        resize = torch.nn.functional.interpolate
        img = resize(img.unsqueeze(0), size=(self.img_size, self.img_size),
                     mode="bilinear", align_corners=False).squeeze(0)
        mask = resize(mask.unsqueeze(0), size=(self.img_size, self.img_size),
                      mode="nearest").squeeze(0)

        # Edge
        mask_np = (mask.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        edge_np = _create_edge(mask_np,kernel_size=3)
        edge = torch.from_numpy(edge_np).unsqueeze(0).float()# / 255.0

        # Recon
        recon = img.clone()
        cls = torch.tensor(sample["class_id"], dtype=torch.long)

        return img, mask, cls, edge, recon

  # ==================================================
# 完整可运行版本（直接复制粘贴就能跑）
# ==================================================
import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision import transforms
from tqdm import tqdm
import datetime
import csv

# ==================== 1. Dataset（完美版） ====================
class MedicalDataset(Dataset):
    def __init__(self, image_dir, mask_dir, img_size=384, transform=None):
        self.image_dir = image_dir
        self.mask_dir  = mask_dir
        self.transform = transform
        self.img_size  = (img_size, img_size) if isinstance(img_size, int) else img_size

        self.image_list = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        if len(self.image_list) == 0:
            raise ValueError(f"No images in {image_dir}")

        # 固定 resize（必须在 ToTensor 之前）
        self.resize_img  = T.Resize(self.img_size, interpolation=T.InterpolationMode.BILINEAR)
        self.resize_mask = T.Resize(self.img_size, interpolation=T.InterpolationMode.NEAREST)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("L")  # (H, W)

        # === 合并双 mask ===
        base_name, _ = os.path.splitext(img_name)
        mask_paths = [
            os.path.join(self.mask_dir, f"{base_name}_mask.png"),
            os.path.join(self.mask_dir, f"{base_name}_mask_1.png")
        ]
        mask_np = np.zeros((image.height, image.width), dtype=np.uint8)
        found = False
        for p in mask_paths:
            if os.path.exists(p):
                m = np.array(Image.open(p).convert("L"))
                mask_np = np.bitwise_or(mask_np, m)
                found = True
        if not found:
            raise FileNotFoundError(f"Mask not found: {img_name}")
        mask = Image.fromarray(mask_np)

        # === 统一 Resize（关键！）===
        image = self.resize_img(image)
        mask  = self.resize_mask(mask)

        # === 转为 Tensor（只此一次！）===
        image = T.ToTensor()(image)      # (1, 384, 384) float32 [0,1]
        mask  = T.ToTensor()(mask)       # (1, 384, 384) float32 [0,1]
        mask  = (mask > 0.5).float()     # 严格 0/1

        # === 额外增强（只对 Tensor 有效，且 image/mask 同步）===
        if self.transform is not None:
            # 固定随机种子，保证 image 和 mask 增强一致
            state = torch.get_rng_state()
            image = self.transform(image)
            torch.set_rng_state(state)
            mask = self.transform(mask)  # 注意：这里不能有 Normalize！

        return image, mask.squeeze(0)  # image: [1,384,384], mask: [384,384]