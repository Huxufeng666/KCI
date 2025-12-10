# train_unet_fixed.py
import os
import random
import csv
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt
from Early_Stopp import EarlyStopping

# 导入你的数据集和模型
from get_data import MedicalDataset
from model.U_net import UNet  # 你刚才贴的那个 UNet
# ------------------------------------------------
from  get_data  import  BUSI_Data ,MedicalDataset
from model.model import EndToEndModel
from model.ConvNeXt_Small_FPN_CBAM import EndToEndModel2
from model.dpt_asf_gan import DPT_ASF_GAN
from tools import plot_loss_curve
from model.FPNUNet import FPNUNet_CBAM_Residual
from model.Swin_unet import SwinUnet
from model.AAUnet import AAUNet
from model.U_net_plus import UNetPlusPlus
from scipy.ndimage import distance_transform_edt

# ==================== 1. 修复点总览 ====================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==================== 数据加载 ====================
train_data = MedicalDataset('dataset/train/images', 'dataset/train/masks')
val_data   = MedicalDataset('dataset/val/images',   'dataset/val/masks')

batch_size = 8
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                          num_workers=0, pin_memory=True, drop_last=True)
val_loader   = DataLoader(val_data,   batch_size=batch_size, shuffle=False,
                          num_workers=0, pin_memory=True)

# ==================== 模型 ====================
# model = AAUNet(in_channels=1, num_classes=1, base_c=64).to(device)  # 完全匹配你的 UNet
model = AAUNet().to(device)  # 完全匹配你的 UNet
model_name = "AAUNet"
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = f"results/{model_name}_{timestamp}"
os.makedirs(log_dir, exist_ok=True)

# ==================== 损失函数（推荐组合）===================
bce_fn = nn.BCEWithLogitsLoss()

def dice_loss(pred_logits, targets, smooth=1e-6):
    probs = torch.sigmoid(pred_logits)
    B = probs.shape[0]
    probs  = probs.view(B, -1)
    targets = targets.view(B, -1)
    inter = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2. * inter + smooth) / (union + smooth)
    return 1 - dice.mean()

# Boundary Loss（训练时用，验证时跳过，防止太慢）
def boundary_loss(pred_logits, targets):
    probs = torch.sigmoid(pred_logits)
    B, _, H, W = targets.shape
    boundary_targets = []
    for i in range(B):
        mask = targets[i, 0].cpu().numpy()
        if mask.sum() == 0:
            dist = np.zeros_like(mask)
        else:
            dist = distance_transform_edt(1 - mask) + distance_transform_edt(mask)
            dist = dist / (dist.max() + 1e-6)
        boundary_targets.append(dist)
    boundary_targets = torch.tensor(np.stack(boundary_targets), dtype=torch.float32, device=device).unsqueeze(1)
    return F.mse_loss(probs, boundary_targets)

# ==================== 优化器 & 调度器 ====================
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
early_stopping = EarlyStopping(patience=30, min_delta=1e-5)

# ==================== 日志 ====================
log_csv = os.path.join(log_dir, "log.csv")
with open(log_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "train_loss", "val_dice", "val_loss", "lr"])

# ==================== 早停类（原样保留）===================
class EarlyStopping:
    def __init__(self, patience=20, min_delta=1e-6, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        self.early_stop = False

    def step(self, val_loss, model):
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_weights = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights and self.best_weights:
                    model.load_state_dict(self.best_weights)
                    print("Restored best weights!")

# ==================== 训练循环（完全修复版）===================
num_epochs = 200
best_val_dice = 0.0

for epoch in range(1, num_epochs + 1):
    model.train()
    train_loss_total = 0.0
    
    for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
        imgs  = imgs.to(device)      # (B,1,H,W) 或 (B,3,H,W) 取决于你的数据集
        masks = masks.to(device)     # 必须是 float32，且有 channel 维
        
        # 关键修复：确保 masks 是 (B,1,H,W) 的 float
        if masks.dim() == 3:
            masks = masks.unsqueeze(1)          # (B,H,W) → (B,1,H,W)
        masks = masks.float()

        # 前向
        logits = model(imgs)                    # ← 现在只返回一个 tensor
        if logits.shape[-2:] != masks.shape[-2:]:
            logits = F.interpolate(logits, size=masks.shape[-2:], mode='bilinear', align_corners=False)

        # 损失
        loss_bce  = bce_fn(logits, masks)
        loss_dice = dice_loss(logits, masks)
        loss_bd   = boundary_loss(logits, masks)
        loss = 0.5 * loss_bce + 0.5 * loss_dice + 0.3 * loss_bd   # boundary 权重可以小一点

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss_total += loss.item()

    # ==================== 验证 ====================
    model.eval()
    val_loss_total = 0.0
    val_dice_total = 0.0
    with torch.no_grad():
        for imgs, masks in tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):
            imgs  = imgs.to(device)
            masks = masks.to(device)
            if masks.dim() == 3:
                masks = masks.unsqueeze(1)
            masks = masks.float()

            logits = model(imgs)
            if logits.shape[-2:] != masks.shape[-2:]:
                logits = F.interpolate(logits, size=masks.shape[-2:], mode='bilinear', align_corners=False)

            loss_bce  = bce_fn(logits, masks)
            loss_dice = dice_loss(logits, masks)
            loss = 0.5 * loss_bce + 0.5 * loss_dice

            probs = torch.sigmoid(logits)
            dice_score = (2. * (probs * masks).sum() + 1e-6) / (probs.sum() + masks.sum() + 1e-6)
            val_dice_total += dice_score.item()
            val_loss_total += loss.item()

    avg_train_loss = train_loss_total / len(train_loader)
    avg_val_loss   = val_loss_total / len(val_loader)
    avg_val_dice   = val_dice_total / len(val_loader)

    print(f"Epoch {epoch:03d} | TrainLoss {avg_train_loss:.4f} | ValLoss {avg_val_loss:.4f} | ValDice {avg_val_dice:.4f}")

    # 日志
    with open(log_csv, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch, f"{avg_train_loss:.6f}", f"{avg_val_dice:.6f}", f"{avg_val_loss:.6f}", optimizer.param_groups[0]['lr']])

    # 保存最佳模型（按 Dice 最高保存）
    if avg_val_dice > best_val_dice:
        best_val_dice = avg_val_dice
        torch.save(model.state_dict(), os.path.join(log_dir, "best_model.pth"))
        print(f" New best Dice: {best_val_dice:.4f}  Model saved!")

    # 每 10 轮保存一次 checkpoint
    if epoch % 10 == 0:
        torch.save(model.state_dict(), os.path.join(log_dir, f"epoch{epoch}_dice{avg_val_dice:.4f}.pth"))

    scheduler.step(avg_val_loss)
    early_stopping.step(avg_val_loss, model,epoch)
    if early_stopping.early_stop:
        print("Early stopping!")
        break

    # ==================== 可视化 ====================
    if epoch % 10 == 0:
        with torch.no_grad():
            val_iter = iter(val_loader)
            imgs, masks = next(val_iter)
            imgs = imgs.to(device)
            if masks.dim() == 3:
                masks = masks.unsqueeze(1)
            masks = masks.float().to(device)

            pred_logits = model(imgs)
            pred_logits = F.interpolate(pred_logits, size=masks.shape[-2:], mode='bilinear', align_corners=False)
            pred_prob   = torch.sigmoid(pred_logits)
            pred_mask   = (pred_prob > 0.5).float()

            # 拼接：原图 + GT + Pred
            vis = torch.cat([imgs[:8], masks[:8], pred_mask[:8]], dim=3)  # 水平拼接
            vutils.save_image(
            vis.cpu(),
            os.path.join(log_dir, f"vis_epoch{epoch}.png"),
            nrow=1,
            normalize=True,
            value_range=(0, 1))      # 改这里就行)
            # vutils.save_image(vis.cpu(), os.path.join(log_dir, f"vis_epoch{epoch}.png"),
            #                   nrow=1, normalize=True, range=(0,1))

print("训练结束！最佳 Dice:", best_val_dice)
print("结果保存在:", log_dir)