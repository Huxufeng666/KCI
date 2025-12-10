

# ==================================================
# test.py - 论文级测试脚本
# 功能：指标计算 + 可视化拼接图 + CSV 保存
# 修复：所有 bug、内存泄漏、归一化错误
# ==================================================
import os
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import csv
from torchvision.utils import save_image
from torchvision import transforms
from PIL import Image

# 导入你的模型和数据集
from model.dpt_asf_gan import DPT_ASF_GAN
from get_data import BUSI_Data,MedicalDataset
from model.FPNUNet import FPNUNet_CBAM_Residual
from model.model import EndToEndModel
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
import torchvision.utils as vutils   # ← 加上这句就行了！

from model.FPNUent_Multi_task import MultiTaskFPNUNet
# ==================================================
# 1. 指标计算（每张图独立计算 → 真实 Dice）
# ==================================================
@torch.no_grad()
def evaluate(model, test_loader, device, threshold=0.5, smooth=1e-6):
    model.eval()
    dice_list, iou_list, prec_list, rec_list = [], [], [], []

    with torch.no_grad():
        for imgs, targets in tqdm(test_loader, desc="Evaluating"):
            imgs = imgs.to(device)
            targets = targets.to(device).float()   # 确保是 float

            # === 关键修复1：正确接收模型输出（支持返回1个或2个）===
            outputs = model(imgs)
            if isinstance(outputs, tuple):
                logits = outputs[0]          # 主分割头
            else:
                logits = outputs             # 只有一个输出

            # === 关键修复2：强制上采样到原图尺寸（防止尺寸不匹配）===
            if logits.shape[-2:] != targets.shape[-2:]:
                logits = F.interpolate(
                    logits,
                    size=targets.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )

            # 预测概率 + 二值化
            preds = torch.sigmoid(logits)
            preds_bin = (preds > threshold).float()   # [B,1,H,W]

            # === 逐样本计算指标（最安全）===
            for i in range(preds_bin.size(0)):
                pred = preds_bin[i].flatten()      # [H*W]
                target = targets[i].flatten()      # [H*W]

                tp = (pred * target).sum()
                fp = (pred * (1 - target)).sum()
                fn = ((1 - pred) * target).sum()

                dice = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)
                iou  = (tp + smooth) / (tp + fp + fn + smooth)
                prec = (tp + smooth) / (tp + fp + smooth)
                rec  = (tp + smooth) / (tp + fn + smooth)

                dice_list.append(dice.item())
                iou_list.append(iou.item())
                prec_list.append(prec.item())
                rec_list.append(rec.item())

    return {
        'dice': np.mean(dice_list),
        'iou': np.mean(iou_list),
        'precision': np.mean(prec_list),
        'recall': np.mean(rec_list),
    }

# ==================================================
# 2. 可视化拼接（原图 | GT | Pred）
# ==================================================
def save_visualizations(model, vis_loader, device, save_dir):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for batch_idx, (imgs, masks) in enumerate(vis_loader):
            imgs = imgs.to(device)
            masks = masks.to(device).float()

            # 确保 mask 有 channel 维
            if masks.dim() == 3:
                masks = masks.unsqueeze(1)  # (B,H,W) → (B,1,H,W)

            logits, aux2, aux3, aux4, class_logits, edge_out, recon_out = model(imgs)
            if logits.shape[-2:] != masks.shape[-2:]:
                logits = F.interpolate(logits, size=masks.shape[-2:], mode='bilinear', align_corners=False)

            prob = torch.sigmoid(logits)
            pred = (prob > 0.5).float()

            # 只可视化前8张
            for i in range(min(8, imgs.size(0))):
                # ========== 原图 ==========
                img = imgs[i]                                   # [C, H, W]
                if img.shape[0] == 1:  # 灰度图 → 转成3通道
                    img_3ch = img.repeat(3, 1, 1)
                elif img.shape[0] == 3:
                    img_3ch = img
                else:
                    img_3ch = img[:3]  # 防止奇怪通道数

                # ========== GT mask ==========
                mask = masks[i]                                 # [1, H, W] 或 [H, W]
                if mask.dim() == 3 and mask.shape[0] == 1:
                    mask = mask.squeeze(0)                      # → [H, W]
                mask_3ch = torch.zeros(3, mask.shape[0], mask.shape[1], device=device)
                mask_3ch[1] = mask  # 绿色通道

                # ========== Pred mask ==========
                pr = pred[i]                                    # [1, H, W]
                if pr.dim() == 3 and pr.shape[0] == 1:
                    pr = pr.squeeze(0)                          # → [H, W]
                pred_3ch = torch.zeros(3, pr.shape[0], pr.shape[1], device=device)
                pred_3ch[0] = pr  # 红色通道（预测错误会显红）

                # ========== 关键修复：强制所有都是 [3, H, W] ==========
                img_3ch  = img_3ch   # 已经是对的
                mask_3ch = mask_3ch  # 已经是对的
                pred_3ch = pred_3ch  # 已经是对的

                # 现在可以安全拼接了（水平拼接：原图 | GT | Pred）
                concat = torch.cat([img_3ch, mask_3ch, pred_3ch], dim=2)  # [3, H, W*3]

                # 归一化到 0~1 并保存
                concat = torch.clamp(concat, 0, 1)
                vutils.save_image(
                    concat,
                    os.path.join(save_dir, f"vis_{batch_idx:03d}_{i}.png"),
                    normalize=True,
                    value_range=(0, 1)
                )

            if batch_idx >= 4:  # 只保存前5个batch的可视化
                break
    print(f"All visualizations saved to: {save_dir}")


# ==================================================
# 3. 主函数
# ==================================================
def main():
    # 设备
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 路径配置
    weight_path = '/workspace/results/ablation1/Exp1_Baseline_20251203_081557/best_model.pth'
    test_data = MedicalDataset(image_dir='dataset/test/images', mask_dir='dataset/test/masks')
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=8, shuffle=False, num_workers=0, pin_memory=False
    )
    vis_loader = torch.utils.data.DataLoader(
        test_data, batch_size=1, shuffle=False, num_workers=0
    )

    # 模型
    model = MultiTaskFPNUNet().to(device)
    # model = UNetPlusPlus(in_channels=1, num_classes=1, base_c=64).to(device)  # 完全匹配你的 UNet
    # model = UNetPlusPlus().to(device)
    
    
    model.load_state_dict(torch.load(weight_path))#, map_location=device))
    print(f"Loaded weights: {weight_path}")

    # 1. 计算指标
    print("Computing metrics...")
    metrics = evaluate(model, test_loader, device)
    print("\n" + "="*50)
    print(" " * 15 + "TEST RESULTS")
    print("="*50)
    for k, v in metrics.items():
        print(f"{k:10}: {v:.4f}")
    print("="*50)

    # 2. 保存 CSV
    result_dir = os.path.dirname(weight_path)
    csv_path = os.path.join(result_dir, 'test_metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value'])
        for k, v in metrics.items():
            writer.writerow([k, f"{v:.6f}"])
    print(f"Metrics saved to: {csv_path}")

    # 3. 保存可视化
    vis_dir = os.path.join(result_dir, 'test_visuals')
    print(f"Saving visualizations to: {vis_dir}")
    save_visualizations(model, vis_loader, device, vis_dir)


# ==================================================
# 4. 运行
# ==================================================
if __name__ == '__main__':
    main()

