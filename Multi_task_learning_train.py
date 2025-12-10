
import os
os.environ["MPLBACKEND"] = "Agg"
import random, datetime, csv
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from scipy.ndimage import distance_transform_edt
from model.ConvNeXt_Small_FPN_CBAM import EndToEndModel2

# ------------------ 你的模块 ------------------
from get_data import BUSI_Data  #, BUSI_Data
from model.FPNUent_Multi_task import MultiTaskFPNUNet
from tools import plot_loss_curve
from utils.tools import visualize_batch

# ==================== 损失函数 ====================
class BCEDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        prob = torch.sigmoid(logits).flatten(1)
        targets = targets.flatten(1)
        inter = (prob * targets).sum(1)
        union = prob.sum(1) + targets.sum(1)
        dice = (2. * inter + self.smooth) / (union + self.smooth)
        return bce_loss + (1 - dice.mean())

def boundary_loss(logits, masks):
    prob = torch.sigmoid(logits)
    B, _, H, W = prob.shape
    mask_np = (masks.squeeze(1).cpu().numpy() > 0.5).astype(np.uint8)
    dist_maps = np.zeros_like(mask_np, dtype=np.float32)
    for i in range(B):
        if mask_np[i].sum() == 0: continue
        dist = distance_transform_edt(1 - mask_np[i])
        if dist.max() > 0: dist /= dist.max()
        dist_maps[i] = dist
    dist_tensor = torch.from_numpy(dist_maps).unsqueeze(1).to(logits.device)
    weight = (masks >= 0.5).float() + 0.1
    return F.mse_loss(prob * weight, dist_tensor * weight)

def dice_coeff(logits, targets, smooth=1e-6):
    prob = torch.sigmoid(logits).flatten(1)
    targets = targets.flatten(1)
    inter = (prob * targets).sum(1)
    union = prob.sum(1) + targets.sum(1)
    return ((2. * inter + smooth) / (union + smooth)).mean()

# ==================== 全局损失实例 ====================
seg_criterion = BCEDiceLoss()
ce_cls  = nn.CrossEntropyLoss()
bce_edge = nn.BCEWithLogitsLoss()
l1_rec  = nn.L1Loss()

# ==================== 设置 ====================
def set_seed(seed=2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
    # os.environ['PYTHONHASHSEED'] = str(seed)
set_seed(2025)

IMG_SIZE = 256
BATCH_SIZE = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_set = BUSI_Data(root_dir="/workspace/dataset", split="train", img_size=IMG_SIZE)
val_set   = BUSI_Data(root_dir="/workspace/dataset", split="val",   img_size=IMG_SIZE)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

model = MultiTaskFPNUNet(in_ch=1, seg_ch=1, num_classes=3).to(device)


model = nn.DataParallel(model)
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)#, verbose=True)


# ==================== 早停 ====================
class EarlyStopping:
    def __init__(self, patience=25):
        self.patience = patience
        self.best = None
        self.cnt = 0
        self.stop = False
        self.best_state = None
    def step(self, val_loss, model):
        if self.best is None or val_loss < self.best - 1e-6:
            self.best = val_loss
            self.cnt = 0
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.cnt += 1
            if self.cnt >= self.patience:
                model.load_state_dict(self.best_state)
                self.stop = True
        return self.stop
early_stop = EarlyStopping(patience=25)

# ==================== 日志 ====================
ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = f"results/stage_wise_{ts}"
os.makedirs(log_dir, exist_ok=True)
log_csv = os.path.join(log_dir, "log.csv")
with open(log_csv, "w", newline="") as f:
    csv.writer(f).writerow(["epoch","stage","train_loss","val_loss","val_dice","high_conf_dice","cls_acc","lr"])

# ==================== 训练主循环（精准分阶段）===================
EPOCHS = 300
WARMUP_EPOCHS = 5          # 前20轮：只训 分类 + 边界 + 重建
SEG_START_EPOCH = 6        # 第21轮开始加分割
CONF_THRESH = 0.95          # 分割只在置信度 ≥95% 的样本上计算
best_val_loss = float('inf')

for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    train_dice_list = []

    for img, mask, cls_label, edge_gt, recon_gt in tqdm(train_loader, desc=f"Epoch {epoch:03d}", leave=False):
        img = img.to(device)
        mask = mask.to(device).float()
        cls_label = cls_label.to(device).long()
        edge_gt = edge_gt.to(device).float()
        recon_gt = recon_gt.to(device)

        optimizer.zero_grad()
        seg_main, aux2, aux3, aux4, cls_logit, edge_out, recon_out = model(img)

        # ==================== 阶段1 & 2 公共损失 ====================
        loss_cls = ce_cls(cls_logit, cls_label)
        loss_edge = bce_edge(edge_out, edge_gt)
        loss_rec = 0.02 * l1_rec(recon_out, recon_gt)
        total_loss = loss_cls + loss_edge + loss_rec

        # ==================== 阶段2：置信度引导的分割（第21轮开始）===================
        if epoch >= SEG_START_EPOCH:
            with torch.no_grad():
                prob = F.softmax(cls_logit, dim=1)
                confidence, _ = torch.max(prob, dim=1)
                high_conf_mask = (confidence >= CONF_THRESH).float()
                tumor_mask = (confidence != 0).float()
                high_conf_tumor = (confidence != 0) & (confidence >= CONF_THRESH)
                final_mask = high_conf_tumor.float()            # 最终决定是否训分割

            if high_conf_mask.sum() > 0:
                H, W = mask.shape[-2:]
                seg_main_up = F.interpolate(seg_main, size=(H,W), mode='bilinear', align_corners=False)
                aux2_up = F.interpolate(aux2, size=(H,W), mode='bilinear', align_corners=False)
                aux3_up = F.interpolate(aux3, size=(H,W), mode='bilinear', align_corners=False)
                aux4_up = F.interpolate(aux4, size=(H,W), mode='bilinear', align_corners=False)

                loss_seg = seg_criterion(seg_main_up, mask)
                loss_aux = 0.8 * seg_criterion(aux4_up, mask) + \
                           0.6 * seg_criterion(aux3_up, mask) + \
                           0.4 * seg_criterion(aux2_up, mask)
                # loss_boundary = 0.6 * boundary_loss(seg_main_up, mask)

                seg_loss_total = (loss_seg + loss_aux ) * high_conf_mask.mean()
                total_loss = total_loss + seg_loss_total

                train_dice_list.append(dice_coeff(seg_main_up, mask).item())

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss += total_loss.item()

    # ==================== 验证 ====================
    model.eval()
    val_losses = []
    dice_tumor_list = []
    cls_correct = 0

    with torch.no_grad():
        for img, mask, cls_gt, edge_gt, recon_gt in val_loader:
            img = img.to(device); mask = mask.to(device).float(); cls_gt = cls_gt.to(device).long()
            edge_gt = edge_gt.to(device).float(); recon_gt = recon_gt.to(device)

            seg_main, _, _, _, cls_logit, edge_out, recon_out = model(img)
            H, W = img.shape[-2:]
            seg_main = F.interpolate(seg_main, size=(H,W), mode='bilinear', align_corners=False)
            edge_out = F.interpolate(edge_out, size=(H,W), mode='bilinear', align_corners=False)

            # 分类/边缘/重建 永远算
            loss_cls = ce_cls(cls_logit, cls_gt)
            loss_edge = bce_edge(edge_out, edge_gt)
            loss_rec = l1_rec(recon_out, recon_gt)

            # 分割损失：只在“预测为肿瘤且高置信”的样本上算（和训练一致！）
            prob = F.softmax(cls_logit, dim=1)
            conf, pred = torch.max(prob, dim=1)
            should_seg = (pred != 0) & (conf >= 0.95)

            if should_seg.sum() > 0:
                loss_seg = seg_criterion(seg_main[should_seg], mask[should_seg])
            else:
                loss_seg = torch.tensor(0.0, device=device)

            val_losses.append((loss_cls + loss_edge + 0.02*loss_rec + loss_seg).item())
            cls_correct += (pred == cls_gt).sum().item()

            # 核心指标：真实有肿瘤样本的 Dice
            real_tumor = (cls_gt != 0)
            if real_tumor.sum() > 0:
                dice_tumor_list.append(dice_coeff(seg_main[real_tumor], mask[real_tumor]).item())

    avg_val_loss = np.mean(val_losses)
    dice_tumor = np.mean(dice_tumor_list) if dice_tumor_list else 0.0
    print(f"→ 真实肿瘤样本 Dice: {dice_tumor:.5f} ↑↑↑")
    # 日志 + 保存 + 可视化（同之前完美版）
    # ==================== 验证 & 早停 & 保存 & 可视化（终极无错版）===================
    model.eval()
    val_losses = []
    val_dice_all = []
    val_dice_high = []
    cls_correct = 0
    total_samples = 0

    with torch.no_grad():
        for img, mask, cls_gt, edge_gt, recon_gt in val_loader:
            img = img.to(device)
            mask = mask.to(device).float()
            cls_gt = cls_gt.to(device).long()

            seg_main, _, _, _, cls_logit, edge_out, recon_out = model(img)

            # 上采样
            H, W = img.shape[-2:]
            seg_main = F.interpolate(seg_main, size=(H, W), mode='bilinear', align_corners=False)

            # 损失（统一计算方式）
            loss_seg = BCEDiceLoss()(seg_main, mask)
            loss_cls = ce_cls(cls_logit, cls_gt)
            loss_rec = l1_rec(recon_out, img)
            total_val_loss_batch = loss_seg + 0.3 * loss_cls + 0.02 * loss_rec

            val_losses.append(total_val_loss_batch.item())
            val_dice_all.append(dice_coeff(seg_main, mask).item())
            cls_correct += (cls_logit.argmax(1) == cls_gt).sum().item()
            total_samples += cls_gt.size(0)

            # 高置信样本的 Dice（用于监控）
            prob = F.softmax(cls_logit, dim=1)
            conf, _ = torch.max(prob, dim=1)
            high_idx = conf >= 0.95
            if high_idx.sum() > 0:
                val_dice_high.append(dice_coeff(seg_main[high_idx], mask[high_idx]).item())

    # ==================== 计算平均指标（关键！）===================
    avg_val_loss = np.mean(val_losses)
    avg_dice = np.mean(val_dice_all)
    high_dice = np.mean(val_dice_high) if val_dice_high else 0.0
    cls_acc = cls_correct / total_samples
    stage = "Joint"  # 你现在全程联合训练，写死也行

    # ==================== 日志写入（永不空表）===================
    try:
        current_lr = float(optimizer.param_groups[0]['lr'])
    except:
        current_lr = 1e-4

    log_row = [
        epoch,
        stage,
        f"{epoch_loss/len(train_loader):.6f}",
        f"{avg_val_loss:.6f}",
        f"{avg_dice:.6f}",
        f"{high_dice:.6f}",
        f"{cls_acc:.4f}",
        f"{current_lr:.2e}"
    ]

    # 确保文件存在并写表头（只执行一次）
    if not os.path.exists(log_csv):
        with open(log_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "stage", "train_loss", "val_loss", "val_dice", "high_conf_dice", "cls_acc", "lr"])

    # 追加本轮日志
    with open(log_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(log_row)

    # ==================== 保存最佳模型 ====================
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        save_dict = {
            'epoch': epoch,
            'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': avg_val_loss,
            'high_conf_dice': high_dice,
        }
        torch.save(save_dict, os.path.join(log_dir, "best_model.pth"))
        print(f" NEW BEST! Epoch {epoch} | ValLoss={avg_val_loss:.6f} | HighConfDice={high_dice:.5f}")

    # ==================== 早停 + 学习率调度 ====================
    if early_stop.step(avg_val_loss, model):
        print(f"\nEarly Stop 触发于 Epoch {epoch}！")
        break

    scheduler.step(avg_val_loss)

    # ==================== 可视化（每10轮）===================
    if epoch % 10 == 0 or epoch <= 5:
        model.eval()
        with torch.no_grad():
            try:
                sample = next(iter(val_loader))
                img, mask, cls_gt, edge_gt, recon_gt = (x.to(device) for x in sample)
                seg_main, _, _, _, cls_logit, edge_out, recon_out = model(img)

                H, W = img.shape[-2:]
                seg_main = F.interpolate(seg_main, size=(H,W), mode='bilinear', align_corners=False)
                edge_out = F.interpolate(edge_out, size=(H,W), mode='bilinear', align_corners=False)
                recon_out = F.interpolate(recon_out, size=(H,W), mode='bilinear', align_corners=False)

                for i in range(min(4, img.size(0))):
                    save_path = os.path.join(log_dir, f"vis_epoch{epoch:03d}_sample{i}.png")
                    visualize_batch(
                        img=img[i:i+1], mask=mask[i:i+1], cls_gt=cls_gt[i:i+1],
                        edge=edge_gt[i:i+1], recon=recon_gt[i:i+1],
                        seg_main=seg_main[i:i+1],
                        cls_logit=cls_logit[i].argmax().item(),
                        edge_out=edge_out[i:i+1], recon_out=recon_out[i:i+1],
                        class_names=['Normal', 'Benign', 'Malignant'],
                        save_path=save_path
                    )
                print(f" Visualization saved → {log_dir}")
            except Exception as e:
                print(f"可视化出错（跳过）: {e}")

        # 画 loss 曲线
        try:
            plot_loss_curve(log_csv, os.path.join(log_dir, "loss_curve.png"))
            print(f" Loss curve updated")
        except Exception as e:
            print(f"画图失败: {e}")

print(f"\n训练完成！全部结果保存在：{log_dir}")
print(f"最佳模型：{log_dir}/best_model.pth")