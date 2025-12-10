
import os
import random
import numpy as np
import datetime
import csv
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

# import csv
from  get_data  import  BUSI_Data 
from model.model import EndToEndModel
from model.ConvNeXt_Small_FPN_CBAM import EndToEndModel2
from model.dpt_asf_gan import DPT_ASF_GAN
from tools import plot_loss_curve
from model.FPNUNet import FPNUNet_CBAM_Residual
from model.Swin_unet import SwinUnet
from model.AAUnet import AAUNet
from model.U_net_plus import UNetPlusPlus
from scipy.ndimage import distance_transform_edt



from model.U_net import UNet



# ==================================================
# 5.1 早停机制类
# ==================================================
class EarlyStopping:
    """
    早停机制：当验证损失不再下降时，停止训练
    """
    def __init__(self, patience=10, min_delta=1e-6, restore_best_weights=True):
        """
        Args:
            patience (int): 验证损失未改善时，等待的轮数
            min_delta (float): 最小改善阈值
            restore_best_weights (bool): 是否恢复最优权重
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        self.early_stop = False

    def step(self, val_loss, model):
        """
        每轮验证后调用
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_weights = {k: v.cpu() for k, v in model.state_dict().items()}
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_weights = {k: v.cpu() for k, v in model.state_dict().items()}
            print(f"Validation loss improved to {val_loss:.6f}")
        else:
            self.counter += 1
            print(f"No improvement. Early stopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                print(f"Early stopping triggered after epoch {epoch}")
                self.early_stop = True
                if self.restore_best_weights:
                    print("Restoring best weights...")
                    model.load_state_dict(self.best_weights)





# ==================================================
# 0. CUDA 设置
# ==================================================
def init_cuda():
    """初始化CUDA设置"""
    # 禁用NCCL相关设置
    os.environ['NCCL_DEBUG'] = 'WARN'  # 降低NCCL日志级别
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 同步CUDA操作
    
    if torch.cuda.is_available():
        # 清理GPU缓存
        torch.cuda.empty_cache()
        
        # 使用确定性算法
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # 设置内存分配器
        torch.cuda.set_per_process_memory_fraction(0.8)  # 使用80%的可用GPU内存
        
        # 打印GPU信息
        print(f"Found {torch.cuda.device_count()} CUDA devices")
        for i in range(torch.cuda.device_count()):
            print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")

init_cuda()

# ==================================================
# 1. 固定随机种子（完全可复现）
# ==================================================
def set_seed(seed: int = 2025):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 确定性设置
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True,warn_only=True)

    def seed_worker(worker_id):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    return seed_worker, torch.Generator().manual_seed(seed)


# ==================================================
# 2. 数据集 & DataLoader
# ==================================================
set_seed(2025)

# ==================================================
train_data = GetData(
    image_dir='dataset/train/images',
    mask_dir='dataset/train/masks'
)
val_data = GetData(
    image_dir='dataset/val/images',
    mask_dir='dataset/val/masks'
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 16



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 减小batch size和worker数量以降低内存压力
# 进一步减小batch size以降低内存压力
batch_size = 16

train_loader = DataLoader(
    train_data, 
    batch_size=batch_size, 
    shuffle=True,
    num_workers=0,  # 暂时不使用多进程加载
    pin_memory=True if torch.cuda.is_available() else False,
    drop_last=True,
    # worker_init_fn=seed_worker, 
    # generator=g,
)

val_loader = DataLoader(
    val_data, 
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,  # 暂时不使用多进程加载
    pin_memory=True if torch.cuda.is_available() else False,
    drop_last=False,
    # worker_init_fn=seed_worker, 
    # generator=g,
)


# ==================================================
# 3. 模型、损失、优化器
# ==================================================
# 选择第一个可用的GPU
if torch.cuda.is_available():
    torch.cuda.set_device(0)  # 强制使用第一个GPU
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Using CPU")

model = EndToEndModel2(in_channels=1, num_classes=1).to(device)
# model = FPNUNet_CBAM_Residual().to(device)

print(f"Model created and moved to {device}")

model_name =  model.__class__.__name__   # 直接使用类名而不是从wrapped model获取
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = f"results/{model_name}_{timestamp}"
os.makedirs(log_dir, exist_ok=True)


class BCEWithLogitsLossWithSmoothing(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, targets):
        targets = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing
        return self.bce(logits, targets).mean()


class ComboLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, gamma=2.0):
        super().__init__()
        self.alpha = alpha  # BCE 权重
        self.beta = beta    # Dice 权重
        self.gamma = gamma  # Focal

    def forward(self, logits, targets):
        # BCE with Focal
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='mean')
        prob = torch.sigmoid(logits)
        focal_weight = (1 - prob) ** self.gamma * targets + prob ** self.gamma * (1 - targets)
        bce = (focal_weight * F.binary_cross_entropy_with_logits(logits, targets, reduction='none')).mean()

        # Dice
        smooth = 1e-6
        intersection = (prob * targets).sum(dim=(2,3))
        dice = 1 - (2 * intersection + smooth) / (prob.sum(dim=(2,3)) + targets.sum(dim=(2,3)) + smooth)
        dice = dice.mean()

        return self.alpha * bce + self.beta * dice



def dice_loss_per_sample(logits, masks, bce_weight=1.0, dice_weight=1.0, smooth=1e-6):
    """
    替换版：BCE + Dice 混合损失（推荐！）
    只改这一段，所有其他代码不动！
    """
    # 1. BCE 部分（使用 logits，数值稳定）
    bce_loss = nn.BCEWithLogitsLoss()(logits, masks)

    # 2. Dice 部分
    probs = torch.sigmoid(logits)
    B = probs.shape[0]
    probs_flat = probs.view(B, -1)
    masks_flat = masks.view(B, -1)

    inter = (probs_flat * masks_flat).sum(dim=1)
    union = probs_flat.sum(dim=1) + masks_flat.sum(dim=1)
    dice = (2 * inter + smooth) / (union + smooth)
    dice_loss = 1 - dice.mean()  # 标量

    # 3. 混合
    return bce_weight * bce_loss + dice_weight * dice_loss



class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=1.0, dice_weight=1.0, smooth=1e-6):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()  # 直接吃 logits，稳定！

    def forward(self, logits, masks):
        # 1. BCE 部分（带 logits，数值稳定）
        bce_loss = self.bce(logits, masks)

        # 2. Dice 部分（加 log 防梯度消失）
        probs = torch.sigmoid(logits)
        B, _, H, W = probs.shape
        probs_flat = probs.view(B, -1)
        masks_flat = masks.view(B, -1)

        inter = (probs_flat * masks_flat).sum(dim=1)
        union = probs_flat.sum(dim=1) + masks_flat.sum(dim=1)
        dice = (2 * inter + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice.mean()  # 标量！

        # 可选：加 log 增强小目标梯度
        # dice_loss = -torch.log(dice.clamp_min(1e-6)).mean()

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss



bce_loss = BCEWithLogitsLossWithSmoothing(smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3, verbose=True)


# =========================
# =========================
# 4. 日志文件
# ==================================================
log_csv = os.path.join(log_dir, f"log_{model_name}_{timestamp}.csv")
model_path = os.path.join(log_dir, f"best_{model_name}_{timestamp}.pth")
loss_plot_path = os.path.join(log_dir, f"loss_plot_{model_name}_{timestamp}.png")
image_save_template = os.path.join(log_dir, "epoch{}_{}.png".format("{:03d}", model_name))

with open(log_csv, mode="w", newline="") as f:
    
    writer = csv.writer(f)
    writer.writerow(["# Hyperparameters"])
    writer.writerow(["Model name", model_name])
    writer.writerow(["Time", timestamp])
    writer.writerow(["Batch size", train_loader.batch_size])
    writer.writerow(["Learning rate", optimizer.param_groups[0]['lr']])
    writer.writerow(["Loss function", "BCEWithLogitsLossWithSmoothing + dice_loss_per_sample"])
    writer.writerow(["Optimizer", type(optimizer).__name__])
    writer.writerow(["Scheduler", type(scheduler).__name__])
    writer.writerow([])
    writer.writerow(["epoch", "train_loss", "val_loss", "learning_rate"])


# ==================================================
# 5. 训练循环
# ==================================================
num_epochs = 100
best_val_loss = float("inf")

early_stopping = EarlyStopping(patience=15, min_delta=1e-6, restore_best_weights=True)


def train_tta(x, enable=False):
    if not enable:
        return x
    if random.random() > 0.5:
        x = torch.flip(x, [3])
    if random.random() > 0.5:
        x = torch.rot90(x, 1, [2, 3])
    return x

def boundary_loss(logits, masks, device='cpu'):

    probs = torch.sigmoid(logits)  
    device = logits.device# [B,1,H,W] -> [0,1]
    B, _, H, W = masks.shape
    
    dist_maps = []
    # 1. 先把 mask 转到 CPU + numpy（EDT 只接受 numpy）
    masks_np = masks.cpu().numpy()               # [B,1,H,W]
    
    for i in range(B):
        # 取阈值 >0.5 的二值图（uint8 能加速 EDT）
        binary = (masks_np[i, 0] > 0.5).astype(np.uint8)
        # 距离变换：背景（0）到最近前景（1）的距离
        dist = distance_transform_edt(1 - binary)   # 这里 1-binary 保证前景为 0（EDT 计算前景到背景的距离）
        # 归一化到 [0,1]，防止除以 0
        dist = dist / (dist.max() + 1e-6)
        dist_maps.append(dist)
    
    # 2. 堆叠成 [B,H,W]，转回 torch 并放到原设备
    dist_map = torch.from_numpy(np.stack(dist_maps)).unsqueeze(1).to(device, dtype=torch.float32)
    # dist_map: [B,1,H,W] 与 probs 对齐
    
    return F.mse_loss(probs, dist_map)





top_k = 3
saved_models = []

for epoch in range(1, num_epochs + 1):
    # ----------- 训练 -----------
    model.train()
    total_train_loss = 0.0
    for batch_idx, (imgs, masks) in enumerate(tqdm(train_loader, desc=f"[Train] Epoch {epoch}")):
        try:
            # 确保数据类型正确并移动到GPU
            imgs = imgs.float().to(device, non_blocking=True)
            masks = masks.float().to(device, non_blocking=True)
            imgss = train_tta(imgs,enable=False).to(device)  # 训练时 TTA
            
                        
            # 前向传播
            optimizer.zero_grad(set_to_none=True)  # 更高效的梯度清零
            segmentation,aux2, aux3, aux4 = model(imgs)
            
            # 将分割结果上采样到与输入相同的尺寸
            if segmentation.shape[-2:] != imgs.shape[-2:]:
                segmentation = F.interpolate(
                    segmentation,
                    size=imgs.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )
                
            # 关键：辅助头也必须上采样到原图尺寸！！
            aux2 = F.interpolate(aux2, size=imgs.shape[-2:], mode='bilinear', align_corners=False)
            aux3 = F.interpolate(aux3, size=imgs.shape[-2:], mode='bilinear', align_corners=False)
            aux4 = F.interpolate(aux4, size=imgs.shape[-2:], mode='bilinear', align_corners=False)

            # 计算各头损失（推荐 ComboLoss 或你现有的 dice+bce）
            loss_main = 0.5 * bce_loss(segmentation, masks) + 0.5 * dice_loss_per_sample(segmentation, masks, bce_weight=1.0, dice_weight=1.0)
            loss_a2   = 0.5 * bce_loss(aux2, masks)         + 0.5 * dice_loss_per_sample(aux2, masks, bce_weight=1.0, dice_weight=1.0)
            loss_a3   = 0.5 * bce_loss(aux3, masks)         + 0.5 * dice_loss_per_sample(aux3, masks, bce_weight=1.0, dice_weight=1.0)
            loss_a4   = 0.5 * bce_loss(aux4, masks)         + 0.5 * dice_loss_per_sample(aux4, masks, bce_weight=1.0, dice_weight=1.0)
                
        
            boundary = 0.8 * boundary_loss(imgss, masks) if 'boundary_loss' in globals() else 0

    
            # 计算损失
            # loss_bce = bce_loss(segmentation, masks)
            # # loss_dice = dice_loss_per_sample(segmentation, masks).mean()
            # loss_dice = dice_loss_per_sample(segmentation, masks, bce_weight=1.0, dice_weight=1.0)
            # loss = 0.5 * loss_bce +  0.5 * loss_dice + 0.8 * boundary_loss(imgss,masks)
            
            
                    # 终极深监督权重（最深层权重最大！实测最强）
            loss = (loss_main + 
                    0.8 * loss_a4 +    # 最深层，权重最大
                    0.6 * loss_a3 + 
                    0.4 * loss_a2) 
            
                
                    
            
            if not torch.isfinite(loss):
                print(f'警告: 检测到非有限损失值，跳过本批次 (batch {batch_idx})')
                continue
                
        except RuntimeError as e:
            print(f"运行时错误: {str(e)}")
            if "out of memory" in str(e):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print("GPU内存不足，跳过该batch")
                continue
            else:
                raise e
        
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item() * imgs.size(0)

    avg_train = total_train_loss / len(train_loader.dataset)

    # ----------- 验证 -----------
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for imgs, masks in tqdm(val_loader, desc=f"[ Val ] Epoch {epoch}"):
            imgs = imgs.float().to(device)
            masks = masks.float().to(device)
            segmentation,aux2, aux3, aux4 = model(imgs)
            
            if segmentation.shape[-2:] != imgs.shape[-2:]:
                segmentation = F.interpolate(segmentation, size=imgs.shape[-2:], mode='bilinear', align_corners=False)
            
                # 关键：辅助头也必须上采样到原图尺寸！！
                aux2 = F.interpolate(aux2, size=imgs.shape[-2:], mode='bilinear', align_corners=False)
                aux3 = F.interpolate(aux3, size=imgs.shape[-2:], mode='bilinear', align_corners=False)
                aux4 = F.interpolate(aux4, size=imgs.shape[-2:], mode='bilinear', align_corners=False)

                # 计算各头损失（推荐 ComboLoss 或你现有的 dice+bce）
                pred = torch.sigmoid(segmentation)
                pred_aux = torch.sigmoid(aux2 + aux3 + aux4 * 1.5)  # 辅助头融合增强
                pred_final = (pred + pred_aux) / 2   # 融合主头和辅助头
                    
                                        
                    
            # 将分割结果上采样到与输入相同的尺寸
            if segmentation.shape[-2:] != imgs.shape[-2:]:
                segmentation = F.interpolate(
                    segmentation,
                    size=imgs.shape[-2:],
                    mode='bilinear',
                    
                    align_corners=False
                )
                
            
            # loss_bce = bce_loss(segmentation, masks)
            
            # loss_dice = dice_loss_per_sample(segmentation, masks).mean()
            
            
            # loss = 0.5 * loss_bce + 0.8 * loss_dice
            
            loss = 0.5 * bce_loss(segmentation, masks) + 0.5 * dice_loss_per_sample(segmentation, masks).mean()
            

            total_val_loss += loss.item() * imgs.size(0)

    avg_val = total_val_loss / len(val_loader.dataset)

    print(f"Epoch {epoch:02d} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")

    # 写日志
    with open(log_csv, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch, f"{avg_train:.6f}", f"{avg_val:.6f}", optimizer.param_groups[0]['lr']])

    # 保存最优模型
    model_file = os.path.join(log_dir, f"model_epoch{epoch}_{avg_val:.4f}.pth")
    torch.save(model.state_dict(), model_file)
    saved_models.append((avg_val, model_file))
    saved_models.sort(key=lambda x: x[0])

    if len(saved_models) > top_k:
        _, to_delete = saved_models.pop(-1)
        if os.path.exists(to_delete):
            os.remove(to_delete)
            print(f"🗑️ Deleted old model: {to_delete}")

    if avg_val < best_val_loss:
        best_val_loss = avg_val


    # 学习率调度
    scheduler.step(avg_val)

    # 早停判断
    early_stopping.step(avg_val, model)
    if early_stopping.early_stop:
        print(f"Training stopped early at epoch {epoch}")
        break

    # 每10轮保存可视化图
    if epoch % 10 == 0:
        sample_imgs, sample_masks = next(iter(val_loader))
        sample_imgs = sample_imgs.to(device)
        with torch.no_grad():
            segmentation,aux2, aux3, aux4 = model(sample_imgs)
            
            # 确保分割结果与输入图像尺寸相同
            if segmentation.shape[-2:] != sample_imgs.shape[-2:]:
                segmentation = F.interpolate(
                    segmentation,
                    size=sample_imgs.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )
            
            sample_probs = torch.sigmoid(segmentation)
            sample_preds = (sample_probs > 0.5).float()

        sample_masks = sample_masks.to(device)
        
        # 确保所有图像具有相同的尺寸
        composites = []
        for i in range(min(4, sample_imgs.size(0))):
            img = sample_imgs[i]  # [1, H, W]
            msk = sample_masks[i]  # [1, H, W]
            pred = sample_preds[i]  # [1, H, W]
            
            # 确保所有图像都具有相同的大小
            target_size = img.shape[-2:]
            if msk.shape[-2:] != target_size:
                msk = F.interpolate(msk.unsqueeze(0), size=target_size, mode='nearest').squeeze(0)
            if pred.shape[-2:] != target_size:
                pred = F.interpolate(pred.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False).squeeze(0)
            
            # 连接图像
            comp = torch.cat([img, msk, pred], dim=2)  # 在宽度维度上连接
            composites.append(comp)

        grid = torch.stack(composites, dim=0)
        vutils.save_image(grid, image_save_template.format(epoch), nrow=2, normalize=True, scale_each=True)

    plot_loss_curve(log_csv, output_path=loss_plot_path, show_head=False)
    last_model_path = os.path.join(log_dir, "model_last.pth")
    torch.save(model.state_dict(), last_model_path)
    print(f"💾 Saved last model to {last_model_path}")

print(f"✅ Training complete! Best val loss: {best_val_loss:.4f}")



