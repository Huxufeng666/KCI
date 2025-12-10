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

# ------------------ 你的模块 ------------------
from get_data import BUSI_Data 
from model.FPNUent_Multi_task import MultiTaskFPNUNet
from utils.tools import visualize_batch

# import csv
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



from model.U_net import UNet


# ==================== 早停机制类 (新增!) ====================
class EarlyStopping:
    """
    当验证集指标（如 Dice）在 patience 轮次内没有提升时，停止训练。
    """
    def __init__(self, patience=20, verbose=False, delta=0):
        """
        Args:
            patience (int): 上次指标提升后，等待多少轮（默认 20）。
            verbose (bool): 是否打印日志。
            delta (float): 指标被认为提升的最小变化量。
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_score_max = -np.inf
        self.delta = delta

    def __call__(self, val_score):
        # 这里的 val_score 是 Dice，越大越好
        score = val_score

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


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

def dice_coeff(logits, targets, smooth=1e-6):
    prob = torch.sigmoid(logits).flatten(1)
    targets = targets.flatten(1)
    inter = (prob * targets).sum(1)
    union = prob.sum(1) + targets.sum(1)
    return ((2. * inter + smooth) / (union + smooth)).mean()

# 全局 Loss 实例
seg_criterion = BCEDiceLoss()
ce_cls  = nn.CrossEntropyLoss()
bce_edge = nn.BCEWithLogitsLoss()
l1_rec  = nn.L1Loss()

# ==================== 设置随机种子 ====================
def set_seed(seed=2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ==================== 核心：单次实验运行函数 ====================
def run_experiment(exp_name, config):
    print(f"\n{'='*20} Start Experiment: {exp_name} {'='*20}")
    print(f"Config: {config}")
    
    set_seed(2025) # 保证每组实验初始权重一致，公平比较

    # 1. 设置路径
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"results/ablation/{exp_name}_{ts}"
    os.makedirs(log_dir, exist_ok=True)
    log_csv = os.path.join(log_dir, "log.csv")
    
    with open(log_csv, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "val_dice", "high_conf_dice", "cls_acc", "lr"])

    # 2. 数据与模型
    IMG_SIZE = 256
    BATCH_SIZE = 4 # 根据显存调整
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set = BUSI_Data(root_dir="/workspace/dataset", split="train", img_size=IMG_SIZE)
    val_set   = BUSI_Data(root_dir="/workspace/dataset", split="val",   img_size=IMG_SIZE)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=0)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # model = UNetPlusPlus(in_ch=1, seg_ch=1, num_classes=3).to(device)
    model = UNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    
    early_stopping = EarlyStopping(patience=20, verbose=True)
    
    # 3. 训练参数
    EPOCHS = 150 # 消融实验可以适当减少轮数，比如100-150轮看趋势
    SEG_START_EPOCH = 6
    CONF_THRESH = 0.95
    best_val_dice = 0.0

    # ==================== 训练循环 ====================
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        
        for img, mask, cls_label, edge_gt, recon_gt in tqdm(train_loader, desc=f"[{exp_name}] Ep {epoch}", leave=False):
            img = img.to(device)
            mask = mask.to(device).float()
            cls_label = cls_label.to(device).long()
            edge_gt = edge_gt.to(device).float()
            recon_gt = recon_gt.to(device)

            optimizer.zero_grad()
            seg_main, aux2, aux3, aux4, cls_logit, edge_out, recon_out = model(img)

            total_loss = torch.tensor(0.0, device=device)

            # ------------------ A. 辅助任务 Loss (根据 Config 开关) ------------------
            if config['use_cls']:
                total_loss += ce_cls(cls_logit, cls_label)
            
            if config['use_edge']:
                total_loss += bce_edge(edge_out, edge_gt)
            
            if config['use_recon']:
                total_loss += 0.02 * l1_rec(recon_out, recon_gt)

            # ------------------ B. 主任务 Segmentation Loss ------------------
            # 策略：如果有 Cls 且开启了置信度筛选，则使用筛选逻辑；否则全量训练
            should_train_seg = False
            high_conf_mask = torch.ones(img.size(0), 1, 1, 1).to(device) # 默认全选

            if config['use_cls']:
                # 原有逻辑：前几轮不训分割，后面按置信度训
                if epoch >= SEG_START_EPOCH:
                    with torch.no_grad():
                        prob = F.softmax(cls_logit, dim=1)
                        confidence, _ = torch.max(prob, dim=1)
                        high_conf_mask = (confidence >= CONF_THRESH).float().view(-1, 1, 1, 1)
                        if high_conf_mask.sum() > 0:
                            should_train_seg = True
            else:
                # 无 Cls 时：始终训练分割 (也可以保留 warmup，这里简化为直接训)
                should_train_seg = True

            if should_train_seg:
                H, W = mask.shape[-2:]
                # 上采样
                seg_up = F.interpolate(seg_main, size=(H,W), mode='bilinear', align_corners=False)
                aux2_up = F.interpolate(aux2, size=(H,W), mode='bilinear', align_corners=False)
                aux3_up = F.interpolate(aux3, size=(H,W), mode='bilinear', align_corners=False)
                aux4_up = F.interpolate(aux4, size=(H,W), mode='bilinear', align_corners=False)

                # 计算基础分割 Loss
                l_main = seg_criterion(seg_up, mask)
                l_aux = 0.8*seg_criterion(aux4_up, mask) + 0.6*seg_criterion(aux3_up, mask) + 0.4*seg_criterion(aux2_up, mask)
                
                # 如果有 Cls，应用 Mask 筛选 (只计算高置信度样本)
                if config['use_cls']:
                    total_loss += (l_main + l_aux) # 简化：只要有样本过线，就加这个 Loss
                else:
                    total_loss += (l_main + l_aux)

            # Backprop
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += total_loss.item()

        # ==================== 验证循环 ====================
        model.eval()
        val_losses = []
        val_dice_all = []
        val_dice_high = []
        cls_correct = 0
        total_samples = 0

        with torch.no_grad():
            for img, mask, cls_gt, edge_gt, recon_gt in val_loader:
                img = img.to(device); mask = mask.to(device).float(); cls_gt = cls_gt.to(device).long()
                edge_gt = edge_gt.to(device).float(); recon_gt = recon_gt.to(device)

                seg_main, _, _, _, cls_logit, edge_out, recon_out = model(img)
                H, W = img.shape[-2:]
                seg_main = F.interpolate(seg_main, size=(H,W), mode='bilinear', align_corners=False)

                # 计算验证 Loss (仅供参考，不影响指标)
                v_loss = BCEDiceLoss()(seg_main, mask) # 始终计算分割 Loss
                if config['use_cls']: v_loss += 0.3 * ce_cls(cls_logit, cls_gt)
                if config['use_edge']: v_loss += bce_edge(F.interpolate(edge_out, size=(H,W)), edge_gt)
                if config['use_recon']: v_loss += 0.02 * l1_rec(F.interpolate(recon_out, size=(H,W)), img)
                
                val_losses.append(v_loss.item())
                val_dice_all.append(dice_coeff(seg_main, mask).item())

                # 记录分类准确率
                if config['use_cls']:
                    cls_correct += (cls_logit.argmax(1) == cls_gt).sum().item()
                    
                    # 记录高置信样本 Dice
                    prob = F.softmax(cls_logit, dim=1)
                    conf, _ = torch.max(prob, dim=1)
                    high_idx = conf >= 0.95
                    if high_idx.sum() > 0:
                        val_dice_high.append(dice_coeff(seg_main[high_idx], mask[high_idx]).item())
                
                total_samples += cls_gt.size(0)

        # 统计指标
        avg_val_loss = np.mean(val_losses)
        avg_dice = np.mean(val_dice_all)
        high_dice = np.mean(val_dice_high) if val_dice_high else 0.0
        cls_acc = cls_correct / total_samples if config['use_cls'] else 0.0
        
        # 保存最佳 Dice 模型
        if avg_dice > best_val_dice:
            best_val_dice = avg_dice
            torch.save(model.state_dict(), os.path.join(log_dir, "best_model.pth"))
            print(f" ✨ New Best Dice: {best_val_dice:.4f}")

        # ==================== 核心修改：早停检查 ====================
        early_stopping(avg_dice) # 传入要监控的指标，这里是 Dice
        if early_stopping.early_stop:
            print(f"🛑 Early stopping triggered at epoch {epoch}")
            break # 跳出 epoch 循环，结束当前实验
        
        
        # 记录日志
        with open(log_csv, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{epoch_loss/len(train_loader):.4f}", f"{avg_val_loss:.4f}", 
                                    f"{avg_dice:.4f}", f"{high_dice:.4f}", f"{cls_acc:.4f}", 
                                    f"{optimizer.param_groups[0]['lr']:.2e}"])
        
        scheduler.step(avg_val_loss)

    print(f"✅ Experiment {exp_name} Finished. Best Dice: {best_val_dice:.4f}")
    torch.cuda.empty_cache()

# ==================== 主入口：执行 5 组实验 ====================
if __name__ == "__main__":
    
    # 定义 5 组实验配置
    experiments = {
        # Exp 1: Baseline (纯分割)
        "Exp1_Baseline": {'use_cls': False, 'use_edge': False, 'use_recon': False},
        
        # Exp 2: + Classification
        "Exp2_with_Cls": {'use_cls': True,  'use_edge': False, 'use_recon': False},
        
        # Exp 3: + Edge
        "Exp3_with_Edge":{'use_cls': False, 'use_edge': True,  'use_recon': False},
        
        # Exp 4: + Recon
        "Exp4_with_Recon":{'use_cls': False, 'use_edge': False, 'use_recon': True},
                       
        # Exp 5: + Recon + Edge
        "Exp5_with_Recon":{'use_cls': False, 'use_edge': True, 'use_recon': True},
               
        # Exp 6: + Recon+ Classification
        "Exp6_with_Recon":{'use_cls': True, 'use_edge': False, 'use_recon': True},
                
        # Exp 7: + Edge + Classification
        "Exp7_with_Recon":{'use_cls': True, 'use_edge': True, 'use_recon': True},
           
        # Exp 8: Full Proposed (全部开启)
        "Exp8_Full_Model":{'use_cls': True,  'use_edge': True,  'use_recon': False},
    }

    # 你可以选择运行全部，或者只运行某一个
    # for name, config in experiments.items():
    #     run_experiment(name, config)
    
    # 例如：只运行 Exp 1
    run_experiment("Unet", experiments["Exp1_Baseline"])
    # 运行 Exp 2
    # run_experiment("Exp2_with_Cls", experiments["Exp2_with_Cls"])
    # 运行 Exp 3
    # run_experiment("Exp3_with_Edge", experiments["Exp3_with_Edge"])

    # 运行 Exp 4
    # run_experiment("Exp4_with_Recon", experiments["Exp4_with_Recon"])
    # 运行 Exp 5
    # run_experiment("Exp5_with_Recon", experiments["Exp5_with_Recon"])
    # 运行 Exp 6        
    # run_experiment("Exp6_with_Recon", experiments["Exp6_with_Recon"])
    # 运行 Exp 7
    # run_experiment("Exp7_with_Recon", experiments["Exp7_with_Recon"])
    # 运行 Exp 8
    # run_experiment("Exp8_Full_Model", experiments["Exp8_Full_Model"])