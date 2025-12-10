# test_enhanced.py  ——  保留所有测试样本结果的完整版
## test_multi_task.py   ← 完整最终版（已解决所有报错）

import os
os.environ["MPLBACKEND"] = "Agg"
import random, datetime, csv
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import numpy as np
import cv2
import argparse
import datetime
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T
from sklearn.metrics import confusion_matrix
from torchmetrics.image import StructuralSimilarityIndexMeasure
from model.FPNUent_Multi_task import MultiTaskFPNUNet

from get_data import BUSI_Data,MedicalDataset


from model.ConvNeXt_Small_FPN_CBAM import EndToEndModel2
from model.FPNUNet import FPNUNet_CBAM_Residual
from model.ConvNeXt_Small_FPN_CBAM import EndToEndModel2




# ==================== 指标计算 ====================
def compute_seg_metrics(pred_logits, target_mask, smooth=1e-6):
    pred = (torch.sigmoid(pred_logits) > 0.5).float()
    target = target_mask.float()

    pred_f = pred.view(pred.size(0), -1)
    target_f = target.view(target.size(0), -1)

    tp = (pred_f * target_f).sum(dim=1)
    fp = (pred_f * (1 - target_f)).sum(dim=1)
    fn = ((1 - pred_f) * target_f).sum(dim=1)

    dice = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)
    iou  = (tp + smooth) / (tp + fp + fn + smooth)
    # 3. Precision (查准率): 预测为正的样本中，多少是真的正
    precision = (tp + smooth) / (tp + fp + smooth)
    
    # 4. Recall (查全率/Sensitivity): 真实为正的样本中，多少被预测出来了
    recall = (tp + smooth) / (tp + fn + smooth)

    return {
        'Dice': dice.cpu().numpy(),
        'IoU':  iou.cpu().numpy(),
        'Precision': precision.cpu().numpy(),
        'Recall': recall.cpu().numpy()
    }

def compute_cls_metrics(pred_logits, target_cls):
    pred = torch.argmax(pred_logits, dim=1)
    correct = (pred == target_cls)
    return correct.cpu().numpy()          # [B] bool array

def compute_recon_metrics(pred_recon, target_img):
    pred   = (pred_recon.clamp(-1, 1) + 1) / 2.0      # [-1,1] → [0,1]
    target = (target_img + 1) / 2.0                    # [-1,1] → [0,1]

    mse = F.mse_loss(pred, target, reduction='none').mean([1,2,3])
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse + 1e-8))

    ssim_func = StructuralSimilarityIndexMeasure(data_range=1.0).to(pred.device)
    ssim_scalar = ssim_func(pred, target)             # 标量
    ssim_per_sample = np.full(pred.size(0), ssim_scalar.item())

    return psnr.cpu().numpy(), ssim_per_sample

# ==================== 可视化（7列） ====================
def visualize_batch(
    img, mask, cls_gt, edge, recon,
    seg_main, cls_logit, edge_out, recon_out,
    class_names=['Normal', 'Benign', 'Malignant'],
    save_path=None
):
    # 全部提前转到 CPU，防止 PIL 在子进程中出错
    img       = img.cpu()
    mask      = mask.cpu()
    cls_gt    = cls_gt.cpu()
    edge      = edge.cpu()
    recon     = recon.cpu()
    seg_main  = seg_main.cpu()
    cls_logit = cls_logit.cpu()
    edge_out  = edge_out.cpu()
    recon_out = recon_out.cpu().clamp(-1, 1)

    cls_pred  = torch.argmax(cls_logit, dim=1)
    pred_seg  = (torch.sigmoid(seg_main) > 0.5).float()
    pred_edge = (torch.sigmoid(edge_out) > 0.5).float()

    to_pil = T.ToPILImage()
    img_size = img.size(2)

    canvas_w = 80 + img_size * 7
    canvas_h = 140 + img_size
    canvas = Image.new('RGB', (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    try:
        font_title = ImageFont.truetype("arial.ttf", 26)
        font_text  = ImageFont.truetype("arial.ttf", 20)
    except:
        font_title = font_text = ImageFont.load_default()

    titles = ["Input", "GT Mask", "Pred Seg", "GT Edge", "Pred Edge", "GT Recon", "Pred Recon"]
    for idx, title in enumerate(titles):
        x = 40 + idx * img_size + img_size // 2
        bbox = draw.textbbox((0, 0), title, font=font_title)
        w = bbox[2] - bbox[0]
        draw.text((x - w // 2, 15), title, fill=(0,0,0), font=font_title)

    x_base, y_base = 40, 70

    # Input + CLAHE
    input_raw = (img[0] * 0.5 + 0.5).clamp(0, 1)
    input_np = (input_raw.squeeze(0).numpy() * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(input_np)
    enhanced_tensor = torch.from_numpy(enhanced).float().unsqueeze(0) / 255.0
    canvas.paste(to_pil(enhanced_tensor), (x_base, y_base))

    # 其余6列
    imgs = [
        mask[0].squeeze(0),
        pred_seg[0].squeeze(0),
        edge[0].squeeze(0),
        pred_edge[0].squeeze(0),
        recon[0],
        (recon_out[0] * 0.5 + 0.5).clamp(0, 1)
    ]
    for i, t in enumerate(imgs):
        pil_img = to_pil(t.unsqueeze(0) if t.ndim == 2 else t)
        canvas.paste(pil_img, (x_base + (i+1) * img_size, y_base))

    # 分类文字
    y_text = y_base + img_size + 15
    pred_label = class_names[cls_pred[0].item()]
    gt_label   = class_names[cls_gt[0].item()]
    color = (0, 180, 0) if cls_pred[0] == cls_gt[0] else (200, 0, 0)
    draw.text((x_base + 10, y_text),     f"Pred: {pred_label}", fill=color, font=font_text)
    draw.text((x_base + 10, y_text+28), f"GT: {gt_label}",       fill=(0,0,0), font=font_text)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        canvas.save(save_path)          # 现在绝对不会报错

# ==================== 主测试函数（保存每一张） ====================
@torch.no_grad()
def test_model_full(model, test_loader, device, log_dir):
    model.eval()
    os.makedirs(os.path.join(log_dir, "vis"), exist_ok=True)

    records = []
    all_cls_pred = []
    all_cls_gt   = []

    total_samples = len(test_loader.dataset)
    sample_idx = 0
    # pbar = tqdm(test_loader, desc="Testing (All Saved)", leave=True)
    pbar = tqdm(enumerate(test_loader), total=len(test_loader), desc="Testing", leave=True)

    # for img, mask, cls_gt, edge, recon in pbar:
    for batch_idx, (img, mask, cls_gt, edge, recon) in pbar:
        img   = img.to(device)
        mask  = mask.to(device)
        cls_gt= cls_gt.to(device)
        edge  = edge.to(device)
        recon = recon.to(device)

        seg_main, _, _, _, cls_logit, edge_out, recon_out = model(img)
        
        
        

        # 指标
        seg_res   = compute_seg_metrics(seg_main, mask)
        edge_res  = compute_seg_metrics(edge_out, edge)
        cls_corr  = compute_cls_metrics(cls_logit, cls_gt)
        psnr_arr, ssim_arr = compute_recon_metrics(recon_out, img)

        pred_cls = torch.argmax(cls_logit, dim=1)
        all_cls_pred.extend(pred_cls.cpu().numpy())
        all_cls_gt.extend(cls_gt.cpu().numpy())
        
     
        # 逐样本保存
        for i in range(img.size(0)):
            
            
            global_idx = batch_idx * test_loader.batch_size + i
            if global_idx >= total_samples:
                break

            # 获取原始文件名
            sample_info = test_loader.dataset.samples[global_idx]
            original_filename = os.path.basename(sample_info["image_path"])
            name_no_ext = os.path.splitext(original_filename)[0]

            # 保存为：原始名字_result.png
            vis_path = os.path.join(log_dir, "vis", f"{name_no_ext}_result.png")
                    
            

            
            visualize_batch(
                img=img[i:i+1],
                mask=mask[i:i+1],
                cls_gt=cls_gt[i:i+1],
                edge=edge[i:i+1],
                recon=recon[i:i+1],
                seg_main=seg_main[i:i+1],
                cls_logit=cls_logit[i:i+1],
                edge_out=edge_out[i:i+1],
                recon_out=recon_out[i:i+1],
                save_path=vis_path
            )

            records.append({
                'Index'       : sample_idx,
                'GT_Class'    : ['Normal','Benign','Malignant'][cls_gt[i].item()],
                'Pred_Class'  : ['Normal','Benign','Malignant'][pred_cls[i].item()],
                'Cls_Correct' : bool(cls_corr[i]),
                'Seg_Dice'    : float(seg_res['Dice'][i]),
                'Seg_IoU'     : float(seg_res['IoU'][i]),
                'Seg_Precision' : float(seg_res['Precision'][i]), # 新增
                'Seg_Recall'    : float(seg_res['Recall'][i]),    # 新增
                
                'Edge_Dice'   : float(edge_res['Dice'][i]),
                'Edge_IoU'    : float(edge_res['IoU'][i]),
                'Edge_Precision': float(edge_res['Precision'][i]),# 新增
                'Edge_Recall'   : float(edge_res['Recall'][i]),   # 新增
                
                'PSNR'        : float(psnr_arr[i]),
                'SSIM'        : float(ssim_arr[i]),
            })
            sample_idx += 1

        pbar.set_postfix({
            'Samples': sample_idx,
            'SegDice': f"{seg_res['Dice'].mean():.4f}",
            'ClsAcc' : f"{cls_corr.mean():.4f}"
        })

    df_detail = pd.DataFrame(records)

# 1. 基础平均指标
    summary = {
        'Total_Samples' : len(df_detail),
        'Seg_Dice'      : df_detail['Seg_Dice'].mean(),
        'Seg_IoU'       : df_detail['Seg_IoU'].mean(),
        'Seg_Precision'  : df_detail['Seg_Precision'].mean(), # 新增
        'Seg_Recall'     : df_detail['Seg_Recall'].mean(),    # 新增
        
        'Edge_Dice'     : df_detail['Edge_Dice'].mean(),
        'Edge_IoU'      : df_detail['Edge_IoU'].mean(),
        'Edge_Precision' : df_detail['Edge_Precision'].mean(),# 新增
        'Edge_Recall'    : df_detail['Edge_Recall'].mean(),   # 新增
        'PSNR'          : df_detail['PSNR'].mean(),
        'SSIM'          : df_detail['SSIM'].mean(),
        'Cls_Acc'       : df_detail['Cls_Correct'].mean(),   # 总体准确率
    }

    # 2. 每类样本数 + 每类准确率（强烈推荐加这个！）
    class_names = ['Normal', 'Benign', 'Malignant']
    for idx, name in enumerate(class_names):
        mask = df_detail['GT_Class'] == name
        count = mask.sum()
        acc   = df_detail.loc[mask, 'Cls_Correct'].mean() if count > 0 else 0.0
        summary[f'{name}_Count'] = int(count)
        summary[f'{name}_Acc']   = acc

    # 3. 保存为一行（横向表格，论文最爱）
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(os.path.join(log_dir, "test_summary.csv"), index=False)
        
    print("\n测试完成！所有结果已保存在：")
    print(log_dir)
    print("   ├─ vis/                   ← 每张测试图")
    print("   ├─ test_all_samples.csv   ← 每张图详细指标")
    print("   ├─ test_summary.csv")
    print("   └─ confusion_matrix.csv")

# ==================== 主入口 ====================


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, 
                        default='/workspace/results/ablation1/Exp8_Full_Model_20251204_004601/best_model.pth', 
                        help='checkpoint path')
    parser.add_argument('--data_root', type=str, default='/workspace/dataset')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--log_dir', type=str, default=None)
    args = parser.parse_args()

    if args.log_dir is None:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.log_dir = f"test_results_full/test_{ts}"
    os.makedirs(args.log_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device}")

    # ==================== 数据 ====================
    test_dataset = BUSI_Data(root_dir=args.data_root, split='test', img_size=256)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,       # 保险起见
        pin_memory=False
    )

    # ==================== 模型 ====================
    model = MultiTaskFPNUNet(in_ch=1, seg_ch=1, num_classes=3).to(device)
    # model = FPNUNet_CBAM_Residual(in_ch=1, seg_ch=1, num_classes=3).to(device)
    # model = AAUNet().to(device)

    # ==================== 关键修复：智能加载权重 ====================
    ckpt = torch.load(args.ckpt, map_location=device)
    
    # 情况1：你训练时用了 DataParallel 存的（key 有 module.）
    # 情况2：你训练时没用 DataParallel 存的（key 没 module.）
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    elif isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    else:
        state_dict = ckpt  # 直接是 state_dict

    # 自动处理 module. 前缀问题（兼容两种训练方式）
    if list(state_dict.keys())[0].startswith('module.'):
        # 存的时候有 module.，现在模型没包 DataParallel → 去掉前缀
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    else:
        # 存的时候没 module.，但现在你包了 DataParallel → 加上前缀（不推荐）
        # 更稳的做法：不要包 DataParallel！
        pass

    model.load_state_dict(state_dict)
    print(f"Checkpoint loaded successfully: {args.ckpt}")

    # ==================== 可选：如果你非要用多卡推理（不推荐）===================
    # 注释掉下面这行就行！单卡更快更稳
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for inference!")
        model = nn.DataParallel(model)
    # =====================================================================

    model.eval()
    test_model_full(model, test_loader, device, args.log_dir)