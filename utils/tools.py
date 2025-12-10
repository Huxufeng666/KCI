import torch
import matplotlib.pyplot as plt
import pandas as pd



def init_loss_log(log_path: str, header: str = "epoch,train_loss\n") -> None:
    """
    初始化或清空训练损失日志文件，并写入表头。
    """
    with open(log_path, "w") as f:
        f.write(header)
        
def append_loss_log(log_path: str, epoch: int, train_loss: float) -> None:
    """
    向日志文件末尾追加一行训练损失数据。

    Args:
        log_path (str): 日志文件路径。
        epoch (int): 当前 epoch 编号（从 1 开始）。
        train_loss (float): 这一轮的训练平均损失。
    """
    # 'a' 模式：不存在则自动创建，存在则追加
    with open(log_path, "a") as f:
        f.write(f"{epoch},{train_loss:.6f}\n")
        
        


def dice_loss(pred, target, smooth=1e-6):
    """
    pred: [B, C, H, W] logits
    target: [B, C, H, W] binary mask (0或1)
    """
    # 1) 把 logits → 概率
    pred = torch.sigmoid(pred)

    # 2) 拉平到 [B, N]，N = C*H*W
    B = pred.shape[0]
    pred_flat   = pred.view(B, -1)
    target_flat = target.view(B, -1)

    # 3) 每个样本分别求交集和并集
    intersection = (pred_flat * target_flat).sum(dim=1)      # [B]
    union        = pred_flat.sum(dim=1) + target_flat.sum(dim=1)  # [B]

    # 4) 计算每个样本的 Dice 系数，再转为损失
    dice_score = (2 * intersection + smooth) / (union + smooth)  # [B]
    loss = 1 - dice_score                                       # [B]

    # 5) 对 batch 取平均
    return loss.mean()



def dice_loss_per_sample(logits: torch.Tensor, masks: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """
    计算每个样本的 Dice Loss，然后返回一个 [B] 的张量。
    
    Args:
        logits: 模型原始输出，shape [B, C, H, W]
        masks: 二值化的 ground-truth，shape [B, C, H, W]
        smooth: 平滑项，防止除零
    
    Returns:
        loss_per_sample: 每个样本的 Dice Loss，shape [B]
    """
    # 1) logits -> 概率
    probs = torch.sigmoid(logits)                      # [B, C, H, W]
    B = probs.shape[0]
    # 2) 展平到 [B, N]
    probs_flat = probs.view(B, -1)
    masks_flat = masks.view(B, -1)
    # 3) 计算交集和并集
    intersection = (probs_flat * masks_flat).sum(dim=1)       # [B]
    union        = probs_flat.sum(dim=1) + masks_flat.sum(dim=1)  # [B]
    # 4) 每个样本 Dice 系数 & Dice Loss
    dice_score = (2 * intersection + smooth) / (union + smooth)  # [B]
    loss_per_sample = 1 - dice_score                             # [B]
    return loss_per_sample





def visualize_with_labels(orig, probs, mask, save_path, n=4):
    """
    在 Matplotlib Canvas 上绘制 n 个样本，每个样本有三列：
      原图 | 概率图 | 二值掩码
    并在第一行加上文字 'Image','Probability','Mask'。

    Args:
        orig (Tensor): 原图，[B,1,H,W]，取值 [0,1]
        probs (Tensor): 预测概率图，[B,1,H,W]
        mask (Tensor): 二值化掩码，[B,1,H,W]
        save_path (str): 保存路径
        n (int): 展示样本数，<= B
    """
    B, _, H, W = orig.shape
    n = min(n, B)

    # 转成 numpy [n,H,W]
    orig_np = orig[:n].cpu().squeeze(1).numpy()
    probs_np = probs[:n].cpu().squeeze(1).numpy()
    # mask_np  = mask[:n].cpu().squeeze(1).numpy()

    # 创建 n 行 3 列的画板
    fig, axes = plt.subplots(n, 3, figsize=(3*3, n*3))
    if n == 1:
        axes = axes.reshape(1, -1)

    # 在第一行加上列标题
    col_titles = ["Image", "Probability"]#, "Mask"]
    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontsize=14)

    # 绘制每个样本
    for i in range(n):
        for j, arr in enumerate([orig_np[i], probs_np[i]]):#, mask_np[i]]):
            ax = axes[i, j]
            ax.imshow(arr, cmap='gray', vmin=0, vmax=1)
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    



def plot_loss_curve(csv_path: str,
                    output_path: str = "loss_curve.png",
                    show_head: bool = True):
    """
    从指定的 CSV 文件读取训练/验证损失日志，并绘制 loss 曲线。

    Args:
        csv_path (str): CSV 文件路径，包含 epoch, train_loss, val_loss 三列。
        output_path (str): 保存生成的曲线图像的路径。
        show_head (bool): 是否在控制台打印 CSV 前五行。
    """
    # 1) 读取 CSV
    # df = pd.read_csv(csv_path, skiprows=8,header=0)
    df = pd.read_csv(csv_path)#, skiprows=8,header=0)

    # 2) 清理列名：去除首尾空格、BOM 等
    df.columns = df.columns.str.strip().str.replace('\ufeff', '')

    # 3) 检查必须列是否存在
    expected = ['epoch', 'train_loss', 'val_loss']
    missing = [col for col in expected if col not in df.columns]
    if missing:
        print("CSV 中实际的列名为:", df.columns.tolist())
        raise KeyError(f"在 CSV 中找不到以下列: {missing}")

    # 4) （可选）打印前几行
    if show_head:
        print(df.head())

    # 5) 绘制曲线
    plt.figure(figsize=(8, 5))
    plt.plot(df['epoch'], df['train_loss'], marker='o', label='Train Loss')
    plt.plot(df['epoch'], df['val_loss'], marker='s', label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim(0, 0.8)
    plt.title('Training & Validation Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # 6) 保存并显示
    plt.savefig(output_path, dpi=150)
    # plt.show()
    print(f"Saved loss curve to {output_path}")

# def plot_loss_curve(csv_path: str,
#                     output_path: str = "loss_curve.png",
#                     show_head: bool = True):
#     """
#     从指定的 CSV 文件读取训练/验证损失日志，并绘制 loss 曲线。
    
#     Args:
#         csv_path (str): CSV 文件路径，包含 epoch, train_loss, val_loss 三列。
#         output_path (str): 保存生成的曲线图像的路径。
#         show_head (bool): 是否在控制台打印 CSV 前几行。
#     """

#     # ---------------------------
#     # 1. 跳过前7行，读取后续数据，指定 header
#     # ---------------------------
#     try:
#         df = pd.read_csv(csv_path)#, skiprows=7, header=0)
#     except Exception as e:
#         raise RuntimeError(f"❌ 读取 CSV 文件失败: {e}")

#     # ---------------------------
#     # 2. 清洗列名（去除 BOM、空格）
#     # ---------------------------
#     df.columns = df.columns.str.strip().str.replace('\ufeff', '')  # 清洗列名

#     # ---------------------------
#     # 3. 确认所需列是否存在
#     # ---------------------------
#     required_cols = ['epoch', 'train_loss', 'val_loss']
#     missing = [col for col in required_cols if col not in df.columns]
#     if missing:
#         print("❌ CSV 实际列名为:", df.columns.tolist())
#         raise KeyError(f"在 CSV 中找不到以下列: {missing}")

#     # ---------------------------
#     # 4. 打印表头
#     # ---------------------------
#     if show_head:
#         print("✅ CSV 文件前几行如下：")
#         print(df[required_cols].head())

#     # ---------------------------
#     # 5. 绘制 Loss 曲线
#     # ---------------------------
#     plt.figure(figsize=(8, 5))
#     plt.plot(df['epoch'], df['train_loss'], marker='o', label='Train Loss')
#     plt.plot(df['epoch'], df['val_loss'], marker='s', label='Val Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.ylim(0, 1)
#     plt.title('Training & Validation Loss over Epochs')
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()

#     # ---------------------------
#     # 6. 保存并显示
#     # ---------------------------
#     plt.savefig(output_path, dpi=150)
#     plt.show()
#     print(f"✅ Loss 曲线保存至: {output_path}")


# plot_loss_curve("train_result/ResUNetEncoder_train_val_log.csv", output_path="train_result/ResUNetEncoder_loss_plot.png", show_head=False)




def dice_loss(logits, targets, smooth=1e-5):
    probs = torch.sigmoid(logits)
    intersection = (probs * targets).sum(dim=(1,2,3))
    union = probs.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3))
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()




# def visualize_batch(img, mask, cls_gt, edge, recon,
#                     seg_main, cls_logit, edge_out, recon_out,
#                     class_names=['Normal', 'Benign', 'Malignant'],
#                     save_path=None):
#     import torch
#     from PIL import Image, ImageDraw, ImageFont
#     import torchvision.transforms as T
#     import cv2
#     import numpy as np
#     # 转为 CPU
#     img = img.cpu()
#     mask = mask.cpu()
#     cls_gt = cls_gt.cpu()
#     seg_main = seg_main.cpu()
#     cls_logit = cls_logit.cpu()
#     edge_out = edge_out.cpu()
#     recon_out = recon_out.cpu().clamp(0, 1)

#     # 预测
#     cls_pred = torch.argmax(cls_logit, dim=1)  # [1]
#     pred_seg = (torch.sigmoid(seg_main) > 0.5).float()
#     pred_edge = (torch.sigmoid(edge_out) > 0.5).float()

#     to_pil = T.ToPILImage()
#     img_size = img.size(2)

#     # 画布
#     canvas_w = 60 + img_size * 6
#     canvas_h = 120 + img_size
#     canvas = Image.new('RGB', (canvas_w, canvas_h), (255, 255, 255))
#     draw = ImageDraw.Draw(canvas)

#     try:
#         font_small = ImageFont.truetype("arial.ttf", 18)
#         font_large = ImageFont.truetype("arial.ttf", 26)
#     except:
#         font_small = ImageFont.load_default()
#         font_large = ImageFont.load_default()

#     # 标题
#     titles = ["Input", "GT Mask", "Pred Seg", "GT Edge", "Pred Edge", "Recon"]
#     for idx, title in enumerate(titles):
#         x = 30 + idx * img_size + img_size // 2
#         bbox = draw.textbbox((0, 0), title, font=font_large)
#         text_w = bbox[2] - bbox[0]
#         draw.text((x - text_w // 2, 10), title, fill=(0,0,0), font=font_large)

#     # 图像
#     x_base = 30
#      # , H, W]，只反归一化

#     input_raw = (img[0] * 0.5 + 0.5).clamp(0, 1)  # [1, H, W]
#     input_np = input_raw.squeeze(0).cpu().numpy()
#     input_np = (input_np * 255).astype(np.uint8)
#     clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
#     enhanced = clahe.apply(input_np)
#     enhanced = torch.from_numpy(enhanced).float() / 255.0
#     enhanced = enhanced.unsqueeze(0)  # [1, H, W]
#     canvas.paste(to_pil(enhanced), (x_base, 70))
    
    

#     # 其他图：squeeze(0) → [H, W] → unsqueeze(0) → [1, H, W]
#     canvas.paste(to_pil(mask[0].squeeze(0).unsqueeze(0)), (x_base + img_size*1, 70))
#     canvas.paste(to_pil(pred_seg[0].squeeze(0).unsqueeze(0)), (x_base + img_size*2, 70))
#     canvas.paste(to_pil(edge[0].squeeze(0).unsqueeze(0)), (x_base + img_size*3, 70))
#     canvas.paste(to_pil(pred_edge[0].squeeze(0).unsqueeze(0)), (x_base + img_size*4, 70))
#     canvas.paste(to_pil(recon_out[0].squeeze(0).unsqueeze(0)), (x_base + img_size*5, 70))

#     # 文字
#     y_text = 70 + img_size + 10
#     pred_text = f"Pred: {class_names[cls_pred[0].item()]}"
#     gt_correct = cls_pred[0].item() == cls_gt[0].item()
#     pred_color = (0, 255, 0) if gt_correct else (255, 0, 0)
#     draw.text((x_base + 10, y_text), pred_text, fill=pred_color, font=font_small)
#     draw.text((x_base + 10, y_text + 25), f"GT: {class_names[cls_gt[0].item()]}", fill=(0, 0, 0), font=font_small)

#     if save_path:
#         canvas.save(save_path)
#     return canvas


# utils/tools.py
# import torch
# from PIL import Image, ImageDraw, ImageFont
# import torchvision.transforms as T
# import cv2
# import numpy as np


# def visualize_batch(
#     img, mask, cls_gt, edge, recon,               # GT: img, mask, cls_gt, edge, recon
#     seg_main, cls_logit, edge_out, recon_out,     # Pred: seg_main, cls_logit, edge_out, recon_out
#     class_names=['Normal', 'Benign', 'Malignant'],
#     save_path=None
# ):
#     """
#     可视化 7 列：
#     [Input] [GT Mask] [Pred Seg] [GT Edge] [Pred Edge] [GT Recon] [Pred Recon]
#     + 分类文字（绿色/红色）
#     + CLAHE 增强输入图
#     + 加粗 GT Edge
#     """
#     # ------------------- 1. 转 CPU -------------------
#     img = img.cpu()
#     mask = mask.cpu()
#     cls_gt = cls_gt.cpu()
#     seg_main = seg_main.cpu()
#     cls_logit = cls_logit.cpu()
#     edge = edge.cpu()
#     edge_out = edge_out.cpu()
#     recon = recon.cpu()                     # GT 重建目标 = 原图
#     recon_out = recon_out.cpu().clamp(0, 1) # 预测重建

#     # ------------------- 2. 预测 -------------------
#     cls_pred = torch.argmax(cls_logit, dim=1)  # [B]
#     pred_seg = (torch.sigmoid(seg_main) > 0.5).float()
#     pred_edge = (torch.sigmoid(edge_out) > 0.5).float()

#     to_pil = T.ToPILImage()
#     img_size = img.size(2)  # H = W

#     # ------------------- 3. 画布（7 列） -------------------
#     col_count = 7
#     canvas_w = 80 + img_size * col_count
#     canvas_h = 140 + img_size
#     canvas = Image.new('RGB', (canvas_w, canvas_h), (255, 255, 255))
#     draw = ImageDraw.Draw(canvas)

#     # ------------------- 4. 字体 -------------------
#     try:
#         font_title = ImageFont.truetype("arial.ttf", 26)
#         font_text = ImageFont.truetype("arial.ttf", 20)
#     except:
#         try:
#             font_title = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 26)
#             font_text = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 20)
#         except:
#             font_title = ImageFont.load_default()
#             font_text = ImageFont.load_default()

#     # ------------------- 5. 标题 -------------------
#     titles = ["Input", "GT Mask", "Pred Seg", "GT Edge", "Pred Edge", "GT Recon", "Pred Recon"]
#     for idx, title in enumerate(titles):
#         x = 40 + idx * img_size + img_size // 2
#         bbox = draw.textbbox((0, 0), title, font=font_title)
#         text_w = bbox[2] - bbox[0]
#         draw.text((x - text_w // 2, 15), title, fill=(0, 0, 0), font=font_title)

#     # ------------------- 6. 图像处理 & 粘贴 -------------------
#     x_base = 40
#     y_base = 70

#     # --- Input + CLAHE 增强 ---
#     input_raw = (img[0] * 0.5 + 0.5).clamp(0, 1)  # 反归一化
#     input_np = (input_raw.squeeze(0).numpy() * 255).astype(np.uint8)
#     clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
#     enhanced = clahe.apply(input_np)
#     enhanced_tensor = torch.from_numpy(enhanced).float().unsqueeze(0) / 255.0
#     canvas.paste(to_pil(enhanced_tensor), (x_base, y_base))

#     # --- 其余 6 张图 ---
#     imgs_to_paste = [
#         mask[0].squeeze(0).unsqueeze(0),
#         pred_seg[0].squeeze(0).unsqueeze(0),
#         edge[0].squeeze(0).unsqueeze(0),      # GT Edge（已加粗）
#         pred_edge[0].squeeze(0).unsqueeze(0),
#         recon[0].squeeze(0).unsqueeze(0),     # GT Recon
#         recon_out[0].squeeze(0).unsqueeze(0)  # Pred Recon
#     ]

#     for i, tensor in enumerate(imgs_to_paste):
#         pil_img = to_pil(tensor.clamp(0, 1))
#         canvas.paste(pil_img, (x_base + (i + 1) * img_size, y_base))

#     # ------------------- 7. 分类文字 -------------------
#     y_text = y_base + img_size + 15
#     pred_label = class_names[cls_pred[0].item()]
#     gt_label = class_names[cls_gt[0].item()]
#     correct = cls_pred[0].item() == cls_gt[0].item()

#     pred_color = (0, 180, 0) if correct else (200, 0, 0)  # 深绿 / 深红
#     draw.text((x_base + 10, y_text), f"Pred: {pred_label}", fill=pred_color, font=font_text)
#     draw.text((x_base + 10, y_text + 28), f"GT: {gt_label}", fill=(0, 0, 0), font=font_text)

#     # ------------------- 8. 保存 -------------------
#     if save_path:
#         canvas.save(save_path)
#         print(f" [Vis] Saved: {save_path} | Pred: {pred_label} | GT: {gt_label}")

#     return canvas


# utils/tools.py
# import torch
# from PIL import Image, ImageDraw, ImageFont
# import torchvision.transforms as T
# import cv2
# import numpy as np


# def visualize_batch(
#     img, mask, cls_gt, edge, recon,
#     seg_main, cls_logit, edge_out, recon_out,
#     class_names=['Normal', 'Benign', 'Malignant'],
#     save_path=None
# ):
#     # ------------------- 1. 转 CPU -------------------
#     img = img.cpu()
#     mask = mask.cpu()
#     cls_gt = cls_gt.cpu()
#     seg_main = seg_main.cpu()
#     cls_logit = cls_logit.cpu()
#     edge = edge.cpu()
#     edge_out = edge_out.cpu()
#     recon = recon.cpu()
#     recon_out = recon_out.cpu().clamp(0, 1)

#     # ------------------- 2. 预测 -------------------
#     cls_pred = torch.argmax(cls_logit, dim=1)
#     pred_seg = (torch.sigmoid(seg_main) > 0.5).float()
#     pred_edge = (torch.sigmoid(edge_out) > 0.5).float()

#     to_pil = T.ToPILImage()
#     img_size = img.size(2)

#     # ------------------- 3. 画布（7 列） -------------------
#     col_count = 7
#     canvas_w = 80 + img_size * col_count
#     canvas_h = 140 + img_size
#     canvas = Image.new('RGB', (canvas_w, canvas_h), (255, 255, 255))
#     draw = ImageDraw.Draw(canvas)

#     # ------------------- 4. 字体 -------------------
#     try:
#         font_title = ImageFont.truetype("arial.ttf", 26)
#         font_text = ImageFont.truetype("arial.ttf", 20)
#     except:
#         try:
#             font_title = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 26)
#             font_text = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 20)
#         except:
#             font_title = ImageFont.load_default()
#             font_text = ImageFont.load_default()

#     # ------------------- 5. 标题 -------------------
#     titles = ["Input", "GT Mask", "Pred Seg", "GT Edge", "Pred Edge", "GT Recon", "Pred Recon"]
#     for idx, title in enumerate(titles):
#         x = 40 + idx * img_size + img_size // 2
#         bbox = draw.textbbox((0, 0), title, font=font_title)
#         text_w = bbox[2] - bbox[0]
#         draw.text((x - text_w // 2, 15), title, fill=(0, 0, 0), font=font_title)

#     # ------------------- 6. 图像处理 & 粘贴 -------------------
#     x_base = 40
#     y_base = 70

#     # --- Input + CLAHE ---
#     input_raw = (img[0] * 0.5 + 0.5).clamp(0, 1)
#     input_np = (input_raw.squeeze(0).numpy() * 255).astype(np.uint8)
#     clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
#     enhanced = clahe.apply(input_np)
#     enhanced_tensor = torch.from_numpy(enhanced).float().unsqueeze(0) / 255.0
#     canvas.paste(to_pil(enhanced_tensor), (x_base, y_base))

#     # --- 其余 6 张 ---
#     imgs_to_paste = [
#         mask[0].squeeze(0).unsqueeze(0),
#         pred_seg[0].squeeze(0).unsqueeze(0),
#         edge[0].squeeze(0).unsqueeze(0),      # GT Edge（已加粗）
#         pred_edge[0].squeeze(0).unsqueeze(0),
#         recon[0].squeeze(0).unsqueeze(0),     # GT Recon
#         recon_out[0].squeeze(0).unsqueeze(0)  # Pred Recon
#     ]

#     for i, tensor in enumerate(imgs_to_paste):
#         pil_img = to_pil(tensor.clamp(0, 1))
#         canvas.paste(pil_img, (x_base + (i + 1) * img_size, y_base))

#     # ------------------- 7. 分类文字 -------------------
#     y_text = y_base + img_size + 15
#     pred_label = class_names[cls_pred[0].item()]
#     gt_label = class_names[cls_gt[0].item()]
#     correct = cls_pred[0].item() == cls_gt[0].item()

#     pred_color = (0, 180, 0) if correct else (200, 0, 0)
#     draw.text((x_base + 10, y_text), f"Pred: {pred_label}", fill=pred_color, font=font_text)
#     draw.text((x_base + 10, y_text + 28), f"GT: {gt_label}", fill=(0, 0, 0), font=font_text)

#     # ------------------- 8. 保存 -------------------
#     if save_path:
#         canvas.save(save_path)
#         print(f" [Vis] Saved: {save_path} | Pred: {pred_label} | GT: {gt_label}")

#     return canvas



# utils/tools.py
import torch
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T
import cv2
import numpy as np


# def visualize_batch(
#     img, mask, cls_gt, edge, recon,
#     seg_main, cls_logit, edge_out, recon_out,
#     class_names=['Normal', 'Benign', 'Malignant'],
#     save_path=None
# ):
#     # ------------------- 1. 转 CPU -------------------
#     img = img.cpu()
#     mask = mask.cpu()
#     cls_gt = cls_gt.cpu()
#     seg_main = seg_main.cpu()
#     cls_logit = cls_logit.cpu()
#     edge = edge.cpu()
#     edge_out = edge_out.cpu()
#     recon = recon.cpu()
#     recon_out = recon_out.cpu().clamp(0, 1)
    

#     # ------------------- 2. 预测 -------------------
#     cls_pred = torch.argmax(cls_logit, dim=1)
#     pred_seg = (torch.sigmoid(seg_main) > 0.5).float()
#     pred_edge = (torch.sigmoid(edge_out) > 0.5).float()

#     to_pil = T.ToPILImage()
#     img_size = img.size(2)

#     # ------------------- 3. 画布（7 列） -------------------
#     col_count = 7
#     canvas_w = 80 + img_size * col_count
#     canvas_h = 140 + img_size
#     canvas = Image.new('RGB', (canvas_w, canvas_h), (255, 255, 255))
#     draw = ImageDraw.Draw(canvas)

#     # ------------------- 4. 字体 -------------------
#     try:
#         font_title = ImageFont.truetype("arial.ttf", 26)
#         font_text = ImageFont.truetype("arial.ttf", 20)
#     except:
#         try:
#             font_title = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 26)
#             font_text = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 20)
#         except:
#             font_title = ImageFont.load_default()
#             font_text = ImageFont.load_default()

#     # ------------------- 5. 标题 -------------------
#     titles = ["Input", "GT Mask", "Pred Seg", "GT Edge", "Pred Edge", "GT Recon", "Pred Recon"]
#     for idx, title in enumerate(titles):
#         x = 40 + idx * img_size + img_size // 2
#         bbox = draw.textbbox((0, 0), title, font=font_title)
#         text_w = bbox[2] - bbox[0]
#         draw.text((x - text_w // 2, 15), title, fill=(0, 0, 0), font=font_title)

#     # ------------------- 6. 图像处理 & 粘贴 -------------------
#     x_base = 40
#     y_base = 70

#     # --- Input + CLAHE ---
#     input_raw = (img[0] * 0.5 + 0.5).clamp(0, 1)
#     input_np = (input_raw.squeeze(0).numpy() * 255).astype(np.uint8)
#     clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
#     enhanced = clahe.apply(input_np)
#     enhanced_tensor = torch.from_numpy(enhanced).float().unsqueeze(0) / 255.0
#     canvas.paste(to_pil(enhanced_tensor), (x_base, y_base))

#     # --- 其余 6 张 ---
#     imgs_to_paste = [
#         mask[0].squeeze(0).unsqueeze(0),
#         pred_seg[0].squeeze(0).unsqueeze(0),
#         edge[0].squeeze(0).unsqueeze(0),      # GT Edge（已加粗）
#         pred_edge[0].squeeze(0).unsqueeze(0),
#         recon[0].squeeze(0).unsqueeze(0),     # GT Recon
#         recon_out[0].squeeze(0).unsqueeze(0)  # Pred Recon
#     ]

#     for i, tensor in enumerate(imgs_to_paste):
#         pil_img = to_pil(tensor.clamp(0, 1))
#         canvas.paste(pil_img, (x_base + (i + 1) * img_size, y_base))

#     # ------------------- 7. 分类文字 -------------------
#     y_text = y_base + img_size + 15
#     pred_label = class_names[cls_pred[0].item()]
#     gt_label = class_names[cls_gt[0].item()]
#     correct = cls_pred[0].item() == cls_gt[0].item()

#     pred_color = (0, 180, 0) if correct else (200, 0, 0)
#     draw.text((x_base + 10, y_text), f"Pred: {pred_label}", fill=pred_color, font=font_text)
#     draw.text((x_base + 10, y_text + 28), f"GT: {gt_label}", fill=(0, 0, 0), font=font_text)

#     # ------------------- 8. 保存 -------------------
#     if save_path:
#         canvas.save(save_path)
#         print(f" [Vis] Saved: {save_path} | Pred: {pred_label} | GT: {gt_label}")

#     return canvas



def visualize_batch(
    img, mask, cls_gt, edge, recon,
    seg_main, cls_logit, edge_out, recon_out,
    class_names=['Normal', 'Benign', 'Malignant'],
    save_path=None
):
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    import torch
    import torchvision.transforms as T

    to_pil = T.ToPILImage()

    # ------------------- 1. 安全转 CPU（关键修复！）-------------------
    img       = img.cpu()
    mask      = mask.cpu()
    seg_main  = seg_main.cpu()
    edge      = edge.cpu()
    edge_out  = edge_out.cpu()
    recon     = recon.cpu()
    recon_out = recon_out.cpu().clamp(0, 1)

    # 关键修复：cls_logit 可能是 int，也可能是 tensor
    if isinstance(cls_logit, torch.Tensor):
        cls_pred = torch.argmax(cls_logit, dim=1).cpu()           # [B]
    else:
        cls_pred = torch.tensor([int(cls_logit)]).cpu()           # 你传的是 int

    if isinstance(cls_gt, torch.Tensor):
        cls_gt_vis = cls_gt.cpu()
    else:
        cls_gt_vis = torch.tensor([int(cls_gt)]).cpu()

    # ------------------- 2. 预测结果 -------------------
    pred_seg  = (torch.sigmoid(seg_main) > 0.5).float()
    pred_edge = (torch.sigmoid(edge_out) > 0.5).float()

    img_size = img.size(2)

    # ------------------- 3. 画布 -------------------
    col_count = 7
    canvas_w = 80 + img_size * col_count
    canvas_h = 140 + img_size
    canvas = Image.new('RGB', (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # ------------------- 4. 字体（防炸）-------------------
    try:
        font_title = ImageFont.truetype("arial.ttf", 26)
        font_text  = ImageFont.truetype("arial.ttf", 20)
    except:
        font_title = font_text = ImageFont.load_default()

    # ------------------- 5. 标题 -------------------
    titles = ["Input", "GT Mask", "Pred Seg", "GT Edge", "Pred Edge", "GT Recon", "Pred Recon"]
    for idx, title in enumerate(titles):
        x = 40 + idx * img_size + img_size // 2
        bbox = draw.textbbox((0, 0), title, font=font_title)
        w = bbox[2] - bbox[0]
        draw.text((x - w // 2, 15), title, fill=(0, 0, 0), font=font_title)

    # ------------------- 6. 图像粘贴 -------------------
    x_base, y_base = 40, 70

    # Input + CLAHE 增强
    input_raw = (img[0] * 0.5 + 0.5).clamp(0, 1)
    input_np = (input_raw.squeeze(0).numpy() * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(input_np)
    enhanced_tensor = torch.from_numpy(enhanced / 255.0).unsqueeze(0)
    canvas.paste(to_pil(enhanced_tensor), (x_base, y_base))

    # 其他图像
    imgs_to_paste = [
        mask[0].squeeze(0).unsqueeze(0),
        pred_seg[0].squeeze(0).unsqueeze(0),
        edge[0].squeeze(0).unsqueeze(0),
        pred_edge[0].squeeze(0).unsqueeze(0),
        recon[0].squeeze(0).unsqueeze(0),
        recon_out[0].squeeze(0).unsqueeze(0),
    ]

    for i, tensor in enumerate(imgs_to_paste):
        pil_img = to_pil(tensor.clamp(0, 1).repeat(3, 1, 1) if tensor.size(0) == 1 else tensor)
        canvas.paste(pil_img, (x_base + (i + 1) * img_size, y_base))

    # ------------------- 7. 分类文字 -------------------
    pred_label = class_names[cls_pred[0].item()]
    gt_label   = class_names[cls_gt_vis[0].item()]
    correct = cls_pred[0].item() == cls_gt_vis[0].item()
    color = (0, 180, 0) if correct else (200, 0, 0)

    y_text = y_base + img_size + 15
    draw.text((x_base + 10, y_text),     f"Pred: {pred_label}", fill=color,      font=font_text)
    draw.text((x_base + 10, y_text + 28), f"GT:   {gt_label}",   fill=(0,0,0),     font=font_text)

    # ------------------- 8. 保存 -------------------
    if save_path:
        canvas.save(save_path)
        status = "Correct" if correct else "Wrong"
        print(f" [Vis] Saved: {save_path} | {status} Pred: {pred_label} ← GT: {gt_label}")

    return canvas