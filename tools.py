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