import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class CombinedLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, gamma=0.2):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha  # MSE loss权重
        self.beta = beta    # GAN loss权重
        self.gamma = gamma  # Edge loss权重
        self.bce = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()

    def forward(self, predictions, targets):
        # GAN loss (using BCE)
        gan_loss = self.bce(predictions, targets)
        
        # MSE loss
        mse_loss = self.mse(predictions, targets)
        
        # 组合损失
        total_loss = (self.alpha * mse_loss + 
                     self.beta * gan_loss)
        
        return total_loss

def compute_edge_from_mask(mask, kernel_size=3):
    """
    从mask中提取边缘信息
    Args:
        mask: 输入的mask张量 [B, 1, H, W]
        kernel_size: 卷积核大小，用于边缘检测
    Returns:
        边缘map
    """
    # 确保输入是4D张量
    if len(mask.shape) == 3:
        mask = mask.unsqueeze(1)
    
    # 创建边缘检测卷积核
    sobel_x = torch.tensor([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]], dtype=torch.float32).to(mask.device)
    sobel_y = torch.tensor([[-1, -2, -1],
                           [0, 0, 0],
                           [1, 2, 1]], dtype=torch.float32).to(mask.device)
    
    sobel_x = sobel_x.view(1, 1, 3, 3).repeat(1, 1, 1, 1)
    sobel_y = sobel_y.view(1, 1, 3, 3).repeat(1, 1, 1, 1)
    
    # 应用Sobel算子
    edge_x = F.conv2d(mask, sobel_x, padding=1)
    edge_y = F.conv2d(mask, sobel_y, padding=1)
    
    # 计算边缘强度
    edge = torch.sqrt(edge_x**2 + edge_y**2)
    
    # 归一化
    edge = edge / edge.max()
    
    return edge

def plot_losses(csv_path, save_path):
    """
    绘制训练过程中的损失曲线
    Args:
        csv_path: 训练日志CSV文件路径
        save_path: 图像保存路径
    """
    # 读取CSV文件
    df = pd.read_csv(csv_path)
    
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['generator_loss'], label='Generator Loss')
    plt.plot(df['epoch'], df['discriminator_loss'], label='Discriminator Loss')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(save_path)
    plt.close()

def plot_losses_2(csv_path, save_path):
    """
    绘制包含测试损失的训练过程损失曲线
    Args:
        csv_path: 训练日志CSV文件路径
        save_path: 图像保存路径
    """
    # 读取CSV文件
    df = pd.read_csv(csv_path)
    
    plt.figure(figsize=(12, 8))
    
    # 创建主坐标轴
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    # 在主坐标轴上绘制生成器和判别器损失
    ln1 = ax1.plot(df['epoch'], df['generator_loss'], 'b-', label='Generator Loss')
    ln2 = ax1.plot(df['epoch'], df['discriminator_loss'], 'r-', label='Discriminator Loss')
    
    # 在次坐标轴上绘制测试分割损失
    ln3 = ax2.plot(df['epoch'], df['Test Seg Loss'], 'g-', label='Test Seg Loss')
    
    # 设置标签和标题
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Generator/Discriminator Loss')
    ax2.set_ylabel('Test Segmentation Loss')
    plt.title('Training and Test Losses')
    
    # 合并图例
    lns = ln1 + ln2 + ln3
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='upper right')
    
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()