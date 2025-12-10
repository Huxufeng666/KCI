import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """双卷积块：Conv -> BN -> ReLU x2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=1, base_c=64):
        """
        U-Net 基础框架
        Args:
            in_channels (int): 输入图像通道数（如灰度图=1，RGB=3）
            num_classes (int): 输出类别数（二值分割=1，多类=N）
            base_c (int): 初始特征通道数
        """
        super().__init__()
        
        # Encoder (下采样路径)
        self.enc1 = ConvBlock(in_channels, base_c)           # -> [B, 64, H, W]
        self.pool1 = nn.MaxPool2d(2)                         # -> [B, 64, H/2, W/2]
        
        self.enc2 = ConvBlock(base_c, base_c*2)              # -> [B, 128, H/2, W/2]
        self.pool2 = nn.MaxPool2d(2)                         # -> [B, 128, H/4, W/4]
        
        self.enc3 = ConvBlock(base_c*2, base_c*4)            # -> [B, 256, H/4, W/4]
        self.pool3 = nn.MaxPool2d(2)                         # -> [B, 256, H/8, W/8]
        
        self.enc4 = ConvBlock(base_c*4, base_c*8)            # -> [B, 512, H/8, W/8]
        self.pool4 = nn.MaxPool2d(2)                         # -> [B, 512, H/16, W/16]
        
        # Bottleneck (瓶颈层)
        self.bottleneck = ConvBlock(base_c*8, base_c*16)     # -> [B, 1024, H/16, W/16]
        
        # Decoder (上采样路径)
        self.up4 = nn.ConvTranspose2d(base_c*16, base_c*8, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(base_c*16, base_c*8)           # 1024 -> 512 (cat 后)
        
        self.up3 = nn.ConvTranspose2d(base_c*8, base_c*4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(base_c*8, base_c*4)
        
        self.up2 = nn.ConvTranspose2d(base_c*4, base_c*2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_c*4, base_c*2)
        
        self.up1 = nn.ConvTranspose2d(base_c*2, base_c, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_c*2, base_c)
        
        # 最终输出层
        self.final_conv = nn.Conv2d(base_c, num_classes, kernel_size=1)
        
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)           # [B, 64, H, W]
        e2 = self.enc2(self.pool1(e1))  # [B, 128, H/2, W/2]
        e3 = self.enc3(self.pool2(e2))  # [B, 256, H/4, W/4]
        e4 = self.enc4(self.pool3(e3))  # [B, 512, H/8, W/8]
        
        # Bottleneck
        b = self.bottleneck(self.pool4(e4))  # [B, 1024, H/16, W/16]
        
        # Decoder + skip connections
        d4 = self.up4(b)                        # 上采样 -> [B, 512, H/8, W/8]
        d4 = torch.cat([d4, e4], dim=1)          # 拼接 skip -> [B, 1024, H/8, W/8]
        d4 = self.dec4(d4)                      # -> [B, 512, H/8, W/8]
        
        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        # 输出 logits
        logits = self.final_conv(d1)            # [B, num_classes, H, W]
        return  logits


