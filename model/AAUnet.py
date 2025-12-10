# model/aau_net.py
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================== 基础 Conv Block ====================
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, activation=nn.ReLU):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            activation(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            activation(inplace=True)
        )
    def forward(self, x):
        return self.block(x)


# ==================== Attention-Augmented Conv (AAC) ====================
class AAC(nn.Module):
    """
    注意力增强卷积（Channel + Spatial Attention）
    参考论文：https://pmc.ncbi.nlm.nih.gov/articles/PMC9311338/
    """
    def __init__(self, in_ch, out_ch, reduction=16):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

        # Channel Attention
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_ch, out_ch // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch // reduction, out_ch, 1, bias=False),
            nn.Sigmoid()
        )

        # Spatial Attention
        self.sa = nn.Sequential(
            nn.Conv2d(out_ch, 1, 7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)

        # Channel Attention
        ca = self.ca(out)
        out = out * ca

        # Spatial Attention
        sa = self.sa(out)
        out = out * sa

        return out


# ==================== AAU-Net 主干 ====================
class AAUNet(nn.Module):
    def __init__(self, in_chans=1, num_classes=1):
        super().__init__()
        filters = [64, 128, 256, 512, 1024]

        # Encoder
        self.enc1 = ConvBlock(in_chans, filters[0])
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(filters[0], filters[1])
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(filters[1], filters[2])
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ConvBlock(filters[2], filters[3])
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck: AAC
        self.bottleneck = AAC(filters[3], filters[4])

        # Decoder
        self.up4 = nn.ConvTranspose2d(filters[4], filters[3], 2, stride=2)
        self.dec4 = ConvBlock(filters[3] * 2, filters[3])
        self.up3 = nn.ConvTranspose2d(filters[3], filters[2], 2, stride=2)
        self.dec3 = ConvBlock(filters[2] * 2, filters[2])
        self.up2 = nn.ConvTranspose2d(filters[2], filters[1], 2, stride=2)
        self.dec2 = ConvBlock(filters[1] * 2, filters[1])
        self.up1 = nn.ConvTranspose2d(filters[1], filters[0], 2, stride=2)
        self.dec1 = ConvBlock(filters[0] * 2, filters[0])

        # Head
        self.head = nn.Conv2d(filters[0], num_classes, 1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)           # 256
        e2 = self.enc2(self.pool1(e1))  # 128
        e3 = self.enc3(self.pool2(e2))  # 64
        e4 = self.enc4(self.pool3(e3))  # 32

        # Bottleneck
        b = self.bottleneck(self.pool4(e4))  # 16

        # Decoder + Skip
        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        out = self.head(d1)
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
        return out

# # ==================== 测试 ====================
# if __name__ == "__main__":
#     model = AAUNet(in_chans=1, num_classes=1)
#     x = torch.randn(1, 1, 256, 256)
#     _,out = model(x)
#     print(f"Input: {x.shape} → Output: {out.shape}")  # [1, 1, 256, 256]