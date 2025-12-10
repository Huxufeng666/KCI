import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models






# --------------------------------------------------------------
# 1. CBAM（Channel + Spatial Attention）
# --------------------------------------------------------------
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x_cat))


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(spatial_kernel)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x




class ASFBlock(nn.Module):
    """Adaptive Spatial Fusion (simple attention-based fusion).
    Takes three feature maps (all spatially aligned) and returns a fused feature.
    """
    def __init__(self, in_channels, mid_channels=None):
        super().__init__()
        if mid_channels is None:
            mid_channels = in_channels // 2

        self.reduce = nn.Conv2d(in_channels * 3, mid_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(mid_channels)
        self.conv = nn.Conv2d(mid_channels, in_channels, kernel_size=3, padding=1)
        self.attn = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, f1, f2, f3):
        # f1, f2, f3: (B, C, H, W) - already spatially aligned
        x = torch.cat([f1, f2, f3], dim=1)
        x = self.reduce(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.conv(x)

        # spatial channel attention
        w = self.attn(x)
        out = x * w + x
        return out


class SegmentationHead(nn.Module):
    def __init__(self, in_channels, num_classes=1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, num_classes, kernel_size=1)
        )

    def forward(self, x):
        return self.head(x)

class BackboneEncoder(nn.Module):
    """ResNet‑50 encoder, 1‑channel input → f1(256), f2(512), f3(1024)"""
    def __init__(self, in_channels=1, pretrained=True):
        super().__init__()
        resnet = models.resnet50(weights='ResNet50_Weights.DEFAULT' if pretrained else None)

        # ---------- 1. 替换第一层卷积 ----------
        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        if pretrained:
            # 用预训练权重的 RGB 均值初始化单通道权重
            with torch.no_grad():
                self.conv1.weight = nn.Parameter(
                    resnet.conv1.weight.mean(dim=1, keepdim=True)   # [64,1,7,7]
                )

        # ---------- 2. 重新构建 stem（使用我们自己的 conv1） ----------
        self.bn1     = resnet.bn1
        self.relu    = resnet.relu
        self.maxpool = resnet.maxpool

        self.stem = nn.Sequential(
            self.conv1,   # ← 关键：这里使用 1‑channel conv
            self.bn1,
            self.relu,
            self.maxpool
        )

        # ---------- 3. 其余层保持不变 ----------
        self.layer1 = resnet.layer1   # C=256
        self.layer2 = resnet.layer2   # C=512
        self.layer3 = resnet.layer3   # C=1024
        self.layer4 = resnet.layer4   # C=2048

        # ---------- 4. 投影到目标通道数 ----------
        self.project2 = nn.Conv2d(512,  256,  kernel_size=1)   # layer2 → f1
        self.project3 = nn.Conv2d(1024, 512,  kernel_size=1)   # layer3 → f2
        self.project4 = nn.Conv2d(2048, 1024, kernel_size=1)   # layer4 → f3

    def forward(self, x):
        x = self.stem(x)          # ← 1‑channel → 64
        x = self.layer1(x)        # C=256
        l2 = self.layer2(x)       # C=512
        l3 = self.layer3(l2)      # C=1024
        l4 = self.layer4(l3)      # C=2048

        f1 = self.project2(l2)    # 256
        f2 = self.project3(l3)    # 512
        f3 = self.project4(l4)    # 1024
        return f1, f2, f3



class EndToEndModel(nn.Module):
    """End-to-end model that returns (processed_image, segmentation_logits).

    The segmentation branch follows the described architecture:
      - Backbone -> f1 (H/4,256), f2 (H/8,512), f3 (H/16,1024)
      - Upsample f2 and f3 to H/4
      - ASF fusion
      - Segmentation head -> logits (1 channel)
    """
    def __init__(self, in_channels=1, num_classes=1, pretrained_backbone=True):
        super().__init__()
        self.encoder = BackboneEncoder(in_channels=in_channels, pretrained=pretrained_backbone)

        # upsample layers to bring f2/f3 to f1 spatial size
        self.up2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.up3 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.asf = ASFBlock(in_channels=256, mid_channels=128)
        self.seg_head = SegmentationHead(in_channels=256, num_classes=num_classes)

    def forward(self, images, masks=None):
        # images: (B,3,H,W)
        # processed_image: for now we return the input image as "processed"
        processed_image = images

        f1, f2, f3 = self.encoder(images)

        # Upsample f2 (H/8 -> H/4) and f3 (H/16 -> H/4)
        # compute target spatial size from f1
        target_size = f1.shape[2:]

        f2_up = F.interpolate(self.up2(f2), size=target_size, mode='bilinear', align_corners=False)
        f3_up = F.interpolate(self.up3(f3), size=target_size, mode='bilinear', align_corners=False)

        # Ensure f1 has same channel dim (256)
        if f1.shape[1] != 256:
            f1 = nn.Conv2d(f1.shape[1], 256, kernel_size=1).to(f1.device)(f1)

        fused = self.asf(f1, f2_up, f3_up)
        segmentation_logits = self.seg_head(fused)

        return segmentation_logits


class Discriminator(nn.Module):
    """A small PatchGAN-like discriminator for segmentation maps.
    Accepts single-channel or multi-channel maps and returns logits.
    """
    def __init__(self, in_channels=1, base_channels=32):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels * 8, 1, kernel_size=4, stride=1, padding=1)  # logits
        )

    def forward(self, x):
        # if input has sigmoid applied externally, it's fine - we return logits
        if x.dim() == 3:
            x = x.unsqueeze(1)
        return self.model(x)

