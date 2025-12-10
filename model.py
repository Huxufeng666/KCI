import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


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
    """ResNet-based encoder that returns layer2, layer3, layer4 features.
    The outputs are:
      - f1: layer2 output (C=256) with stride 4
      - f2: layer3 output (C=512) with stride 8
      - f3: layer4 output (C=1024) with stride 16
    """
    def __init__(self, in_channels=1, pretrained=True):
        super().__init__()
        resnet = models.resnet50(pretrained=pretrained)
        
        # 修改第一层卷积以接受单通道输入
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # 如果使用预训练模型，需要调整第一层权重
        if pretrained:
            # 对于单通道输入，我们取预训练权重的平均值
            pretrained_weight = resnet.conv1.weight.data
            self.conv1.weight.data = pretrained_weight.mean(dim=1, keepdim=True)

        # Keep initial layers
        self.stem = nn.Sequential(
            self.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )

        # ResNet layers
        self.layer1 = resnet.layer1  # stride 4 -> usually keeps C=256
        self.layer2 = resnet.layer2  # C=512 for resnet50
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # To match the channel sizes expected in the design
        self.project2 = nn.Conv2d(512, 256, kernel_size=1)
        self.project3 = nn.Conv2d(1024, 512, kernel_size=1)
        self.project4 = nn.Conv2d(2048, 1024, kernel_size=1)

    def forward(self, x):
        x = self.stem(x)
        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)

        f1 = self.project2(l2)
        f2 = self.project3(l3)
        f3 = self.project4(l4)

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
        # images: (B,C,H,W)
        processed_image = images

        f1, f2, f3 = self.encoder(images)

        # Upsample f2 (H/8 -> H/4) and f3 (H/16 -> H/4)
        target_size = f1.shape[2:]

        f2_up = F.interpolate(self.up2(f2), size=target_size, mode='bilinear', align_corners=False)
        f3_up = F.interpolate(self.up3(f3), size=target_size, mode='bilinear', align_corners=False)

        # Ensure f1 has same channel dim (256)
        if f1.shape[1] != 256:
            f1 = nn.Conv2d(f1.shape[1], 256, kernel_size=1).to(f1.device)(f1)

        fused = self.asf(f1, f2_up, f3_up)
        segmentation_logits = self.seg_head(fused)

        # Upsample segmentation to original image size if needed
        if segmentation_logits.shape[2:] != images.shape[2:]:
            segmentation_logits = F.interpolate(
                segmentation_logits, 
                size=images.shape[2:], 
                mode='bilinear', 
                align_corners=False
            )

        return processed_image, segmentation_logits


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