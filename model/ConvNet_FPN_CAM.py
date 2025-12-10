

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.ops import FeaturePyramidNetwork
import timm


# ==============================================================
# 1. CBAM（Channel + Spatial Attention）—— 替代 ASFBlock
# ==============================================================
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


# ==============================================================
# 2. ConvNeXt-Tiny Encoder（1通道输入，预训练）
# ==============================================================
class ConvNeXtEncoder(nn.Module):
    def __init__(self, in_channels=1, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            'convnext_tiny',
            pretrained=pretrained,
            in_chans=in_channels,
            features_only=True,
            out_indices=(0, 1, 2, 3)   # 关键：返回 4 层！
        )

        # 修复：单通道 + 预训练 → 权重取平均
        if in_channels == 1 and pretrained:
            for module in self.backbone.modules():
                if isinstance(module, nn.Conv2d) and module.in_channels == 3:
                    with torch.no_grad():
                        w = module.weight
                        w_mean = w.mean(dim=1, keepdim=True)
                        module.weight = nn.Parameter(w_mean)
                        module.in_channels = 1
                    break

    def forward(self, x):
        feats = self.backbone(x)  # len(feats) == 4
        return {
            'p2': feats[0],  # H/4,  96
            'p3': feats[1],  # H/8,  192
            'p4': feats[2],  # H/16, 384
            'p5': feats[3],  # H/32, 768
        }

# ==============================================================
# 3. FPN（统一通道 + 上采样融合）
# ==============================================================
class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels=256):
        super().__init__()
        # FPN 主干：输入原始通道，输出统一 out_channels
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,  # [96,192,384,768]
            out_channels=out_channels           # 256
        )
        # 3×3 平滑层（在 FPN 之后）
        self.smooth_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in in_channels_list
        ])

    def forward(self, feats):
        # 固定顺序：p2 -> p5
        names = ['p2', 'p3', 'p4', 'p5']
        
        # 直接传入原始特征（未降维）
        laterals = {name: feats[name] for name in names}
        
        # FPN 主干：自动做上采样 + 相加
        fpn_out = self.fpn(laterals)  # dict: p2~p5, 256ch

        # 3×3 平滑
        out = {}
        for name, conv in zip(names, self.smooth_convs):
            out[name] = conv(fpn_out[name])

        return out  # {'p2':256, 'p3':256, ...}


# ==============================================================
# 4. 分割头（保持不变）
# ==============================================================
class SegmentationHead(nn.Module):
    def __init__(self, in_channels, num_classes=1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels // 4, 3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, num_classes, 1)
        )

    def forward(self, x):
        return self.head(x)


# ==============================================================
# 5. 完整模型（ConvNeXt + FPN + CBAM）
# ==============================================================
class EndToEndModel(nn.Module):
    def __init__(self, in_channels=1, num_classes=1, pretrained_backbone=True):
        super().__init__()
        self.encoder = ConvNeXtEncoder(in_channels=in_channels, pretrained=pretrained_backbone)

        # FPN 输入：原始通道数
        fpn_in_channels = [96, 192, 384, 768]  # p2~p5
        self.fpn = FPN(fpn_in_channels, out_channels=256)

        self.cbam = CBAM(channels=256)
        self.seg_head = SegmentationHead(in_channels=256, num_classes=num_classes)
        self.aux_head = SegmentationHead(in_channels=256, num_classes=num_classes)

    def forward(self, images, masks=None):
        processed_image = images

        # 1. Encoder
        feats = self.encoder(images)  # {'p2':96, 'p3':192, 'p4':384, 'p5':768}

        # 2. FPN（输入原始通道）
        fpn_feats = self.fpn(feats)   # {'p2':256, 'p3':256, ...}

        # 3. 主特征 + CBAM
        main_feat = fpn_feats['p2']
        fused = self.cbam(main_feat)

        # 4. 分割
        logit = self.seg_head(fused)

        
        logits = F.interpolate(
                logit,
                size=images.shape[2:],            # (H, W)
                mode='bilinear',
                align_corners=False
            )
        
        if self.training:
            aux_feat = fpn_feats['p3']
            aux_logits = self.aux_head(aux_feat)
            aux_logits = F.interpolate(aux_logits, size=logits.shape[2:], mode='bilinear', align_corners=False)
            return processed_image, logits#, aux_logits
        else:
            return processed_image, logits

# ==============================================================
# 6. PatchGAN 判别器（保持不变）
# ==============================================================
class Discriminator(nn.Module):
    def __init__(self, in_channels=1, base_channels=32):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels * 4, base_channels * 8, 4, 1, 1),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels * 8, 1, 4, 1, 1)  # logits
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        return self.model(x)
    
    
# model = EndToEndModel(in_channels=1, pretrained_backbone=True)
# x = torch.randn(1, 1, 256, 256)
# processed, logits,*_ = model(x)
# print("Output shape:", logits.shape)  # 应输出 [1,1,256,256]