import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DPTForSemanticSegmentation, DPTImageProcessor
from typing import List


# ================== Reassemble Module (from DPT official) ==================
class ReassembleTokens(nn.Module):
    def __init__(self, embed_dim: int, patch_size: int = 16):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)

    def forward(self, x_tokens: torch.Tensor, img_size: tuple):
        """
        x_tokens: [B, N, C]  (N = H*W / p^2)
        img_size: (H, W)
        """
        B, N, C = x_tokens.shape
        H, W = img_size
        p = self.patch_size

        # assert H % p == 0 and W % p == 0, f"Image size {H}x{W} not divisible by patch size {p}"

        # h, w = H // p, W // p
        # x = x_tokens.permute(0, 2, 1).reshape(B, C, h, w)  # [B, C, h, w]
        # x = F.interpolate(x, size=(H // 4, W // 4), mode='bilinear', align_corners=False)
        # return self.proj(x)


            # 去掉 CLS token
        cls_token = x_tokens[:, 0:1, :]   # [B, 1, C]
        patch_tokens = x_tokens[:, 1:, :] # [B, 196, C]

        h, w = H // p, W // p
        assert h * w == patch_tokens.shape[1], f"{h*w} != {patch_tokens.shape[1]}"

        x = patch_tokens.permute(0, 2, 1).reshape(B, C, h, w)
        x = F.interpolate(x, size=(H // 4, W // 4), mode='bilinear', align_corners=False)
        return self.proj(x)


# ================== ASF Block (你的原版，优化过) ==================
class ASFBlock(nn.Module):
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
        x = torch.cat([f1, f2, f3], dim=1)
        x = F.relu(self.bn(self.reduce(x)))
        x = self.conv(x)
        w = self.attn(x)
        return x * w + x


# ================== Segmentation Head ==================
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


# ================== PatchGAN Discriminator ==================
class Discriminator(nn.Module):
    def __init__(self, in_channels=1, base_channels=32):
        super().__init__()
        def block(in_c, out_c, norm=True):
            layers = [nn.Conv2d(in_c, out_c, 4, 2, 1)]
            if norm:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(in_channels, base_channels, norm=False),
            *block(base_channels, base_channels * 2),
            *block(base_channels * 2, base_channels * 4),
            *block(base_channels * 4, base_channels * 8),
            nn.Conv2d(base_channels * 8, 1, 4, 1, 1)
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        return self.model(x)


# ================== 主模型：DPT + ASF + GAN ==================
class DPT_ASF_GAN(nn.Module):
    def __init__(self, num_classes=1, pretrained_dpt="Intel/dpt-large", in_channels=1):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # 1. 1→3 通道映射（医学图像）
        self.input_proj = nn.Conv2d(in_channels, 3, kernel_size=1)

        # 2. DPT 模型（只用 encoder + neck）
        self.dpt = DPTForSemanticSegmentation.from_pretrained(pretrained_dpt,num_labels=1, ignore_mismatched_sizes=True)
 
        self.dpt.classifier = nn.Identity()  # 移除原头
        for name,param in self.dpt.named_parameters():
            # param.requires_grad = True  # 建议先冻结，后面微调
            
            if "neck" in name or "seg_head" in name:
                param.requires_grad = True
            
            elif "encoder.layer." in name:  # 只看 encoder.layer.X
                parts = name.split(".")
                if parts[-2] == "layer":   # 确保是 layer.X
                    try:
                        layer_idx = int(parts[-1])
                        if layer_idx >= 9:
                            param.requires_grad = True
                    except ValueError:
                        pass  # 忽略非数字
            else:
                param.requires_grad = False
                    
            
            
            
        # for name, param in self.dpt.named_parameters():
        #     if int(name.split(".")[-1]) >= 9 in name or "neck" in name:
        #         param.requires_grad = True

        # 3. Hook 提取多层特征
        self.hooks = []
        self.features = []
        self.selected_layers = [2, 5, 8, 11]  # ViT 层索引
        self.embed_dim = self.dpt.config.hidden_size  # 768 or 1024

        def make_hook(idx):
            def hook(module, input, output):
                if idx in self.selected_layers:
                    self.features.append(output[0])  # [B, N, C]
            return hook

        for i, layer in enumerate(self.dpt.dpt.encoder.layer):
            hook = layer.register_forward_hook(make_hook(i))
            self.hooks.append(hook)

        # 4. Reassemble 模块（不同尺度）
        self.reassemblers = nn.ModuleList([
            ReassembleTokens(self.embed_dim, patch_size=16),  # H/4
            ReassembleTokens(self.embed_dim, patch_size=16),
            ReassembleTokens(self.embed_dim, patch_size=16),
            ReassembleTokens(self.embed_dim, patch_size=16),
        ])

        # 5. 投影层
        self.proj = nn.ModuleList([
            nn.Conv2d(self.embed_dim, 256, 1),
            nn.Conv2d(self.embed_dim, 512, 1),
            nn.Conv2d(self.embed_dim, 1024, 1),
        ])

        # 6. 上采样到 H/4
        self.up2 = nn.Sequential(nn.Conv2d(512, 256, 1), nn.BatchNorm2d(256), nn.ReLU())
        self.up3 = nn.Sequential(nn.Conv2d(1024, 256, 1), nn.BatchNorm2d(256), nn.ReLU())

        # 7. ASF + Head
        self.asf = ASFBlock(256)
        self.seg_head = SegmentationHead(256, num_classes)

        # 8. 判别器
        self.discriminator = Discriminator(in_channels=1)

    def forward(self, x, return_features=False):
        B, C, H, W = x.shape
        img_size = (H, W)

        # 1. 1→3 通道
        x_rgb = self.input_proj(x)  # [B,3,H,W]

        # 2. DPT 前向（触发 hook）
        # _ = self.dpt(x_rgb)
        # feats = self.features[-4:]  # 取最后4层
        # self.features.clear()

        # 关键：必须让 DPT 完整前向
        with torch.no_grad():
            _ = self.dpt(x_rgb)   # 触发所有 hook

        # 防御性编程：只取 min(len(self.features), 3)
        feats = self.features[-3:]  # 最多取 3 个
        self.features.clear()

        if len(feats) < 3:
            raise RuntimeError(f"Only got {len(feats)} features, expected 3. Check hooks!")



        # 3. Reassemble
        f_list = []
        for i, tokens in enumerate(feats):
            f = self.reassemblers[i](tokens, img_size)
            f = self.proj[i](f)
            f_list.append(f)

        # 假设 f_list[0] 是最高分辨率 (H/4)
        f1 = f_list[0]  # [B,256,H/4,W/4]
        f2 = f_list[1]  # [B,512,H/8,W/8] → up to H/4
        f3 = f_list[2]  # [B,1024,H/16,W/16] → up to H/4

        target_size = f1.shape[2:]
        f2_up = F.interpolate(self.up2(f2), size=target_size, mode='bilinear', align_corners=False)
        f3_up = F.interpolate(self.up3(f3), size=target_size, mode='bilinear', align_corners=False)

        # 4. ASF 融合
        fused = self.asf(f1, f2_up, f3_up)  # [B,256,H/4,W/4]

        # 5. 分割头
        logits = self.seg_head(fused)  # [B,1,H/4,W/4]
        logits_up = F.interpolate(logits, size=img_size, mode='bilinear', align_corners=False)

        if return_features:
            return logits_up, fused
        return logits_up

    def get_discriminator(self):
        return self.discriminator


# ================== 测试代码 ==================
if __name__ == "__main__":
    pass
    model = DPT_ASF_GAN(in_channels=1, num_classes=1)
    x = torch.randn(2, 1, 256, 256)
    logits = model(x)
    print("Logits shape:", logits.shape)  # [2, 1, 224, 224]

    # 测试判别器
    D = model.get_discriminator()
    fake = torch.sigmoid(logits)
    d_out = D(fake)
    print("D output shape:", d_out.shape)