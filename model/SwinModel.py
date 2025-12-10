
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
# ================== Swin-T Backbone ==================
class SwinBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=pretrained,
            features_only=True,
            out_indices=(1, 2, 3)  # stage2,3,4
        )
        # 输出通道: [192, 384, 768]
        self.proj = nn.ModuleList([
            nn.Conv2d(192, 256, 1),
            nn.Conv2d(384, 512, 1),
            nn.Conv2d(768, 1024, 1),
        ])

    def forward(self, x):
        feats = self.backbone(x)  # List of feature tensors (expected [B, C, H, W])
        out = []
        for proj, f in zip(self.proj, feats):
            # Some timm/backbone versions may return channel-last tensors [B, H, W, C].
            # Ensure input to Conv2d is channel-first [B, C, H, W].
            if f.ndim == 4 and f.shape[1] != proj.in_channels:
                # if last dim matches expected in_channels, permute
                if f.shape[-1] == proj.in_channels:
                    f = f.permute(0, 3, 1, 2).contiguous()
            out.append(proj(f))
        return out

# ================== ASF + 深监督头 ==================
class ASFBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.reduce = nn.Conv2d(c*3, c, 1)
        self.conv = nn.Conv2d(c, c, 3, padding=1)
        self.bn = nn.BatchNorm2d(c)
        self.attn = nn.Sequential(
            nn.Conv2d(c, c//4, 1), nn.ReLU(), nn.Conv2d(c//4, c, 1), nn.Sigmoid()
        )
    def forward(self, f1, f2, f3):
        x = torch.cat([f1, f2, f3], dim=1)
        x = F.relu(self.bn(self.reduce(x)))
        x = self.conv(x)
        return x * self.attn(x) + x

class SegHead(nn.Module):
    def __init__(self, in_c, out_c=1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_c, in_c//2, 3, padding=1),
            nn.BatchNorm2d(in_c//2), nn.ReLU(),
            nn.Conv2d(in_c//2, out_c, 1)
        )
    def forward(self, x): return self.head(x)



class BUSISwinModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = SwinBackbone(pretrained=True)
        self.up2 = nn.Sequential(nn.Conv2d(512, 256, 1), nn.BatchNorm2d(256), nn.ReLU())
        self.up3 = nn.Sequential(nn.Conv2d(1024, 256, 1), nn.BatchNorm2d(256), nn.ReLU())
        self.asf = ASFBlock(256)
        self.seg_head = SegHead(256, 1)

        # 深监督
        self.aux_head1 = SegHead(256, 1)
        self.aux_head2 = SegHead(512, 1)

    def forward(self, x):
        f1, f2, f3 = self.backbone(x)  # [B,256,H/4], [B,512,H/8], [B,1024,H/16]

        target_sz = f1.shape[2:]
        f2_up = F.interpolate(self.up2(f2), size=target_sz, mode='bilinear', align_corners=False)
        f3_up = F.interpolate(self.up3(f3), size=target_sz, mode='bilinear', align_corners=False)

        fused = self.asf(f1, f2_up, f3_up)
        logits = self.seg_head(fused)

        aux1 = self.aux_head1(f1)
        aux2 = F.interpolate(self.aux_head2(f2), size=target_sz, mode='bilinear', align_corners=False)

        return logits, aux1, aux2
    
    
    
if  __name__ == "__main__":
    model = BUSISwinModel()
    x = torch.randn(2, 3, 224, 224)
    logits, aux1, aux2 = model(x)
    print(logits.shape, aux1.shape, aux2.shape)  # Expected: [2,1,224,224] [2,1,56,56] [2,1,56,56]