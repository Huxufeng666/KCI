# model/swin_unet.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==================== Swin Block ====================
class SwinBlock(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)
        shortcut = x_flat
        x_flat = self.norm1(x_flat)
        attn_out, _ = self.attn(x_flat, x_flat, x_flat)
        x_flat = shortcut + attn_out
        shortcut = x_flat
        x_flat = self.norm2(x_flat)
        x_flat = self.mlp(x_flat)
        x_flat = shortcut + x_flat
        x = x_flat.transpose(1, 2).view(B, C, H, W)
        return x

# ==================== Patch Embed ====================
class PatchEmbed(nn.Module):
    def __init__(self, in_chans=1, embed_dim=48, patch_size=4):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = x.transpose(1, 2).view(B, C, H, W)
        return x

# ==================== Patch Merging ====================
class PatchMerging(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        pad_h = (2 - H % 2) % 2
        pad_w = (2 - W % 2) % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
            H, W = x.shape[2], x.shape[3]

        x0 = x[:, :, 0::2, 0::2]
        x1 = x[:, :, 1::2, 0::2]
        x2 = x[:, :, 0::2, 1::2]
        x3 = x[:, :, 1::2, 1::2]
        x = torch.cat([x0, x1, x2, x3], 1)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = self.reduction(x)
        x = x.transpose(1, 2).view(B, 2*C, H//2, W//2)
        return x

# ==================== Patch Expanding (修复版) ====================
class PatchExpanding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.expand = nn.Linear(dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(2 * dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)      # (B, H*W, C)
        x = self.expand(x)                     # (B, H*W, 2*C)
        x = self.norm(x)
        # 修复：contiguous() + view
        x = x.contiguous().view(B, 2*C, H, W)  # (B, 2*C, H, W)
        x = F.pixel_shuffle(x, 2)              # (B, C//2, 2H, 2W)
        return x

# ==================== Swin-Unet ====================
class SwinUnet(nn.Module):
    def __init__(self, in_chans=1, num_classes=1, embed_dim=48, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24]):
        super().__init__()
        self.patch_embed = PatchEmbed(in_chans, embed_dim)
        self.layer1 = self._make_layer(SwinBlock, embed_dim, depths[0], num_heads[0])
        self.merge1 = PatchMerging(embed_dim)
        self.layer2 = self._make_layer(SwinBlock, embed_dim*2, depths[1], num_heads[1])
        self.merge2 = PatchMerging(embed_dim*2)
        self.layer3 = self._make_layer(SwinBlock, embed_dim*4, depths[2], num_heads[2])
        self.merge3 = PatchMerging(embed_dim*4)
        self.layer4 = self._make_layer(SwinBlock, embed_dim*8, depths[3], num_heads[3])

        self.up4 = PatchExpanding(embed_dim*8)
        self.dec4 = self._make_layer(SwinBlock, embed_dim*4, depths[2], num_heads[2])
        self.up3 = PatchExpanding(embed_dim*4)
        self.dec3 = self._make_layer(SwinBlock, embed_dim*2, depths[1], num_heads[1])
        self.up2 = PatchExpanding(embed_dim*2)
        self.dec2 = self._make_layer(SwinBlock, embed_dim, depths[0], num_heads[0])
        self.up1 = PatchExpanding(embed_dim)
        self.dec1 = self._make_layer(SwinBlock, embed_dim//2, depths[0], num_heads[0]//3)

        self.head = nn.Conv2d(embed_dim//2, num_classes, 1)

    def _make_layer(self, block, dim, blocks, heads):
        return nn.Sequential(*[block(dim, heads) for _ in range(blocks)])

    def forward(self, x):
        x1 = self.layer1(self.patch_embed(x))
        x2 = self.layer2(self.merge1(x1))
        x3 = self.layer3(self.merge2(x2))
        x4 = self.layer4(self.merge3(x3))

        d4 = self.up4(x4)
        d4 = d4 + x3
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = d3 + x2
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = d2 + x1
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = self.dec1(d1)

        out = self.head(d1)
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
        return out

# # ==================== 测试 ====================
# if __name__ == "__main__":
#     model = SwinUnet(in_chans=1, num_classes=1)
#     x = torch.randn(1, 1, 256, 256)
#     out = model(x)
#     print(f"Input: {x.shape} → Output: {out.shape}")