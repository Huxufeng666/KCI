import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 修改为 Avg 以避免某些确定性问题，可恢复为 Max
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.fc(self.avg_pool(x))
        return self.sigmoid(avg)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
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

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.residual = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        identity = self.residual(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)



class TopDownFPN(nn.Module):
    def __init__(self, in_channels_list, out_channels=128):
        super().__init__()
        self.laterals = nn.ModuleList([nn.Conv2d(c, out_channels, 1) for c in in_channels_list])
        self.fusions = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=out_channels//8),
                nn.Conv2d(out_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                CBAM(out_channels)
            ) for _ in in_channels_list
        ])
    def forward(self, feats):  # feats: [e1, e2, e3, e4, b]
        feats = feats[::-1]  # b, e4, e3, e2, e1
        laterals = [lat(f) for lat, f in zip(self.laterals, feats)]
        x = laterals[0]
        out = [self.fusions[0](x)]
        for i in range(1, len(feats)):
            x = F.interpolate(x, size=laterals[i].shape[2:], mode='nearest') + laterals[i]
            out.append(self.fusions[i](x))
        return out[::-1]  # p1 to p5





class MultiTaskFPNUNet(nn.Module):
    def __init__(self, in_ch=1, seg_ch=1, features=[64, 128, 256, 512], num_classes=3):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc1 = ConvBlock(in_ch, features[0])
        self.enc2 = ConvBlock(features[0], features[1])
        self.enc3 = ConvBlock(features[1], features[2])
        self.enc4 = ConvBlock(features[2], features[3])
        self.bottleneck = ConvBlock(features[3], features[3]*2)

        self.e1_adapter = nn.Conv2d(features[0], features[0], kernel_size=1)
        self.e2_adapter = nn.Conv2d(features[1], features[1], kernel_size=1)
        self.fuse3 = nn.Conv2d(features[1] + features[0], features[1], kernel_size=1)
        self.fuse4 = nn.Conv2d(features[2] + features[1], features[2], kernel_size=1)

        self.fpn = TopDownFPN(
            in_channels_list=[features[3]*2, features[3], features[2], features[1], features[0]],
            out_channels=128
        )

        self.up4 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.cbam4 = CBAM(128*2 + features[3])
        self.dec4 = ConvBlock(128*2 + features[3], 128)

        self.up3 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.cbam3 = CBAM(128*2 + features[2])
        self.dec3 = ConvBlock(128*2 + features[2], 128)

        self.up2 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.cbam2 = CBAM(128*2 + features[1])
        self.dec2 = ConvBlock(128*2 + features[1], 128)

        self.up1 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.cbam1 = CBAM(128*2 + features[0])
        self.dec1 = ConvBlock(128*2 + features[0], 128)
        
        # ====================== 两个独立的 Head ======================
        # Decoder A：用于 edge + recon + cls
        self.edge_head   = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1)
        )
        self.recon_head  = nn.Sequential(
            nn.Conv2d(128, in_ch, kernel_size=1),
            nn.Tanh()
        )

    
        # Segmentation head
        self.final_conv = nn.Conv2d(128, seg_ch, kernel_size=1)
        self.aux4 = nn.Conv2d(128, seg_ch, kernel_size=1)
        self.aux3 = nn.Conv2d(128, seg_ch, kernel_size=1)
        self.aux2 = nn.Conv2d(128, seg_ch, kernel_size=1)

        # Classification head (任务1)
        self.class_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(features[3]*2, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

        # Edge detection head (任务2)
        # self.edge_conv = nn.Conv2d(128, 1, kernel_size=1)
        self.edge_conv = nn.Sequential(
                nn.Conv2d(128, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 1, kernel_size=1)
            )
        # Reconstruction head (任务3)
        self.recon_conv = nn.Conv2d(128, in_ch, kernel_size=1)
        self.recon_activate = nn.Tanh()  # 假设图像归一化到[-1,1]，否则用 Sigmoid [0,1]


    def forward(self, x):
        # ====================== 共享编码器（不变）======================
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))

        e3_input = self.pool(e2)
        e1_resized = F.interpolate(self.e1_adapter(e1), size=e3_input.shape[2:], mode='bilinear', align_corners=True)
        combined3 = torch.cat([e3_input, e1_resized], dim=1)
        e3 = self.enc3(self.fuse3(combined3))

        e4_input = self.pool(e3)
        e2_resized = F.interpolate(self.e2_adapter(e2), size = e4_input.shape[2:], mode = 'bilinear', align_corners = True)
        combined4 = torch.cat([e4_input, e2_resized], dim=1)
        e4 = self.enc4(self.fuse4(combined4))

        b = self.bottleneck(self.pool(e4))
        fpn_feats = self.fpn([e1, e2, e3, e4, b])

        # ====================== Decoder A：edge + recon ======================
        d4 = self.up4(fpn_feats[4])
        d4 = torch.cat([d4, fpn_feats[3], e4], dim=1)
        d4 = self.cbam4(d4)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, fpn_feats[2], e3], dim=1)
        d3 = self.cbam3(d3)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, fpn_feats[1], e2], dim=1)
        d2 = self.cbam2(d2)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, fpn_feats[0], e1], dim=1)
        d1 = self.cbam1(d1)
        d1 = self.dec1(d1)   # ← Decoder A 最终特征

        # ====================== Decoder B：segmentation（参数共享）======================
        d4_b = self.up4(fpn_feats[4])
        d4_b = torch.cat([d4_b, fpn_feats[3], e4], dim=1)

        d4_b = self.cbam4(d4_b)
        d4_b = self.dec4(d4_b)
        d3_b = self.up3(d4_b)
        d3_b = torch.cat([d3_b, fpn_feats[2], e3], dim=1)

        d3_b = self.cbam3(d3_b)
        d3_b = self.dec3(d3_b)
        d2_b = self.up2(d3_b)
        d2_b = torch.cat([d2_b, fpn_feats[1], e2], dim=1)

        d2_b = self.cbam2(d2_b)
        d2_b = self.dec2(d2_b)
        d1_b = self.up1(d2_b)
        d1_b = torch.cat([d1_b, fpn_feats[0], e1], dim=1)
        d1_b = self.cbam1(d1_b)
        d1_b = self.dec1(d1_b)   # ← Decoder B 最终特征

        # ====================== 输出（完全正确版）======================
        # Segmentation（来自 Decoder B）
        seg_main = self.final_conv(d1_b)
        aux2 = F.interpolate(self.aux2(d2_b), size=x.shape[2:], mode='bilinear', align_corners=True)
        aux3 = F.interpolate(self.aux3(d3_b), size=x.shape[2:], mode='bilinear', align_corners=True)
        aux4 = F.interpolate(self.aux4(d4_b), size=x.shape[2:], mode='bilinear', align_corners=True)

        # Classification（来自 bottleneck）
        class_logits = self.class_head(b)

        # Edge + Recon（来自 Decoder A）
        edge_out  = F.interpolate(self.edge_head(d1), size=x.shape[2:], mode='bilinear', align_corners=True)
        recon_out = F.interpolate(self.recon_head(d1), size=x.shape[2:], mode='bilinear', align_corners=True)

        return seg_main, aux2, aux3, aux4, class_logits, edge_out, recon_out