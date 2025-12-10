import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class UNetPlusPlus(nn.Module):
    def __init__(self, in_channels=1, num_classes=1, deep_supervision=True, base_c=32):
        super().__init__()
        self.deep_supervision = deep_supervision
        filters = [base_c, base_c*2, base_c*4, base_c*8, base_c*16]  # [32,64,128,256,512]

        # ------------------- Encoder (Downsampling) -------------------
        self.pool = nn.MaxPool2d(2, 2)
        self.conv0_0 = ConvBlock(in_channels, filters[0])
        self.conv1_0 = ConvBlock(filters[0], filters[1])
        self.conv2_0 = ConvBlock(filters[1], filters[2])
        self.conv3_0 = ConvBlock(filters[2], filters[3])
        self.conv4_0 = ConvBlock(filters[3], filters[4])

        # ------------------- Nested Decoder Blocks -------------------
        # Level 1
        self.conv0_1 = ConvBlock(filters[0]+filters[1], filters[0])
        self.conv0_2 = ConvBlock(filters[0]*2 + filters[1], filters[0])
        self.conv0_3 = ConvBlock(filters[0]*3 + filters[1], filters[0])
        self.conv0_4 = ConvBlock(filters[0]*4 + filters[1], filters[0])

        # Level 2
        self.conv1_1 = ConvBlock(filters[1]+filters[2], filters[1])
        self.conv1_2 = ConvBlock(filters[1]*2 + filters[2], filters[1])
        self.conv1_3 = ConvBlock(filters[1]*3 + filters[2], filters[1])

        # Level 3
        self.conv2_1 = ConvBlock(filters[2]+filters[3], filters[2])
        self.conv2_2 = ConvBlock(filters[2]*2 + filters[3], filters[2])

        # Level 4
        self.conv3_1 = ConvBlock(filters[3]+filters[4], filters[3])

        # ------------------- Upsampling -------------------
        self.up1_0 = nn.ConvTranspose2d(filters[1], filters[1], 2, stride=2)
        self.up2_0 = nn.ConvTranspose2d(filters[2], filters[2], 2, stride=2)
        self.up3_0 = nn.ConvTranspose2d(filters[3], filters[3], 2, stride=2)
        self.up4_0 = nn.ConvTranspose2d(filters[4], filters[4], 2, stride=2)

        self.up0_1 = nn.ConvTranspose2d(filters[0], filters[0], 2, stride=2)
        self.up1_1 = nn.ConvTranspose2d(filters[1], filters[1], 2, stride=2)
        self.up2_1 = nn.ConvTranspose2d(filters[2], filters[2], 2, stride=2)
        self.up3_1 = nn.ConvTranspose2d(filters[3], filters[3], 2, stride=2)

        self.up0_2 = nn.ConvTranspose2d(filters[0], filters[0], 2, stride=2)
        self.up1_2 = nn.ConvTranspose2d(filters[1], filters[1], 2, stride=2)
        self.up2_2 = nn.ConvTranspose2d(filters[2], filters[2], 2, stride=2)

        self.up0_3 = nn.ConvTranspose2d(filters[0], filters[0], 2, stride=2)
        self.up1_3 = nn.ConvTranspose2d(filters[1], filters[1], 2, stride=2)

        self.up0_4 = nn.ConvTranspose2d(filters[0], filters[0], 2, stride=2)

        # ------------------- Final 1x1 Convs (for each output) -------------------
        self.final0_0 = nn.Conv2d(filters[0], num_classes, 1)
        self.final0_1 = nn.Conv2d(filters[0], num_classes, 1)
        self.final0_2 = nn.Conv2d(filters[0], num_classes, 1)
        self.final0_3 = nn.Conv2d(filters[0], num_classes, 1)
        self.final0_4 = nn.Conv2d(filters[0], num_classes, 1)  # deep supervision

    def forward(self, x):
        # ------------------- Encoder -------------------
        x0_0 = self.conv0_0(x)         # (B,32,H,W)
        x1_0 = self.conv1_0(self.pool(x0_0))  # (B,64,H/2,W/2)
        x2_0 = self.conv2_0(self.pool(x1_0))  # (B,128,H/4,W/4)
        x3_0 = self.conv3_0(self.pool(x2_0))  # (B,256,H/8,W/8)
        x4_0 = self.conv4_0(self.pool(x3_0))  # (B,512,H/16,W/16)

        # ------------------- Nested Decoder -------------------
        # Level 1: x0_1
        x0_1 = self.conv0_1(torch.cat([
            x0_0,
            self.up1_0(x1_0)
        ], dim=1))

        # Level 2: x1_1, x0_2
        x1_1 = self.conv1_1(torch.cat([
            x1_0,
            self.up2_0(x2_0)
        ], dim=1))

        x0_2 = self.conv0_2(torch.cat([
            x0_0, x0_1,
            self.up1_1(x1_1)
        ], dim=1))

        # Level 3: x2_1, x1_2, x0_3
        x2_1 = self.conv2_1(torch.cat([
            x2_0,
            self.up3_0(x3_0)
        ], dim=1))

        x1_2 = self.conv1_2(torch.cat([
            x1_0, x1_1,
            self.up2_1(x2_1)
        ], dim=1))

        x0_3 = self.conv0_3(torch.cat([
            x0_0, x0_1, x0_2,
            self.up1_2(x1_2)
        ], dim=1))

        # Level 4: x3_1, x2_2, x1_3, x0_4
        x3_1 = self.conv3_1(torch.cat([
            x3_0,
            self.up4_0(x4_0)
        ], dim=1))

        x2_2 = self.conv2_2(torch.cat([
            x2_0, x2_1,
            self.up3_1(x3_1)
        ], dim=1))

        x1_3 = self.conv1_3(torch.cat([
            x1_0, x1_1, x1_2,
            self.up2_2(x2_2)
        ], dim=1))

        x0_4 = self.conv0_4(torch.cat([
            x0_0, x0_1, x0_2, x0_3,
            self.up1_3(x1_3)
        ], dim=1))

        # ------------------- Outputs (Deep Supervision) -------------------
        # if self.deep_supervision:
        #     out0 = self.final0_0(x0_0)
        #     out1 = self.final0_1(self.up0_1(x0_1))
        #     out2 = self.final0_2(self.up0_2(x0_2))
        #     out3 = self.final0_3(self.up0_3(x0_3))
        #     out4 = self.final0_4(self.up0_4(x0_4))
        #     return [out0, out1, out2, out3, out4]  # 5 outputs
        # else:
        out = self.final0_4(self.up0_4(x0_4))  # only finest
        return out