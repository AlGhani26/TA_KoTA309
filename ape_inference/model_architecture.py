import torch
import torch.nn as nn
import torch.nn.functional as F
from SKResNeXt import SKResNeXt50 
from AM import AttentionBlock  
from DDU import DUpsampling 

class UNet_SKResNeXt50(nn.Module):
    def __init__(self, in_channels=4, out_channels=6, cardinality=32, bwidth=4):
        super(UNet_SKResNeXt50, self).__init__()
        self.encoder = SKResNeXt50(img_channels=in_channels, num_classes=out_channels, cardinality=cardinality, bwidth=bwidth)

        # Decoder
        self.expansion_filter = (cardinality * bwidth) * 2

        self.up1 = nn.ConvTranspose2d(self.expansion_filter*8, self.expansion_filter*4, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(self.expansion_filter*8, self.expansion_filter*4)  # 1024 + skip 1024

        self.up2 = nn.ConvTranspose2d(self.expansion_filter*4, self.expansion_filter*2, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(self.expansion_filter*4, self.expansion_filter*2)   # 512 + skip 512

        self.up3 = nn.ConvTranspose2d(self.expansion_filter*2, self.expansion_filter, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(self.expansion_filter*2, self.expansion_filter)    # 256 + skip 256

        self.up4 = nn.ConvTranspose2d(self.expansion_filter, 128, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(128 + 64, 64)

        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)


    def conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Encoder
        x0, x1, x2, x3, x4 = self.encoder(x)

        # Decoder
        d1 = self.up1(x4)
        d1 = self.dec1(torch.cat([d1, x3], dim=1))

        d2 = self.up2(d1)
        d2 = self.dec2(torch.cat([d2, x2], dim=1))

        d3 = self.up3(d2)
        d3 = self.dec3(torch.cat([d3, x1], dim=1))

        d4 = self.up4(d3)
        d4 = self.dec4(torch.cat([d4, x0], dim=1))

        out =  self.out_conv(d4)
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
        return out

# === U-Net SKResNeXt Decoder Modifikasi ===
class UNet_SKResNeXt50_Modifikasi_A(nn.Module):
    def __init__(self, in_channels=4, out_channels=6, cardinality=32, bwidth=4):
        super(UNet_SKResNeXt50_Modifikasi_A, self).__init__()
        self.encoder = SKResNeXt50(img_channels=in_channels, num_classes=out_channels, cardinality=cardinality, bwidth=bwidth)

        self.expansion_filter = (cardinality * bwidth) * 2

        self.up1 = DUpsampling(self.expansion_filter*8, 2, self.expansion_filter*4)
        self.dec1 = self.conv_block(self.expansion_filter*8, self.expansion_filter*4)
        self.dec1_att = AttentionBlock(self.expansion_filter*4)

        self.up2 = DUpsampling(self.expansion_filter*4, 2, self.expansion_filter*2)
        self.dec2 = self.conv_block(self.expansion_filter*4 + self.expansion_filter*4, self.expansion_filter*2)
        self.dec2_att = AttentionBlock(self.expansion_filter*2)

        self.up3 = DUpsampling(self.expansion_filter*2, 2, self.expansion_filter)
        self.dec3 = self.conv_block(self.expansion_filter*2 + self.expansion_filter*2 + self.expansion_filter*4, self.expansion_filter)
        self.dec3_att = AttentionBlock(self.expansion_filter)

        self.up4 = DUpsampling(self.expansion_filter, 2, 128)
        self.dec4 = self.conv_block(128 + 64 + self.expansion_filter + self.expansion_filter*2 + self.expansion_filter*4, 64)
        self.dec4_att = AttentionBlock(64)

        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Encoder
        x0, x1, x2, x3, x4 = self.encoder(x)

        # Decoder with dense connections
        up1 = self.up1(x4)
        d1_input = torch.cat([up1, x3], dim=1)
        dec1_block = self.dec1(d1_input)
        dec1 = self.dec1_att(dec1_block)

        up2 = self.up2(dec1)
        d2_input = torch.cat([
            up2,
            x2,
            F.interpolate(dec1_block, size=x2.shape[2:], mode='bilinear', align_corners=False)
        ], dim=1)
        dec2_block = self.dec2(d2_input)
        dec2 = self.dec2_att(dec2_block)

        up3 = self.up3(dec2)
        d3_input = torch.cat([
            up3,
            x1,
            F.interpolate(dec2_block, size=x1.shape[2:], mode='bilinear', align_corners=False),
            F.interpolate(dec1_block, size=x1.shape[2:], mode='bilinear', align_corners=False)
        ], dim=1)
        dec3_block = self.dec3(d3_input)
        dec3 = self.dec3_att(dec3_block)

        up4 = self.up4(dec3)
        d4_input = torch.cat([
            up4,
            x0,
            F.interpolate(dec3_block, size=x0.shape[2:], mode='bilinear', align_corners=False),
            F.interpolate(dec2_block, size=x0.shape[2:], mode='bilinear', align_corners=False),
            F.interpolate(dec1_block, size=x0.shape[2:], mode='bilinear', align_corners=False)
        ], dim=1)
        dec4_block = self.dec4(d4_input)
        dec4 = self.dec4_att(dec4_block)

        # Final output
        out = self.out_conv(dec4)
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
        return out

# === U-Net SKResNeXt Decoder Modifikasi Versi 2===
class UNet_SKResNeXt50_Modifikasi_B(nn.Module):
    def __init__(self, in_channels=4, out_channels=6, cardinality=32, bwidth=4):
        super(UNet_SKResNeXt50_Modifikasi_B, self).__init__()
        self.encoder = SKResNeXt50(img_channels=in_channels, num_classes=out_channels, cardinality=cardinality, bwidth=bwidth)

        # Decoder
        self.expansion_filter = (cardinality * bwidth) * 2

        self.up1 = DUpsampling(self.expansion_filter*8, 2, self.expansion_filter*4)
        self.dec1 = nn.Sequential(self.conv_block(self.expansion_filter*8, self.expansion_filter*4), AttentionBlock(self.expansion_filter*4))  # 1024 + skip 1024

        self.up2 = DUpsampling(self.expansion_filter*4, 2, self.expansion_filter*2)
        self.dec2 = nn.Sequential(self.conv_block(self.expansion_filter*4, self.expansion_filter*2), AttentionBlock(self.expansion_filter*2))   # 512 + skip 512

        self.up3 = DUpsampling(self.expansion_filter*2, 2, self.expansion_filter)
        self.dec3 = nn.Sequential(self.conv_block(self.expansion_filter*2, self.expansion_filter), AttentionBlock(self.expansion_filter))    # 256 + skip 256

        self.up4 = DUpsampling(self.expansion_filter, 2, 128)
        self.dec4 = nn.Sequential(self.conv_block(128 + 64, 64), AttentionBlock(64))  # 128 + skip 64

        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)


    def conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Encoder
        x0, x1, x2, x3, x4 = self.encoder(x)

        # Decoder
        d1 = self.up1(x4)
        d1 = self.dec1(torch.cat([d1, x3], dim=1))

        d2 = self.up2(d1)
        d2 = self.dec2(torch.cat([d2, x2], dim=1))

        d3 = self.up3(d2)
        d3 = self.dec3(torch.cat([d3, x1], dim=1))

        d4 = self.up4(d3)
        d4 = self.dec4(torch.cat([d4, x0], dim=1))

        out =  self.out_conv(d4)
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
        return out