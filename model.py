import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class CBAMBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(CBAMBlock, self).__init__()
        self.channel_fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch, channels, _, _ = x.size()
        avg_pool = torch.mean(x, dim=(2, 3))
        max_pool = torch.amax(x, dim=(2, 3))
        channel_attention = self.channel_fc(avg_pool) + self.channel_fc(max_pool)
        channel_attention = channel_attention.view(batch, channels, 1, 1)
        x = x * channel_attention
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        spatial_attention = self.sigmoid(self.spatial_conv(torch.cat([avg_pool, max_pool], dim=1)))
        return x * spatial_attention

class FPNBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FPNBlock, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x, skip):
        x = self.conv1x1(x)
        x = self.upsample(x)
        return x + skip

class UNetFPN(nn.Module):
    def __init__(self):
        super(UNetFPN, self).__init__()
        self.enc1 = DoubleConv(3, 64)
        self.cbam1 = CBAMBlock(64)
        self.enc2 = DoubleConv(64, 128)
        self.cbam2 = CBAMBlock(128)
        self.enc3 = DoubleConv(128, 256)
        self.cbam3 = CBAMBlock(256)
        self.enc4 = DoubleConv(256, 512)
        self.cbam4 = CBAMBlock(512)
        self.enc5 = DoubleConv(512, 1024)
        self.cbam5 = CBAMBlock(1024)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.fpn1 = FPNBlock(1024, 512)
        self.fpn2 = FPNBlock(512, 256)
        self.fpn3 = FPNBlock(256, 128)
        self.fpn4 = FPNBlock(128, 64)
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        enc1 = self.enc1(x)
        enc1 = self.cbam1(enc1)
        enc2 = self.enc2(self.pool(enc1))
        enc2 = self.cbam2(enc2)
        enc3 = self.enc3(self.pool(enc2))
        enc3 = self.cbam3(enc3)
        enc4 = self.enc4(self.pool(enc3))
        enc4 = self.cbam4(enc4)
        enc5 = self.enc5(self.pool(enc4))
        enc5 = self.cbam5(enc5)
        fpn1 = self.fpn1(enc5, enc4)
        fpn2 = self.fpn2(fpn1, enc3)
        fpn3 = self.fpn3(fpn2, enc2)
        fpn4 = self.fpn4(fpn3, enc1)
        out = self.sigmoid(self.final_conv(fpn4))
        return out