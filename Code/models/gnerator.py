import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, use_relu=True):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.relu = nn.ReLU(inplace=True) if use_relu else nn.Identity()

    def forward(self, x):
        return self.relu(self.conv(x))


class EncodingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncodingBlock, self).__init__()
        self.encode = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )

    def forward(self, x):
        return self.encode(x)


class WindowSelfAttention(nn.Module):
    def __init__(self, dim, window_size=4, num_heads=4):
        super(WindowSelfAttention, self).__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H % self.window_size == 0 and W % self.window_size == 0, "Height and Width must be divisible by window_size"

        x = x.view(B, C, H // self.window_size, self.window_size, W // self.window_size, self.window_size)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous().view(-1, self.window_size * self.window_size, C)

        x, _ = self.attn(x, x, x)

        x = x.view(B, H // self.window_size, W // self.window_size, self.window_size, self.window_size, C)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, C, H, W)
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(self, channels):
        super(SwinTransformerBlock, self).__init__()
        self.norm1 = nn.BatchNorm2d(channels)
        self.attn = WindowSelfAttention(channels)
        self.norm2 = nn.BatchNorm2d(channels)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1),
        )

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.attn(x)
        x = x + residual

        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + residual
        return x


class DecodingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=True):
        super(DecodingBlock, self).__init__()
        layers = [
            ConvBlock(in_channels, in_channels),
            ConvBlock(in_channels, out_channels)
        ]
        if upsample:
            layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
            layers.append(nn.ReLU(inplace=True))
        self.decode = nn.Sequential(*layers)

    def forward(self, x):
        return self.decode(x)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.encoder1 = EncodingBlock(2, 64)       # [B, 64, H, W]
        self.encoder2 = EncodingBlock(64, 128)     # [B, 128, H, W]

        self.swin = SwinTransformerBlock(128)      # [B, 128, H, W]

        self.decoder1 = DecodingBlock(128, 64, upsample=False)  # [B, 64, H, W]
        self.decoder2 = DecodingBlock(64, 32, upsample=False)   # [B, 32, H, W]

        self.output_conv = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        e1 = self.encoder1(x)        # [B, 64, H, W]
        e2 = self.encoder2(e1)       # [B, 128, H, W]
        swin = self.swin(e2)         # [B, 128, H, W]
        d1 = self.decoder1(swin)     # [B, 64, H, W]
        d2 = self.decoder2(d1 + e1)  # [B, 32, H, W]
        out = torch.sigmoid(self.output_conv(d2))  # [B, 3, H, W] in [0, 1]
        return out
