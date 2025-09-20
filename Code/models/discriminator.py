import torch
import torch.nn as nn


class PatchGANDiscriminator(nn.Module):
    def __init__(self, input_channels=5, ndf=64):
        """
        input_channels: 5 for SAR+Optical (VH, VV, R, G, B)
        ndf: base number of discriminator filters
        """
        super(PatchGANDiscriminator, self).__init__()

        self.model = nn.Sequential(
            # Input: (5) x 256 x 256 → (ndf) x 128 x 128
            nn.Conv2d(input_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # (ndf) x 128 x 128 → (ndf*2) x 64 x 64
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # (ndf*2) x 64 x 64 → (ndf*4) x 32 x 32
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # (ndf*4) x 32 x 32 → (ndf*8) x 16 x 16
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # Final output layer → 1 x (patch_size x patch_size)
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, x):
        return self.model(x)
