import torch
import torch.nn.functional as F
import numpy as np
import math

def psnr(pred, target, max_val=1.0):
    """
    Compute Peak Signal to Noise Ratio (PSNR)
    Args:
        pred, target: Tensors [B, C, H, W], assumed in [0, 1]
        max_val: Max pixel value (default 1.0 for normalized images)
    Returns:
        PSNR value (float)
    """
    pred = torch.clamp(pred, 0.0, 1.0)
    target = torch.clamp(target, 0.0, 1.0)

    mse = F.mse_loss(pred, target, reduction='mean').item()
    if not math.isfinite(mse) or mse == 0:
        return float('inf')
    return 20 * math.log10(max_val / math.sqrt(mse))


def ssim(pred, target, window_size=11, size_average=True):
    """
    Compute Structural Similarity Index (SSIM)
    Args:
        pred, target: Tensors [B, C, H, W], assumed in [0, 1]
        window_size: Sliding window size (default 11)
    Returns:
        SSIM value (float)
    """
    pred = torch.clamp(pred, 0.0, 1.0)
    target = torch.clamp(target, 0.0, 1.0)

    # Convert to grayscale if 3 channels
    if pred.shape[1] == 3:
        pred = 0.2989 * pred[:, 0, :, :] + 0.5870 * pred[:, 1, :, :] + 0.1140 * pred[:, 2, :, :]
        target = 0.2989 * target[:, 0, :, :] + 0.5870 * target[:, 1, :, :] + 0.1140 * target[:, 2, :, :]

    pred = pred.unsqueeze(1)
    target = target.unsqueeze(1)

    mu1 = F.avg_pool2d(pred, window_size, stride=1, padding=window_size // 2)
    mu2 = F.avg_pool2d(target, window_size, stride=1, padding=window_size // 2)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.avg_pool2d(pred * pred, window_size, stride=1, padding=window_size // 2) - mu1_sq
    sigma2_sq = F.avg_pool2d(target * target, window_size, stride=1, padding=window_size // 2) - mu2_sq
    sigma12 = F.avg_pool2d(pred * target, window_size, stride=1, padding=window_size // 2) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean().item()
    else:
        return ssim_map.mean([1, 2, 3])

if __name__ == "__main__":
    x = torch.rand(1, 3, 256, 256)
    y = x + 0.01 * torch.randn_like(x)
    print("PSNR:", psnr(x, y))
    print("SSIM:", ssim(x, y))
