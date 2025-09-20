# utils/plot_utils.py

import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_sample_triplet(sar_tensor, generated_tensor, target_tensor, filename=None, save_path=None):
    """
    Plots a triplet: SAR input (2-ch), generated RGB output (3-ch), ground-truth RGB (3-ch)
    
    Args:
        sar_tensor: [2, H, W] - input SAR (VH, VV)
        generated_tensor: [3, H, W] - predicted RGB output
        target_tensor: [3, H, W] - ground-truth RGB
        filename: optional filename to title/save
        save_path: if provided, saves the figure to this path
    """
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy().transpose(1, 2, 0)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # SAR: Combine VH/VV as grayscale composite for visualization
    sar_image = to_numpy(sar_tensor)
    sar_image = (sar_image - sar_image.min()) / (sar_image.max() - sar_image.min())
    if sar_image.shape[2] == 2:
        sar_image = np.stack([sar_image[:, :, 0]] * 3, axis=-1)

    # Predicted and Ground Truth
    gen_image = to_numpy(generated_tensor)
    gen_image = np.clip(gen_image, 0, 1)

    tgt_image = to_numpy(target_tensor)
    tgt_image = np.clip(tgt_image, 0, 1)

    axes[0].imshow(sar_image)
    axes[0].set_title("SAR Input (VH+VV)")
    axes[0].axis("off")

    axes[1].imshow(gen_image)
    axes[1].set_title("Generated Optical")
    axes[1].axis("off")

    axes[2].imshow(tgt_image)
    axes[2].set_title("Ground Truth Optical")
    axes[2].axis("off")

    if filename:
        fig.suptitle(filename, fontsize=12)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()

    plt.close()
