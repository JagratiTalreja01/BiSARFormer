# utils/image_utils.py

import numpy as np
import matplotlib.pyplot as plt
import torch

def normalize_tensor(img_tensor):
    """Normalize a tensor to [0, 1] range."""
    return (img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min())

def denormalize_tensor(img_tensor):
    """Reverses normalization if originally scaled between [0, 1]."""
    return img_tensor * 255.0

def show_tensor_images(tensor, title="", nrow=1):
    """Visualize tensor images using matplotlib"""
    tensor = tensor.detach().cpu()
    grid_img = torchvision.utils.make_grid(tensor, nrow=nrow)
    npimg = grid_img.numpy().transpose((1, 2, 0))
    plt.figure(figsize=(10, 5))
    plt.imshow(npimg)
    plt.title(title)
    plt.axis("off")
    plt.show()
