import os
import torch
from torch.utils.data import Dataset
import rasterio
import numpy as np


class SAROpticalDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.filenames = sorted([
            fname for fname in os.listdir(root_dir) 
            if fname.endswith(".tif")
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        filepath = os.path.join(self.root_dir, filename)

        with rasterio.open(filepath) as src:
            image = src.read()  # [5, H, W]

        image = image.astype(np.float32)

        # --- Separate bands ---
        vh = image[0]
        vv = image[1]
        r = image[2]
        g = image[3]
        b = image[4]

        # --- Normalize SAR (clip dB range and scale to [0,1]) ---
        vh = np.clip(vh, -25, 0)
        vv = np.clip(vv, -25, 0)
        vh = (vh + 25) / 25.0
        vv = (vv + 25) / 25.0

        # --- Normalize Optical ---
        if r.max() > 1.0:  # Assume uint8 or 16-bit data
            r /= 255.0
            g /= 255.0
            b /= 255.0

        # --- Stack and convert to tensor ---
        sar = np.stack([vh, vv], axis=0)
        optical = np.stack([r, g, b], axis=0)

        sar_tensor = torch.from_numpy(sar).float()
        optical_tensor = torch.from_numpy(optical).float()

        return {
            "input": sar_tensor,
            "target": optical_tensor,
            "filename": filename
        }
