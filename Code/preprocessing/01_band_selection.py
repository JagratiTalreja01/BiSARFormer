# 01_band_selection.py

import rasterio
from rasterio import windows
from rasterio.enums import Resampling
import os
import numpy as np

# -----------------------------
# Config (Edit as needed)
# -----------------------------
input_path = "/home/jagrati/Desktop/Dataset_Image_Translation/SENTINEL_2/Washington_Florence_14_09_18_01.tif"
output_path = "/home/jagrati/Desktop//Dataset_Image_Translation/BS_Sentinel-2/Washington_Florence_14_09_18_01.tif"
selected_band_indices = [1, 2, 3]  # B2, B3, B4 (1-based indexing)

# -----------------------------
# Band Selection Logic
# -----------------------------
def select_bands(input_path, output_path, selected_band_indices):
    with rasterio.open(input_path) as src:
        profile = src.profile
        profile.update(count=len(selected_band_indices))

        # Create output file
        with rasterio.open(output_path, "w", **profile) as dst:
            for i, band_idx in enumerate(selected_band_indices):
                print(f"Reading band {band_idx}...")
                band = src.read(band_idx)
                dst.write(band, i + 1)
        print(f"✅ Selected bands saved to: {output_path}")

# -----------------------------
# Run It
# -----------------------------
if __name__ == "__main__":
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    select_bands(input_path, output_path, selected_band_indices)