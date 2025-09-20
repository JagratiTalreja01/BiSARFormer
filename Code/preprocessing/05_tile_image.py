# 05_tile_image.py

import rasterio
import os
import numpy as np
from rasterio.windows import Window

# -----------------------------
# Config (Edit as needed)
# -----------------------------
input_image = "/home/jagrati/Desktop/Dataset_Image_Translation/Normalized_SAR_VV/Washington_Florence_14_09_18_01_Normalized_VV.tif"
output_dir = "/home/jagrati/Desktop/Dataset_Image_Translation/Tiles_folders_SAR_VV/Washington_Florence_14_09_18_01_Normalized_VV"
tile_size = 256  # Size of each tile (tile_size x tile_size)
stride = 256     # Use < tile_size for overlap

# -----------------------------
# Tiling Logic
# -----------------------------
def tile_image(input_path, output_folder, tile_size, stride):
    os.makedirs(output_folder, exist_ok=True)

    with rasterio.open(input_path) as src:
        meta = src.meta.copy()
        width, height = src.width, src.height
        bands = src.count

        tile_id = 0
        for top in range(0, height - tile_size + 1, stride):
            for left in range(0, width - tile_size + 1, stride):
                window = Window(left, top, tile_size, tile_size)
                transform = rasterio.windows.transform(window, src.transform)
                tile = src.read(window=window)

                meta.update({
                    "height": tile_size,
                    "width": tile_size,
                    "transform": transform
                })

                out_path = os.path.join(output_folder, f"tile_{tile_id:04d}.tif")
                with rasterio.open(out_path, 'w', **meta) as dst:
                    dst.write(tile)
                print(f"✅ Saved tile: {out_path}")
                tile_id += 1

        print(f"✅ Total tiles created: {tile_id}")

# -----------------------------
# Run it
# -----------------------------
if __name__ == "__main__":
    tile_image(input_image, output_dir, tile_size, stride)
