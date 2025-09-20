# 03_resample_sar_to_optical.py

import rasterio
from rasterio.enums import Resampling
import os

def resample_sar_to_optical(sar_path, ref_optical_path, output_path):
    with rasterio.open(sar_path) as sar_src, rasterio.open(ref_optical_path) as ref_src:
        # Match resolution and transform to optical
        transform = ref_src.transform
        width = ref_src.width
        height = ref_src.height

        # Update SAR metadata to match target
        profile = sar_src.profile
        profile.update({
            "transform": transform,
            "width": width,
            "height": height
        })

        with rasterio.open(output_path, "w", **profile) as dst:
            for i in range(1, sar_src.count + 1):
                resampled = sar_src.read(
                    i,
                    out_shape=(height, width),
                    resampling=Resampling.bilinear  # or Resampling.nearest
                )
                dst.write(resampled, i)

    print(f"✅ SAR image resampled and saved to: {output_path}")

if __name__ == "__main__":
    sar_input_path = "/home/jagrati/Desktop/Dataset_Image_Translation/SENTINEL_1/SAR_VV/Elizabethtown_Florence_14_09_18_01.tif"
    optical_ref_path = "/home/jagrati/Desktop/Dataset_Image_Translation/Crop_patch/Elizabethtown_Florence_14_09_18_01_RGB_Cropped.tif"
    output_resampled_sar = "/home/jagrati/Desktop/Dataset_Image_Translation/Resampled_SAR_VV/Elizabethtown_Florence_14_09_18_01_Resampled_VV.tif"

    os.makedirs(os.path.dirname(output_resampled_sar), exist_ok=True)
    resample_sar_to_optical(sar_input_path, optical_ref_path, output_resampled_sar)
