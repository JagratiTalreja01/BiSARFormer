# 02_crop_optical_to_sar.py

import os
import rasterio
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds

def crop_optical_to_sar(optical_path, sar_ref_path, output_path):
    with rasterio.open(optical_path) as opt_src, rasterio.open(sar_ref_path) as sar_src:
        # Handle CRS mismatch
        if opt_src.crs != sar_src.crs:
            print("⚠ CRS mismatch detected. Reprojecting SAR bounds to Optical CRS...")
            sar_bounds = transform_bounds(sar_src.crs, opt_src.crs, *sar_src.bounds)
        else:
            sar_bounds = sar_src.bounds

        # Create a cropping window in optical image coordinates
        window = from_bounds(*sar_bounds, transform=opt_src.transform)

        # Update metadata
        profile = opt_src.profile
        profile.update({
            "height": int(window.height),
            "width": int(window.width),
            "transform": opt_src.window_transform(window)
        })

        with rasterio.open(output_path, "w", **profile) as dst:
            for i in range(1, opt_src.count + 1):
                band = opt_src.read(i, window=window)
                dst.write(band, i)

        print(f"✅ Cropped optical image saved to: {output_path}")

if __name__ == "__main__":
    optical_rgb_path = "/home/jagrati/Desktop/Dataset_Image_Translation/BS_Sentinel-2/Washington_Florence_14_09_18_01.tif"
    sar_vh_path = "/home/jagrati/Desktop/Dataset_Image_Translation/SENTINEL_1/SAR_VH/Washington_Florence_14_09_18_01.tif"
    output_crop_path = "/home/jagrati/Desktop/Dataset_Image_Translation/Crop_patch/Washington_Florence_14_09_18_01_RGB_Cropped.tif"

    os.makedirs(os.path.dirname(output_crop_path), exist_ok=True)
    crop_optical_to_sar(optical_rgb_path, sar_vh_path, output_crop_path)
