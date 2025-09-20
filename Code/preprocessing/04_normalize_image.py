# 04_normalize_image.py

import rasterio
import numpy as np
import os

def normalize_image(input_path, output_path, min_val=None, max_val=None):
    with rasterio.open(input_path) as src:
        data = src.read().astype(np.float32)
        profile = src.profile

        # Sanitize NoData
        nodata_value = profile.get('nodata')
        if nodata_value is not None and (nodata_value > np.finfo(np.float32).max or np.isnan(nodata_value)):
            print(f"⚠️ Replacing nodata value {nodata_value} with -9999.0")
            profile['nodata'] = -9999.0
            data[data == nodata_value] = -9999.0  # Optional: mask it too

        # Compute min/max if not given
        if min_val is None or max_val is None:
            min_val = data.min(axis=(1, 2), keepdims=True)
            max_val = data.max(axis=(1, 2), keepdims=True)
        else:
            min_val = np.array(min_val).reshape(-1, 1, 1)
            max_val = np.array(max_val).reshape(-1, 1, 1)

        # Normalize
        norm_data = (data - min_val) / (max_val - min_val)
        norm_data = np.clip(norm_data, 0.0, 1.0)

        profile.update(dtype=rasterio.float32)

        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(norm_data)

    print(f"✅ Normalized image saved to: {output_path}")

if __name__ == "__main__":
    input_image = "/home/jagrati/Desktop/Dataset_Image_Translation/Resampled_SAR_VV/Washington_Florence_14_09_18_01_Resampled.tif"
    output_image = "/home/jagrati/Desktop/Dataset_Image_Translation/Normalized_SAR_VV/Washington_Florence_14_09_18_01_Normalized_VV.tif"

    os.makedirs(os.path.dirname(output_image), exist_ok=True)
    normalize_image(input_image, output_image)
