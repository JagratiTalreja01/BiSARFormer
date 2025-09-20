# 07_stack_sar_optical.py

import rasterio
import numpy as np
import os

# Set your paths
sar_vh_dir = "/home/jagrati/Desktop/Dataset_Image_Translation/SAR_VH"
sar_vv_dir = "/home/jagrati/Desktop/Dataset_Image_Translation/SAR_VV"
optical_dir = "/home/jagrati/Desktop/Dataset_Image_Translation/UAV"
output_dir = "/home/jagrati/Desktop/Dataset_Image_Translation/Stacked"
report_file = os.path.join(output_dir, "missing_tiles_report.txt")

os.makedirs(output_dir, exist_ok=True)

tile_names = sorted(os.listdir(sar_vh_dir))
missing_log = []

for tile_name in tile_names:
    sar_vh_path = os.path.join(sar_vh_dir, tile_name)
    sar_vv_path = os.path.join(sar_vv_dir, tile_name)
    optical_path = os.path.join(optical_dir, tile_name)
    output_path = os.path.join(output_dir, tile_name)

    if not os.path.exists(sar_vv_path) or not os.path.exists(optical_path):
        missing_log.append(tile_name)
        continue

    try:
        with rasterio.open(sar_vh_path) as src_vh, \
             rasterio.open(sar_vv_path) as src_vv, \
             rasterio.open(optical_path) as src_opt:

            vh = src_vh.read(1)
            vv = src_vv.read(1)
            rgb = src_opt.read([1, 2, 3])  # RGB

            # Stack: SAR_VH, SAR_VV, R, G, B
            stacked = np.stack([rgb[0], rgb[1], rgb[2], vh, vv])

            profile = src_opt.profile
            profile.update(count=5)

            with rasterio.open(output_path, "w", **profile) as dst:
                dst.write(stacked)

            print(f"✅ Stacked saved: {tile_name}")

    except Exception as e:
        print(f"❌ Error with {tile_name}: {e}")
        missing_log.append(tile_name)

# Save report
with open(report_file, "w") as f:
    f.write("Missing or mismatched tile sets:\n")
    for name in missing_log:
        f.write(name + "\n")

print(f"\n📄 Missing tile report saved to: {report_file}")
