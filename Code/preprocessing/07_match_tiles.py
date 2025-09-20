import os

# === CONFIG ===
sar_tile_dir = "/home/jagrati/Desktop/Dataset_Image_Translation/SAR_VV"
optical_tile_dir = "/home/jagrati/Desktop/Dataset_Image_Translation/SAR_VH"
output_match_list = "matched_tile_names.txt"  # This file will store matched tile filenames

# === FUNCTION ===
def match_tiles(sar_dir, optical_dir):
    sar_tiles = set(f for f in os.listdir(sar_dir) if f.endswith(".tif"))
    optical_tiles = set(f for f in os.listdir(optical_dir) if f.endswith(".tif"))

    matched_tiles = sorted(list(sar_tiles & optical_tiles))
    unmatched_sar = sorted(list(sar_tiles - optical_tiles))
    unmatched_optical = sorted(list(optical_tiles - sar_tiles))

    print(f"✅ Total SAR tiles: {len(sar_tiles)}")
    print(f"✅ Total Optical tiles: {len(optical_tiles)}")
    print(f"✅ Matched tile names: {len(matched_tiles)}")
    print(f"❌ SAR-only tiles: {len(unmatched_sar)}")
    print(f"❌ Optical-only tiles: {len(unmatched_optical)}")

    if matched_tiles:
        with open(output_match_list, "w") as f:
            for tile in matched_tiles:
                f.write(tile + "\n")
        print(f"📁 Matched tile list saved to: {output_match_list}")

# === RUN ===
if __name__ == "__main__":
    match_tiles(sar_tile_dir, optical_tile_dir)
