import os
import random
import shutil
from glob import glob

# -----------------------------
# Configurations (EDIT HERE)
# -----------------------------
input_dir = "/home/jagrati/Desktop/Dataset_Image_Translation/Stacked"
output_base_dir = "/home/jagrati/Desktop/Dataset_Image_Translation/Split"
split_ratios = {"train": 0.7, "val": 0.15, "test": 0.15}
file_extension = ".tif"
seed = 42  # for reproducibility

# -----------------------------
# Create Output Folders
# -----------------------------
def create_dirs(base_dir):
    for split in split_ratios.keys():
        split_path = os.path.join(base_dir, split)
        os.makedirs(split_path, exist_ok=True)

# -----------------------------
# Split and Move Files
# -----------------------------
def split_dataset():
    random.seed(seed)
    all_files = sorted(glob(os.path.join(input_dir, f"*{file_extension}")))
    total = len(all_files)
    print(f"Total TIFF files found: {total}")

    # Shuffle files
    random.shuffle(all_files)

    # Compute split sizes
    train_end = int(split_ratios["train"] * total)
    val_end = train_end + int(split_ratios["val"] * total)

    splits = {
        "train": all_files[:train_end],
        "val": all_files[train_end:val_end],
        "test": all_files[val_end:]
    }

    for split, files in splits.items():
        split_dir = os.path.join(output_base_dir, split)
        for file in files:
            shutil.copy(file, split_dir)
        print(f"✅ {split.capitalize()} set: {len(files)} files copied to {split_dir}")

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    create_dirs(output_base_dir)
    split_dataset()
