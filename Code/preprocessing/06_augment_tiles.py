import os
import rasterio
import numpy as np
from torchvision import transforms
from PIL import Image

# -----------------------------
# Config
# -----------------------------
input_dir = "/home/jagrati/Desktop/Dataset_Image_Translation/Tiles_SAR_VV"
output_dir = "/home/jagrati/Desktop/Dataset_Image_Translation/Augmented_tiles_SAR_VV"
AUG_PER_IMAGE = 3

# -----------------------------
# Augmentation Setup
# -----------------------------
augmentation_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(degrees=(0, 360)),
])

# -----------------------------
# Augment Function
# -----------------------------
def augment_tiles(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_files = [f for f in os.listdir(input_dir) if f.endswith(".tif")]

    for file in image_files:
        input_path = os.path.join(input_dir, file)
        with rasterio.open(input_path) as src:
            img = src.read()
            profile = src.profile

            # CHW -> HWC
            img_hwc = np.moveaxis(img, 0, -1)
            img_hwc = np.nan_to_num(img_hwc, nan=0.0, posinf=255.0, neginf=0.0)

            for i in range(AUG_PER_IMAGE):
                # Convert to PIL
                if img_hwc.shape[-1] == 1:
                    pil_img = Image.fromarray(img_hwc[:, :, 0].astype(np.uint8), mode="L")
                else:
                    pil_img = Image.fromarray(img_hwc.astype(np.uint8), mode="RGB")

                aug_img = augmentation_transforms(pil_img)
                aug_array = np.array(aug_img)

                # HWC -> CHW
                if aug_array.ndim == 2:
                    aug_array = aug_array[:, :, np.newaxis]

                aug_tensor_chw = np.moveaxis(aug_array, -1, 0)

                # Clean profile
                profile.update(
                    count=aug_tensor_chw.shape[0],
                    dtype=rasterio.uint8,
                    nodata=None  # 💥 This fixes the error
                )

                output_filename = f"{os.path.splitext(file)[0]}_aug{i+1}.tif"
                output_path = os.path.join(output_dir, output_filename)

                with rasterio.open(output_path, "w", **profile) as dst:
                    for band in range(aug_tensor_chw.shape[0]):
                        dst.write(aug_tensor_chw[band], band + 1)

                print(f"✅ Saved: {output_filename}")

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    augment_tiles(input_dir, output_dir)
