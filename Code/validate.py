# validate.py

import os
import torch
from torch.utils.data import DataLoader
from datasets.sar_optical_dataset import SAROpticalDataset
from models.generator import Generator
#from utils.metrics import calculate_psnr, calculate_ssim
from torchvision.utils import save_image
from tqdm import tqdm

# --- Config ---
val_dir = "data/val"
checkpoint_path = "checkpoints/generator_latest.pth"
output_dir = "outputs/validation"
os.makedirs(output_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load model ---
model = Generator().to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

# --- Load validation data ---
val_dataset = SAROpticalDataset(val_dir)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

total_psnr = 0
total_ssim = 0
count = 0

with torch.no_grad():
    for batch in tqdm(val_loader, desc="Validating"):
        input_sar = batch["input"].to(device)
        target_rgb = batch["target"].to(device)
        filename = batch["filename"][0]

        output_rgb = model(input_sar)

        

        # Save images
        save_image(output_rgb, os.path.join(output_dir, f"{filename}_pred.png"))
        save_image(target_rgb, os.path.join(output_dir, f"{filename}_gt.png"))

print("✅ All test images generated and saved.")