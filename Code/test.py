# test.py

import os
import torch
from torch.utils.data import DataLoader
from datasets.sar_optical_dataset import SAROpticalDataset
from models.generator import Generator
from torchvision.utils import save_image
from tqdm import tqdm

# --- Config ---
test_dir = "data/test"
checkpoint_path = "checkpoints/generator_latest.pth"
output_dir = "outputs/test"
os.makedirs(output_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load model ---
model = Generator().to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

# --- Load test data ---
test_dataset = SAROpticalDataset(test_dir)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Testing"):
        input_sar = batch["input"].to(device)
        filename = batch["filename"][0]

        output_rgb = model(input_sar)

        # Save output image
        save_image(output_rgb, os.path.join(output_dir, f"{filename}_generated.png"))

print("✅ All test images generated and saved.")
