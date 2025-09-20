import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torch.nn.functional as F
from datasets.sar_optical_dataset import SAROpticalDataset
from datasets.transform import get_default_transforms
from models.generator import Generator
from models.discriminator import PatchGANDiscriminator
from utils.metrics import psnr, ssim
import yaml
from datetime import datetime
from tqdm import tqdm

# Load config
with open("configs/config.yaml", 'r') as f:
    config = yaml.safe_load(f)

# Training params
LR = config['training']['lr']
BATCH_SIZE = config['training']['batch_size']
EPOCHS = config['training']['epochs']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
TRAIN_DIR = config['paths']['train_dir']
VAL_DIR = config['paths']['val_dir']
CHECKPOINT_DIR = config['paths']['checkpoint_dir']
LOG_INTERVAL = config['training']['log_interval']

# Prepare datasets and loaders
train_dataset = SAROpticalDataset(TRAIN_DIR)
val_dataset = SAROpticalDataset(VAL_DIR)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1)

# Initialize models
generator = Generator().to(DEVICE)
discriminator = PatchGANDiscriminator().to(DEVICE)

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=LR, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=LR, betas=(0.5, 0.999))

# Loss functions
bce_loss = nn.BCELoss()
l1_loss = nn.L1Loss()

print("🚀 Starting Training...")

for epoch in range(EPOCHS):
    generator.train()
    discriminator.train()

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        sar = batch['input'].to(DEVICE)            # [B, 2, H, W]
        real_optical = batch['target'].to(DEVICE)  # [B, 3, H, W]

        # Train Generator
        optimizer_G.zero_grad()
        fake_optical = generator(sar)

        # ✅ DEBUG: Check for invalid values
        if not torch.isfinite(fake_optical).all():
            print("❌ Non-finite values in fake_optical")
            continue

        # ✅ Ensure output is in range [0, 1]
        fake_optical = torch.clamp(fake_optical, 0.0, 1.0)
        real_optical = torch.clamp(real_optical, 0.0, 1.0)

        # Downsample generated optical to match SAR resolution for discriminator
        fake_optical_down = F.interpolate(fake_optical, size=sar.shape[2:], mode='bilinear', align_corners=False)

        fake_input = torch.cat([sar, fake_optical_down], dim=1)
        real_input = torch.cat([sar, real_optical], dim=1)

        pred_fake = discriminator(fake_input)
        valid = torch.ones_like(pred_fake).to(DEVICE)

        # ✅ Apply sigmoid manually before BCE loss
        g_adv = bce_loss(torch.sigmoid(pred_fake), valid)
        g_l1 = l1_loss(fake_optical, real_optical)
        g_loss = g_adv + 100 * g_l1
        g_loss.backward()
        optimizer_G.step()

        # Train Discriminator
        optimizer_D.zero_grad()
        pred_real = discriminator(real_input)
        pred_fake_detach = discriminator(fake_input.detach())

        valid = torch.ones_like(pred_real).to(DEVICE)
        fake = torch.zeros_like(pred_fake_detach).to(DEVICE)

        d_real_loss = bce_loss(torch.sigmoid(pred_real), valid)
        d_fake_loss = bce_loss(torch.sigmoid(pred_fake_detach), fake)
        d_loss = 0.5 * (d_real_loss + d_fake_loss)
        d_loss.backward()
        optimizer_D.step()

    # Validation
    generator.eval()
    with torch.no_grad():
        for val_batch in val_loader:
            val_sar = val_batch['input'].to(DEVICE)
            val_real_optical = val_batch['target'].to(DEVICE)
            val_fake_optical = generator(val_sar)

            # Clamp values for metrics safety
            val_fake_optical = torch.clamp(val_fake_optical, 0, 1)
            val_real_optical = torch.clamp(val_real_optical, 0, 1)

            psnr_score = psnr(val_fake_optical, val_real_optical)
            ssim_score = ssim(val_fake_optical, val_real_optical)

    print(f"📸 Epoch {epoch+1}/{EPOCHS} | G Loss: {g_loss.item():.4f} | D Loss: {d_loss.item():.4f} | PSNR: {psnr_score:.2f} | SSIM: {ssim_score:.4f}")

    # Save checkpoints
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.save(generator.state_dict(), os.path.join(CHECKPOINT_DIR, "generator_latest.pth"))
    torch.save(discriminator.state_dict(), os.path.join(CHECKPOINT_DIR, "discriminator_latest.pth"))

    # Save sample image
    if (epoch + 1) % LOG_INTERVAL == 0:
        sample_dir = "outputs"
        os.makedirs(sample_dir, exist_ok=True)
        save_image(val_fake_optical, os.path.join(sample_dir, f"fake_epoch_{epoch+1}.png"))
        save_image(val_real_optical, os.path.join(sample_dir, f"real_epoch_{epoch+1}.png"))

print("✅ Training Complete.")
