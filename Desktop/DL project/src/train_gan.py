import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np

# --- Dataset for loading preprocessed images ---
class PreprocessedImageDataset(Dataset):
    def __init__(self, root_dir):
        self.image_paths = [os.path.join(root_dir, f) for f in os.listdir(root_dir)
                            if os.path.isfile(os.path.join(root_dir, f))]
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
        if img is None:
            # If image can't be loaded, return a blank 64x64 image (or handle as you wish)
            img = np.zeros((64, 64), dtype=np.float32)
        img = cv2.resize(img, (64, 64))
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5  # Normalize to [-1, 1]
        img = np.expand_dims(img, axis=0)  # (1, 64, 64)
        return torch.tensor(img)

# --- DCGAN-style Generator ---
class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),  # (512, 4, 4)
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),         # (256, 8, 8)
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),         # (128, 16, 16)
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),          # (64, 32, 32)
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),            # (1, 64, 64)
            nn.Tanh()
        )

    def forward(self, z):
        z = z.view(z.size(0), z.size(1), 1, 1)
        return self.model(z)

# --- DCGAN-style Discriminator ---
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 1),  # Corrected: 256*4*4 = 4096
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.model(x)

def get_dataloader(data_dir, batch_size=64):
    dataset = PreprocessedImageDataset(data_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def train_gan():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latent_dim = 100
    batch_size = 64
    epochs = 50

    data_dir = os.path.join("data", "preprocessed", "train", "genuine")
    dataloader = get_dataloader(data_dir, batch_size)

    generator = Generator(latent_dim).to(device)
    discriminator = Discriminator().to(device)

    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    for epoch in range(epochs):
        for imgs in dataloader:
            imgs = imgs.to(device)

            valid = torch.ones(imgs.size(0), 1, device=device)
            fake = torch.zeros(imgs.size(0), 1, device=device)

            # Train Generator
            optimizer_G.zero_grad()
            z = torch.randn(imgs.size(0), latent_dim, device=device)
            gen_imgs = generator(z)
            g_loss = criterion(discriminator(gen_imgs), valid)
            g_loss.backward()
            optimizer_G.step()

            # Train Discriminator
            optimizer_D.zero_grad()
            real_loss = criterion(discriminator(imgs), valid)
            fake_loss = criterion(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()


        print(f"Epoch [{epoch+1}/{epochs}]  D loss: {d_loss.item():.4f}  G loss: {g_loss.item():.4f}")

    # Save models
    os.makedirs("../models", exist_ok=True)
    torch.save(generator.state_dict(), "../models/generator.pth")
    torch.save(discriminator.state_dict(), "../models/discriminator.pth")
    print("Training complete. Models saved.")

if __name__ == "__main__":
    train_gan()
