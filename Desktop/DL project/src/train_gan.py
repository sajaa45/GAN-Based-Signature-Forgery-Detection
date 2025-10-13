import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image

IMG_SIZE = 128           # Try 128 if GPU allows
BATCH_SIZE = 64
EPOCHS = 1000           # Longer training helps sharpness
LATENT_DIM = 100
LR = 1e-4
BETA1 = 0.0
BETA2 = 0.9
LAMBDA_GP = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Paths
GENUINE_DIR = os.path.join("data", "preprocessed", "train", "genuine")
SAVE_DIR = "/content/generated_samp"
os.makedirs(SAVE_DIR, exist_ok=True)

class SignatureDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.img_files = [os.path.join(root_dir, f)
                          for f in os.listdir(root_dir)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        # No resizing, just convert to tensor and normalize (images are already 128x128)
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # Converts to [C, H, W], values in [0,1]
            transforms.Normalize([0.5], [0.5])  # Scale to [-1,1]
        ])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        img = Image.open(img_path).convert("L")  # Ensure grayscale
        img = self.transform(img)
        return img


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, latent_dim, img_size):
        super().__init__()
        self.init_size = img_size // 16
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 256 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            ResidualBlock(128),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            ResidualBlock(64),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            ResidualBlock(32),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 1, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 256, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self, img_size):
        super().__init__()
        def block(in_channels, out_channels, bn=True):
            layers = [nn.Conv2d(in_channels, out_channels, 4, 2, 1)]
            if bn:
                layers.append(nn.InstanceNorm2d(out_channels, affine=True))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(1, 32, bn=False),
            *block(32, 64),
            *block(64, 128),
            *block(128, 256),
            *block(256, 512),
        )
        ds_size = img_size // 32
        self.adv_layer = nn.Linear(512 * ds_size ** 2, 1)

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity


def compute_gradient_penalty(D, real_samples, fake_samples):
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=DEVICE)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = torch.ones(d_interpolates.size(), device=DEVICE, requires_grad=False)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def train():
    dataset = SignatureDataset(GENUINE_DIR, IMG_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    generator = Generator(LATENT_DIM, IMG_SIZE).to(DEVICE)
    discriminator = Discriminator(IMG_SIZE).to(DEVICE)

    optimizer_G = optim.Adam(generator.parameters(), lr=LR, betas=(BETA1, BETA2))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=LR, betas=(BETA1, BETA2))

    print(f"Training on {DEVICE} with {len(dataset)} images...")

    for epoch in range(EPOCHS):
        for i, real_imgs in enumerate(dataloader):
            real_imgs = real_imgs.to(DEVICE)
            bs = real_imgs.size(0)

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            z = torch.randn(bs, LATENT_DIM, device=DEVICE)
            fake_imgs = generator(z).detach()

            real_validity = discriminator(real_imgs)
            fake_validity = discriminator(fake_imgs)
            gradient_penalty = compute_gradient_penalty(discriminator, real_imgs, fake_imgs)

            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + LAMBDA_GP * gradient_penalty
            d_loss.backward()
            optimizer_D.step()

            # Train the generator every 5 steps
            if i % 5 == 0:
                optimizer_G.zero_grad()
                z = torch.randn(bs, LATENT_DIM, device=DEVICE)
                gen_imgs = generator(z)
                g_loss = -torch.mean(discriminator(gen_imgs))
                g_loss.backward()
                optimizer_G.step()

        print(f"[Epoch {epoch+1}/{EPOCHS}] D loss: {d_loss.item():.4f} | G loss: {g_loss.item():.4f}")

        # Save samples
        if (epoch + 1) % 25 == 0:
            save_image_samples(generator, epoch + 1)

    torch.save(generator.state_dict(), os.path.join(SAVE_DIR, "generator_final.pth"))
    print("✅ Training complete — model saved!")

def save_image_samples(generator, epoch, n_row=5):
    z = torch.randn(n_row ** 2, LATENT_DIM, device=DEVICE)
    gen_imgs = generator(z)
    gen_imgs = gen_imgs.detach().cpu()
    utils.save_image(gen_imgs, os.path.join(SAVE_DIR, f"epoch_{epoch}.png"), nrow=n_row, normalize=True)


if __name__ == "__main__":
    train()
