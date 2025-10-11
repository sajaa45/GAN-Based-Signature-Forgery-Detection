import os
import torch
import torchvision.utils as vutils
from train_gan import Generator

def generate_fakes(output_dir="data/generated_samples", latent_dim=100, num_samples=500):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    generator = Generator(latent_dim).to(device)
    generator.load_state_dict(torch.load("../models/generator.pth", map_location=device))
    generator.eval()

    with torch.no_grad():
        for i in range(num_samples):
            z = torch.randn(1, latent_dim, device=device)
            fake = generator(z)
            fake = (fake + 1) / 2  # Scale from [-1,1] to [0,1]
            vutils.save_image(fake, f"{output_dir}/fake_{i+1:04d}.png")
    print(f"Generated and saved {num_samples} fake signatures in {output_dir}")

if __name__ == "__main__":
    generate_fakes()
