import os
import torch
import torchvision.utils as vutils
from train_gan import Generator, LATENT_DIM, IMG_SIZE, DEVICE

def generate_fakes(model_path=None, output_dir=None, num_samples=500):
    """
    Generate fake images using the trained GAN generator.
    """
    device = torch.device(DEVICE)
    
    if model_path is None:
        model_path = os.path.join("data", "generator_final.pth")
    if output_dir is None:
        output_dir = os.path.join("data", "generated_fakes")

    os.makedirs(output_dir, exist_ok=True)

    generator = Generator(LATENT_DIM, IMG_SIZE).to(device)
    generator.load_state_dict(torch.load(model_path, map_location=device))
    generator.eval()

    print(f"Loaded trained generator from {model_path}")

    with torch.no_grad():
        for i in range(num_samples):
            z = torch.randn(1, LATENT_DIM, device=device)
            fake = generator(z)
            fake = (fake + 1) / 2  # Convert from [-1,1] to [0,1]
            vutils.save_image(fake, os.path.join(output_dir, f"fake_{i+1:04d}.png"))

    print(f"Generated and saved {num_samples} fake images in: {output_dir}")

if __name__ == "__main__":
    generate_fakes(num_samples=500)
