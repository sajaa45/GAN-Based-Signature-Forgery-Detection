import os
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

from train_signature_classifier import SmallSignatureCNN, IMG_SIZE, DEVICE

def load_images_from_folder(folder, img_size=128):
    image_paths = [os.path.join(folder, f) for f in os.listdir(folder)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    images = []
    for path in image_paths:
        img = Image.open(path).convert("L")
        img = transform(img)
        images.append(img)
    return image_paths, torch.stack(images)

def evaluate_gan_images(cnn_path, gan_folder):
    model = SmallSignatureCNN().to(DEVICE)
    model.load_state_dict(torch.load(cnn_path, map_location=DEVICE))
    model.eval()

    image_paths, images = load_images_from_folder(gan_folder, IMG_SIZE)
    images = images.to(DEVICE)
    with torch.no_grad():
        outputs = model(images)
        preds = (outputs > 0.5).cpu().numpy().astype(int).flatten()

    genuine_count = np.sum(preds)
    forge_count = len(preds) - genuine_count

    print(f"Total GAN images evaluated: {len(preds)}")
    print(f"Predicted as Genuine: {genuine_count} ({genuine_count/len(preds)*100:.2f}%)")
    print(f"Predicted as Forge:   {forge_count} ({forge_count/len(preds)*100:.2f}%)")

    # Optionally, print filenames and predictions
    # for path, pred in zip(image_paths, preds):
    #     print(f"{os.path.basename(path)}: {'Genuine' if pred else 'Forge'}")

if __name__ == "__main__":
    # Adjust these paths as needed
    cnn_model_path = "best_signature_cnn.pth"
    gan_images_folder = os.path.join("data", "generated_fakes")  # or wherever your GAN images are
    evaluate_gan_images(cnn_model_path, gan_images_folder)
