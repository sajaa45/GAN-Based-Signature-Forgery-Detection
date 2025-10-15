import os
import random
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import confusion_matrix, classification_report

# --- Config ---
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Dataset ---
def get_image_paths(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

class SignaturePairDataset(Dataset):
    def __init__(self, real_paths, fake_paths, img_size=128, augment=False):
        self.samples = [(p, 1) for p in real_paths] + [(p, 0) for p in fake_paths]
        random.shuffle(self.samples)
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("L")
        if self.augment and random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32)

# --- CNN Classifier ---
class SmallSignatureCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),  # 64x64
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2), # 32x32
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),# 16x16
            nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),# 8x8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# --- Data Loading ---
def make_loader(real_dir, fake_dir, batch_size, shuffle=True, augment=False):
    real_paths = get_image_paths(real_dir)
    fake_paths = get_image_paths(fake_dir)
    dataset = SignaturePairDataset(real_paths, fake_paths, img_size=IMG_SIZE, augment=augment)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)

# --- Training ---
def train_classifier():
    base = os.path.join("data", "preprocessed")
    train_loader = make_loader(
        os.path.join(base, "train", "genuine"),
        os.path.join(base, "train", "fake"),
        BATCH_SIZE, shuffle=True, augment=True
    )
    val_loader = make_loader(
        os.path.join(base, "val", "genuine"),
        os.path.join(base, "val", "fake"),
        BATCH_SIZE, shuffle=False
    )
    test_loader = make_loader(
        os.path.join(base, "test", "genuine"),
        os.path.join(base, "test", "fake"),
        BATCH_SIZE, shuffle=False
    )

    model = SmallSignatureCNN().to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_val_acc = 0
    for epoch in range(EPOCHS):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
            preds = (outputs > 0.5).float()
            train_correct += (preds == labels).sum().item()
            train_total += imgs.size(0)
        train_acc = train_correct / train_total

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE).unsqueeze(1)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
                preds = (outputs > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += imgs.size(0)
        val_acc = val_correct / val_total

        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss/train_total:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss/val_total:.4f} | Val Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_signature_cnn.pth")

    # Test
    model.load_state_dict(torch.load("best_signature_cnn.pth"))
    model.eval()
    test_correct, test_total = 0, 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE).unsqueeze(1)
            outputs = model(imgs)
            preds = (outputs > 0.5).float()
            test_correct += (preds == labels).sum().item()
            test_total += imgs.size(0)
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
    print(f"Test Accuracy: {test_correct / test_total:.4f}")

    # Confusion matrix and classification report
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=["Fake", "Genuine"]))

if __name__ == "__main__":
    train_classifier()
