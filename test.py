import os
import torch
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from torchvision import transforms
from models import ClassifierModel, FallbackBackbone
from utils import load_checkpoint
import glob

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------- Build Student Model --------
def build_model(num_classes=9, img_size=224):
    backbone = FallbackBackbone()
    model = ClassifierModel(backbone, n_classes=num_classes)
    return model

# -------- Load Dataset --------
def load_dataset(img_dir, label_file, img_size=224):
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    images, labels = [], []
    with open(label_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            img_name, lbl = parts[0], int(parts[1])

            # Try with extension automatically
            pattern = os.path.join(img_dir, img_name + ".*")
            matches = glob.glob(pattern)
            if len(matches) == 0:
                print(f"[WARN] No match for {img_name}")
                continue
            img_path = matches[0]  # first found

            try:
                img = Image.open(img_path).convert("RGB")
                img = transform(img)
                images.append(img)
                labels.append(lbl)
            except Exception as e:
                print(f"[ERROR] Failed to load {img_path}: {e}")

    if len(images) == 0:
        raise RuntimeError(f"No images loaded! Check paths. img_dir={img_dir}, label_file={label_file}")

    return torch.stack(images), torch.tensor(labels)



# -------- Evaluation --------
def evaluate(model_path, img_dir, label_file, img_size=224):
    print(f"Loading model from {model_path}...")
    model = build_model()
    model = load_checkpoint(model, model_path)
    model.to(DEVICE)
    model.eval()

    images, labels = load_dataset(img_dir, label_file, img_size)
    images, labels = images.to(DEVICE), labels.to(DEVICE)

    with torch.no_grad():
        logits, _ = model(images)
        preds = torch.argmax(logits, dim=1)

    acc = accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())
    f1 = f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average="macro")

    print(f"\n✅ Accuracy: {acc*100:.2f}%")
    print(f"✅ Macro F1 Score: {f1:.4f}")

# -------- Main --------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate trained student model on Fundus images")
    parser.add_argument("--model_path", type=str, default="./checkpoints/student/best_student.pth",
                        help="Path to trained model checkpoint")
    parser.add_argument("--img_dir", type=str, required=True,
                        help="Path to image folder (e.g., ...\\ImageData\\cfp-clahe-224x224)")
    parser.add_argument("--label_file", type=str, required=True,
                        help="Path to label file (e.g., ...\\train\\large9cls.txt)")
    parser.add_argument("--img_size", type=int, default=224, help="Image size to resize")
    args = parser.parse_args()

    evaluate(args.model_path, args.img_dir, args.label_file, img_size=args.img_size)

