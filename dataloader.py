import glob, os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import multiprocessing

def read_label_file(path):
    data = []
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                img, lbl = parts[0], int(parts[1])
                data.append((img, lbl))
    return data

class RetinalDataset(Dataset):
    def __init__(self, base_dir, subset='train', modality='fundus',
                 img_size=224, transform=None, label_file=None, image_root=None):
        self.base_dir = base_dir
        self.modality = modality

        # pick label file
        txt_path = label_file or os.path.join(base_dir, subset, 'large9cls.txt')
        if not os.path.exists(txt_path):
            raise FileNotFoundError(f"Label file not found: {txt_path}")
        self.items = read_label_file(txt_path)

        # pick image root (force override if provided)
        if image_root:
            self.img_root = image_root
        else:
            image_data_dir = os.path.join(base_dir, subset)
            possible = ['ImageData', 'imagedata', 'Imagedata']
            self.img_root = next(
                (os.path.join(image_data_dir, p) for p in possible if os.path.isdir(os.path.join(image_data_dir, p))),
                image_data_dir
            )

        if not os.path.isdir(self.img_root):
            raise FileNotFoundError(f"ImageData folder not found: {self.img_root}")

        # set transforms
        if transform:
            self.transform = transform
        else:
            if subset == 'train':
                self.transform = transforms.Compose([
                    transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
                    transforms.RandomRotation(10),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize((img_size, img_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_name, label = self.items[idx]
        img_path = os.path.join(self.img_root, img_name)

        # try exact match + extensions
        if not os.path.exists(img_path):
            found = None
            for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".JPG", ".PNG"]:
                candidate = os.path.join(self.img_root, img_name + ext)
                if os.path.exists(candidate):
                    found = candidate
                    break
            if not found:
                # last fallback: glob search
                matches = glob.glob(os.path.join(self.img_root, img_name + ".*"))
                if matches:
                    found = matches[0]
            if found:
                img_path = found
            else:
                raise FileNotFoundError(f"Image not found: {img_name} in {self.img_root}")

        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        return img, label, img_name


def make_loader(base_dir, subset='train', modality='fundus',
                img_size=224, batch_size=16, shuffle=True,
                num_workers=None, label_file=None, image_root=None):
    if num_workers is None:
        num_workers = min(8, multiprocessing.cpu_count())
    ds = RetinalDataset(base_dir, subset=subset, modality=modality,
                        img_size=img_size, transform=None,
                        label_file=label_file, image_root=image_root)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True)
