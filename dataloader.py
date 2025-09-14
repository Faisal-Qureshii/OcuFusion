import glob, os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

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
    def __init__(self, base_dir, subset='train', modality='fundus', img_size=224, transform=None):
        self.base_dir = base_dir
        self.modality = modality
        txt_path = os.path.join(base_dir, subset, 'large9cls.txt')
        if not os.path.exists(txt_path):
            raise FileNotFoundError(f"Label file not found: {txt_path}")
        self.items = read_label_file(txt_path)
        image_data_dir = os.path.join(base_dir, subset, 'ImageData')

        # Auto-detect if there's exactly one subfolder inside ImageData
        subfolders = [f for f in os.listdir(image_data_dir) if os.path.isdir(os.path.join(image_data_dir, f))]
        if len(subfolders) == 1:
            self.img_root = os.path.join(image_data_dir, subfolders[0])
        else:
            self.img_root = image_data_dir

        if not os.path.isdir(self.img_root):
            raise FileNotFoundError(f"ImageData folder not found: {self.img_root}")
        self.transform = transform or transforms.Compose([
            transforms.Resize((img_size,img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
        ])
    def __len__(self):
        return len(self.items)
    def __getitem__(self, idx):
        img_name, label = self.items[idx]
        img_path = os.path.join(self.img_root, img_name)
        if not os.path.exists(img_path):
            possible_files = glob.glob(img_path + ".*")
            if len(possible_files) > 0:
                img_path = possible_files[0]  # pick first matching file
            else:
                raise FileNotFoundError(f"Image not found: {img_path}")

        img = Image.open(img_path).convert('RGB')

        img = self.transform(img)
        return img, label, img_name

def make_loader(base_dir, subset='train', modality='fundus', img_size=224, batch_size=16, shuffle=True):
    ds = RetinalDataset(base_dir, subset=subset, modality=modality, img_size=img_size)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=4, pin_memory=True)
