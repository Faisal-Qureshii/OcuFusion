import os, torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from dataloader import make_loader
from models import ClassifierModel, CLIPBackbone, FallbackBackbone
from losses import SoftCrossEntropy, global_prototypical_distillation, local_contrastive_distillation
from utils import save_checkpoint, load_checkpoint, plot_confusion_matrix, plot_tsne, load_model_for_inference
from typing import Optional, Dict
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
import numpy as np
from PIL import Image
import io
from torchvision import transforms

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class TrainingStatus:
    def __init__(self):
        self.phase = None
        self.epoch = 0
        self.loss = 0.0
        self.metrics = {}
        self.history = {'train':[], 'val':[]}
        self.stop = False
        self.last_eval = {}
    def update(self, **kwargs):
        self.__dict__.update(kwargs)
    def get_status(self):
        return {'phase':self.phase, 'epoch':self.epoch, 'loss':self.loss, 'stop_requested':self.stop}
    def get_results(self):
        return self.history
    def request_stop(self):
        self.stop = True

def build_model(backbone_name='clip', img_size=224, num_classes=9):
    if backbone_name=='clip':
        bb = CLIPBackbone()
    else:
        bb = FallbackBackbone()
    model = ClassifierModel(bb, n_classes=num_classes).to(DEVICE)
    return model

def train_teacher(data_dir, save_dir, epochs=10, batch_size=32, img_size=224, status:Optional[TrainingStatus]=None):
    os.makedirs(save_dir, exist_ok=True)
    model = build_model(backbone_name='clip', img_size=img_size)
    opt = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = SoftCrossEntropy()
    loader = make_loader(data_dir, subset='train', modality='oct', img_size=img_size, batch_size=batch_size)
    model.train()
    best_loss = 1e9
    for epoch in range(epochs):
        running=0.0
        if status and status.stop:
            print("Stop requested — exiting teacher training early.")
            break
        for imgs, labels, _ in tqdm(loader, desc=f"Teacher Epoch {epoch+1}/{epochs}"):
            if status and status.stop:
                break
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            opt.zero_grad()
            logits, feats = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            opt.step()
            running += loss.item()
        avg = running/len(loader) if len(loader)>0 else 0.0
        if status:
            status.phase='teacher'; status.epoch=epoch+1; status.loss=avg
        print(f"Epoch {epoch+1} loss={avg:.4f}")
        if avg < best_loss:
            best_loss = avg
            save_checkpoint(model, os.path.join(save_dir,'best_teacher.pth'))
    save_checkpoint(model, os.path.join(save_dir,'last_teacher.pth'))
    return os.path.join(save_dir,'best_teacher.pth')

def train_student(data_dir, teacher_ckpt, save_dir, epochs=10, batch_size=32, img_size=224, alpha=0.6, beta=0.05, status:Optional[TrainingStatus]=None):
    os.makedirs(save_dir, exist_ok=True)
    teacher = build_model(backbone_name='clip', img_size=224)
    teacher = load_checkpoint(teacher, teacher_ckpt)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad=False
    student = build_model(backbone_name='fallback', img_size=img_size)
    student.train()
    opt = optim.AdamW(student.parameters(), lr=1e-4)
    cls_criterion = SoftCrossEntropy()
    loader = make_loader(data_dir, subset='train', modality='fundus', img_size=img_size, batch_size=batch_size)
    best_loss=1e9
    for epoch in range(epochs):
        running=0.0
        if status and status.stop:
            print("Stop requested — exiting student training early.")
            break
        for imgs, labels, _ in tqdm(loader, desc=f"Student Epoch {epoch+1}/{epochs}"):
            if status and status.stop:
                break
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            student_logits, student_feats = student(imgs)
            with torch.no_grad():
                t_in = nn.functional.interpolate(imgs, size=(224,224), mode='bilinear', align_corners=False)
                t_logits, t_feats = teacher(t_in)
            cls_loss = cls_criterion(student_logits, labels)
            gpd = global_prototypical_distillation(student_feats, t_feats, labels, num_classes=9)
            lcd = local_contrastive_distillation(student_feats, t_feats, labels)
            loss = cls_loss + alpha * gpd + beta * lcd
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()
        avg = running/len(loader) if len(loader)>0 else 0.0
        if status:
            status.phase='student'; status.epoch=epoch+1; status.loss=avg
        print(f"Epoch {epoch+1} loss={avg:.4f}")
        if avg < best_loss:
            best_loss = avg
            save_checkpoint(student, os.path.join(save_dir,'best_student.pth'))
    save_checkpoint(student, os.path.join(save_dir,'last_student.pth'))
    return os.path.join(save_dir,'best_student.pth')

def start_teacher_training(data_dir, epochs, batch_size, status, extra):
    print('Starting teacher training...')
    status.phase='teacher'; status.epoch=0; status.loss=0.0; status.stop=False
    ckpt = train_teacher(data_dir, save_dir=extra.get('save_dir','./checkpoints/teacher'), epochs=epochs, batch_size=batch_size, img_size=extra.get('img_size',224), status=status)
    status.history['train'].append({'phase':'teacher','ckpt':ckpt})
    return ckpt

def start_student_training(data_dir, epochs, batch_size, status, extra):
    print('Starting student training...')
    status.phase='student'; status.epoch=0; status.loss=0.0; status.stop=False
    ckpt = train_student(data_dir, teacher_ckpt=extra.get('teacher_ckpt'), save_dir=extra.get('save_dir','./checkpoints/student'), epochs=epochs, batch_size=batch_size, img_size=extra.get('img_size',224), alpha=extra.get('alpha',0.6), beta=extra.get('beta',0.05), status=status)
    status.history['train'].append({'phase':'student','ckpt':ckpt})
    return ckpt

def predict_image(image_bytes, modality='fundus', model_path=None):
    model_path = model_path or './checkpoints/student/best_student.pth'
    model = load_model_for_inference(model_path, device='cpu')
    model.eval()
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    size = (224,224) if modality=='fundus' else (512,512)
    t = transforms.Compose([transforms.Resize(size), transforms.ToTensor(), transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])])
    x = t(img).unsqueeze(0)
    with torch.no_grad():
        logits, feats = model(x)
        pred = torch.argmax(logits, dim=1).item()
    return int(pred)

def evaluate_model(model_path: str, data_dir: str, subset: str = 'dev', modality: str = 'fundus', img_size: int = 224, batch_size: int = 32, analysis_dir: str = './analysis') -> Dict:
    os.makedirs(analysis_dir, exist_ok=True)
    model = load_checkpoint(build_model(backbone_name='fallback', img_size=img_size), model_path)
    model.to(DEVICE)
    model.eval()
    loader = make_loader(data_dir, subset=subset, modality=modality, img_size=img_size, batch_size=batch_size, shuffle=False)
    y_true, y_pred, feats_all = [], [], []
    with torch.no_grad():
        for imgs, labels, _ in tqdm(loader, desc=f"Evaluate {subset}"):
            imgs = imgs.to(DEVICE)
            logits, feats = model(imgs)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            y_pred.extend(preds.tolist())
            y_true.extend(labels.numpy().tolist())
            feats_all.append(feats.cpu().numpy())
    if len(y_true)==0:
        return {"error":"no samples found in loader"}
    feats_all = np.concatenate(feats_all, axis=0)
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    precisions, recalls, f1s, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    cm_path = os.path.join(analysis_dir, f'confusion_{os.path.basename(model_path)}_{subset}.png')
    plot_confusion_matrix(cm, classes=[str(i) for i in range(len(cm))], save_path=cm_path, title=f'Confusion Matrix ({subset})')
    tsne_path = os.path.join(analysis_dir, f'tsne_{os.path.basename(model_path)}_{subset}.png')
    plot_tsne(feats_all, labels=y_true, save_path=tsne_path, title=f't-SNE Features ({subset})')
    metrics = {
        "accuracy": float(acc),
        "per_class": [{"class": int(i), "precision": float(precisions[i]), "recall": float(recalls[i]), "f1": float(f1s[i])} for i in range(len(precisions))],
        "macro": {"precision": float(macro_prec), "recall": float(macro_rec), "f1": float(macro_f1)},
        "confusion_matrix_path": cm_path,
        "tsne_path": tsne_path
    }
    return metrics

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MultiEYE Training Script")
    parser.add_argument("--phase", type=str, choices=["teacher", "student"], required=True,
                        help="Training phase: 'teacher' for OCT pretraining, 'student' for Fundus distillation")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to dataset directory")
    parser.add_argument("--save_dir", type=str, default="./checkpoints",
                        help="Directory to save model checkpoints")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--img_size", type=int, default=None, help="Image size override")
    parser.add_argument("--teacher_ckpt", type=str, help="Path to pretrained teacher model (for student phase)")

    args = parser.parse_args()

    status = TrainingStatus()

    if args.phase == "teacher":
        img_size = args.img_size or 224
        start_teacher_training(args.data_dir, args.epochs, args.batch_size, status,
                               extra={"save_dir": args.save_dir, "img_size": img_size})
    elif args.phase == "student":
        if not args.teacher_ckpt:
            raise ValueError("You must provide --teacher_ckpt when training the student model.")
        img_size = args.img_size or 224
        start_student_training(args.data_dir, args.epochs, args.batch_size, status,
                               extra={"save_dir": args.save_dir, "img_size": img_size, "teacher_ckpt": args.teacher_ckpt})
