import torch
import torch.nn as nn
from transformers import CLIPVisionModel

class SimpleHead(nn.Module):
    def __init__(self, in_dim, n_classes=9):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_dim, n_classes)
    def forward(self,x):
        if x.ndim==4:
            x = self.pool(x).view(x.size(0), -1)
        return self.fc(x)

class CLIPBackbone(nn.Module):
    def __init__(self, model_name='openai/clip-vit-base-patch32'):
        super().__init__()
        self.vision = CLIPVisionModel.from_pretrained(model_name)
        self.out_dim = self.vision.config.hidden_size
    def forward(self, x):
        outputs = self.vision(pixel_values=x)
        feats = outputs.pooler_output if hasattr(outputs,'pooler_output') else outputs.last_hidden_state[:,0,:]
        return feats

class FallbackBackbone(nn.Module):
    def __init__(self, input_channels=3, out_dim=768):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels,32,3,stride=2,padding=1),
            nn.ReLU(),
            nn.Conv2d(32,64,3,stride=2,padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.out_dim = out_dim
        self.fc = nn.Linear(64, out_dim)
    def forward(self,x):
        x = self.conv(x).view(x.size(0),-1)
        x = self.fc(x)
        return x

class ClassifierModel(nn.Module):
    def __init__(self, backbone, n_classes=9):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(self.backbone.out_dim, n_classes)
    def forward(self, x):
        feats = self.backbone(x)
        logits = self.head(feats)
        return logits, feats
