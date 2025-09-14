import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftCrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, inputs, targets):
        if targets.ndim==1:
            return F.cross_entropy(inputs, targets)
        else:
            logp = F.log_softmax(inputs, dim=1)
            return -(targets * logp).sum(dim=1).mean()

def global_prototypical_distillation(student_feats, teacher_feats, labels, num_classes=9):
    device = student_feats.device
    prototypes = torch.zeros((num_classes, teacher_feats.size(1)), device=device)
    for c in range(num_classes):
        mask = labels==c
        if mask.sum()>0:
            prototypes[c] = teacher_feats[mask].mean(dim=0)
    proto_for_sample = prototypes[labels]
    loss = F.mse_loss(student_feats, proto_for_sample)
    return loss

def local_contrastive_distillation(student_feats, teacher_feats, labels, temperature=0.1):
    s = F.normalize(student_feats, dim=1)
    t = F.normalize(teacher_feats, dim=1)
    logits = torch.matmul(s, t.T) / temperature  # (B,B)
    labels_matrix = (labels.unsqueeze(1) == labels.unsqueeze(0)).long()
    exp_logits = torch.exp(logits)
    denom = exp_logits.sum(dim=1)
    pos_mask = labels_matrix.bool()
    pos_exp = exp_logits * pos_mask.float()
    pos_sum = pos_exp.sum(dim=1)
    loss = -torch.log(pos_sum / denom + 1e-8).mean()
    return loss
