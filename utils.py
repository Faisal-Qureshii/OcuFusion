import torch, os
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for servers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import itertools

def save_checkpoint(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

def load_checkpoint(model, path):
    state = torch.load(path, map_location='cpu')
    model.load_state_dict(state)
    return model

def load_model_for_inference(path, device='cpu'):
    # reconstruct fallback student architecture for inference
    from models import FallbackBackbone, ClassifierModel
    bb = FallbackBackbone()
    model = ClassifierModel(bb, n_classes=9)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    return model

# -------------------------
# plotting helpers
# -------------------------
def plot_confusion_matrix(cm, classes, save_path='analysis/confusion.png', title='Confusion matrix', cmap=None):
    """
    Plots and saves confusion matrix.
    cm : numpy array (C,C)
    classes : list of class labels (strings)
    """
    plt.figure(figsize=(8,6))
    if cmap is None:
        cmap = plt.cm.Blues
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # annotate each cell
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()

def plot_tsne(features: np.ndarray, labels: list, save_path='analysis/tsne.png', title='t-SNE Features (2D)'):
    """
    features : np.ndarray (N, D)
    labels : list or np.array length N
    """
    if features.shape[0] > 2000:
        # subsample for speed/clarity
        idx = np.random.choice(features.shape[0], 2000, replace=False)
        feats = features[idx]
        labs = np.array(labels)[idx]
    else:
        feats = features
        labs = np.array(labels)

    tsne = TSNE(n_components=2, perplexity=30, init='random', random_state=42)
    reduced = tsne.fit_transform(feats)

    plt.figure(figsize=(8,6))
    unique_labels = np.unique(labs)
    for lab in unique_labels:
        mask = labs == lab
        plt.scatter(reduced[mask,0], reduced[mask,1], label=str(lab), alpha=0.6, s=10)
    plt.legend(markerscale=2, bbox_to_anchor=(1.05,1), loc='upper left', fontsize='small')
    plt.title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
