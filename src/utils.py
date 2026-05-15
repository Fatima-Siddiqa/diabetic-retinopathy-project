import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
import yaml
import random


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ben_graham_enhance(pil_img, sigma=30):
    """Ben Graham retinal contrast enhancement."""
    img = np.array(pil_img.convert("RGB"))
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    enhanced = cv2.addWeighted(img, 4, blurred, -4, 128)
    return Image.fromarray(np.clip(enhanced, 0, 255).astype(np.uint8))


def coral_loss(outputs, targets, num_classes=5):
    """CORAL ordinal loss function."""
    sets = [targets > i for i in range(num_classes - 1)]
    loss = sum(
        F.binary_cross_entropy(outputs[:, i], sets[i].float())
        for i in range(num_classes - 1)
    )
    return loss / (num_classes - 1)


def predict_grade_from_coral(outputs):
    """Convert CORAL outputs to grade predictions."""
    return (outputs > 0.5).sum(dim=1)


def compute_metrics(preds, labels):
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score,
        recall_score, cohen_kappa_score
    )
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "macro_precision": precision_score(labels, preds, average="macro", zero_division=0),
        "macro_recall": recall_score(labels, preds, average="macro", zero_division=0),
        "qwk": cohen_kappa_score(labels, preds, weights="quadratic"),
    }
