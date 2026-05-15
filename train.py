"""
Training script for DR Grading with EfficientNet + CORAL + Ben Graham.
Usage: python train.py --config config.yaml --coral --ben-graham
"""
import argparse
import os
import json
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.model import build_model
from src.dataset import APTOSDataset, APTOSDatasetBG, get_transforms
from src.utils import set_seed, load_config, coral_loss, predict_grade_from_coral, compute_metrics


def get_sampler(labels):
    class_counts = pd.Series(labels).value_counts().sort_index()
    weights = 1.0 / class_counts
    sample_weights = [weights[l] for l in labels]
    return WeightedRandomSampler(sample_weights, len(sample_weights))


def train_one_epoch(model, loader, optimizer, device, coral=False):
    model.train()
    total_loss = 0
    for imgs, labels in tqdm(loader, leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        if coral:
            loss = coral_loss(outputs, labels)
        else:
            loss = torch.nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device, coral=False):
    model.eval()
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        if coral:
            preds = predict_grade_from_coral(outputs).cpu()
        else:
            preds = outputs.argmax(dim=1).cpu()
        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())
    return compute_metrics(all_labels, all_preds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--coral", action="store_true")
    parser.add_argument("--ben-graham", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = pd.read_csv(os.path.join(cfg["data"]["data_dir"], "train.csv"))
    train_df, val_df = train_test_split(df, test_size=cfg["data"]["val_split"],
                                        stratify=df["diagnosis"],
                                        random_state=cfg["training"]["seed"])

    DatasetClass = APTOSDatasetBG if args.ben_graham else APTOSDataset
    train_ds = DatasetClass(train_df, cfg["data"]["img_dir"], get_transforms(augment=True))
    val_ds   = DatasetClass(val_df,   cfg["data"]["img_dir"], get_transforms(augment=False))

    sampler = get_sampler(train_df["diagnosis"].tolist())
    train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"], sampler=sampler)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["training"]["batch_size"], shuffle=False)

    model = build_model(coral=args.coral).to(device)

    # Stage 1 — head only
    for p in model.features.parameters():
        p.requires_grad = False
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=cfg["training"]["stage1_lr"])
    print("Stage 1: Training head only...")
    for epoch in range(cfg["training"]["stage1_epochs"]):
        loss = train_one_epoch(model, train_loader, optimizer, device, args.coral)
        metrics = evaluate(model, val_loader, device, args.coral)
        print(f"  Epoch {epoch+1} | Loss: {loss:.4f} | QWK: {metrics['qwk']:.4f} | Acc: {metrics['accuracy']:.4f}")

    # Stage 2 — full fine-tune
    for p in model.parameters():
        p.requires_grad = True
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=cfg["training"]["stage2_lr"],
                                  weight_decay=cfg["training"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg["training"]["stage2_epochs"])
    best_qwk, patience = 0, 0
    os.makedirs(cfg["data"]["checkpoint_dir"], exist_ok=True)

    print("Stage 2: Full fine-tuning...")
    for epoch in range(cfg["training"]["stage2_epochs"]):
        loss = train_one_epoch(model, train_loader, optimizer, device, args.coral)
        metrics = evaluate(model, val_loader, device, args.coral)
        scheduler.step()
        print(f"  Epoch {epoch+1} | Loss: {loss:.4f} | QWK: {metrics['qwk']:.4f} | Acc: {metrics['accuracy']:.4f}")
        if metrics["qwk"] > best_qwk:
            best_qwk = metrics["qwk"]
            patience = 0
            torch.save(model.state_dict(), os.path.join(cfg["data"]["checkpoint_dir"], "best_model.pth"))
            print(f"  ✅ Saved best model (QWK={best_qwk:.4f})")
        else:
            patience += 1
            if patience >= cfg["training"]["early_stopping_patience"]:
                print("Early stopping.")
                break

    print(f"\nDone! Best QWK: {best_qwk:.4f}")


if __name__ == "__main__":
    main()
