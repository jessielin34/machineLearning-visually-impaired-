import os
import argparse
from typing import Tuple, List

import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    top_k_accuracy_score,
)
import matplotlib.pyplot as plt


# -------------------------------
# Dataset
# -------------------------------

class VizWizClassificationDataset(Dataset):
    """
    Simple dataset wrapper.
    Assumes a CSV file with columns:
        image,label
    where:
        - image is a filename in img_root
        - label is an integer class id in [0, num_classes-1]
    """

    def __init__(self, csv_path: str, img_root: str, transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_root = img_root
        self.transform = transform

        assert "image" in self.df.columns and "label" in self.df.columns, \
            "CSV must have 'image' and 'label' columns"

        self.images = self.df["image"].tolist()
        self.labels = self.df["label"].tolist()

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img_path = os.path.join(self.img_root, self.images[idx])
        img = Image.open(img_path).convert("RGB")
        label = int(self.labels[idx])

        if self.transform is not None:
            img = self.transform(img)

        return img, label


# -------------------------------
# Models
# -------------------------------

def get_model(model_name: str, num_classes: int) -> nn.Module:
    """
    Returns a PyTorch model based on the requested name.

    Supported:
        - resnet18_linear      (frozen backbone + linear classifier)
        - resnet50_finetune    (fine-tune whole model)
        - vit_b16_finetune     (Vision Transformer, if available)
    """
    model_name = model_name.lower()

    if model_name == "resnet18_linear":
        backbone = models.resnet18(pretrained=True)
        # Freeze all params
        for p in backbone.parameters():
            p.requires_grad = False
        num_feats = backbone.fc.in_features
        backbone.fc = nn.Identity()
        classifier = nn.Linear(num_feats, num_classes)
        model = nn.Sequential(backbone, classifier)

    elif model_name == "resnet50_finetune":
        model = models.resnet50(pretrained=True)
        num_feats = model.fc.in_features
        model.fc = nn.Linear(num_feats, num_classes)

    elif model_name == "vit_b16_finetune":
        # Requires torchvision >= 0.13
        try:
            vit = models.vit_b_16(pretrained=True)
        except AttributeError:
            raise RuntimeError(
                "vit_b_16 not found in torchvision.models. "
                "Upgrade torchvision or choose another model."
            )
        num_feats = vit.heads.head.in_features
        vit.heads.head = nn.Linear(num_feats, num_classes)
        model = vit

    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    return model


def get_transforms(image_size: int = 224, mode: str = "train"):
    if mode == "train":
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet stats
                std=[0.229, 0.224, 0.225],
            ),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])


# -------------------------------
# Training & Evaluation Loops
# -------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    pbar = tqdm(loader, desc="Train", leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        preds = outputs.argmax(dim=1)
        all_preds.append(preds.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)

    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    compute_confusion: bool = False,
):
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []

    for images, labels in tqdm(loader, desc="Eval", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        preds = outputs.argmax(dim=1)

        all_preds.append(preds.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)

    top1 = accuracy_score(all_labels, all_preds)
    # Top-5 accuracy (guard if classes < 5)
    k = min(5, num_classes)
    top5 = top_k_accuracy_score(all_labels, all_probs, k=k, labels=list(range(num_classes)))
    macro_f1 = f1_score(all_labels, all_preds, average="macro")

    cm = None
    if compute_confusion:
        cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))

    return {
        "top1": top1,
        "top5": top5,
        "macro_f1": macro_f1,
        "confusion_matrix": cm,
        "labels": all_labels,
        "preds": all_preds,
    }


def plot_confusion_matrix(cm: np.ndarray, num_classes: int, out_path: str):
    plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("True")

    # Just show a subset of ticks if many classes
    if num_classes <= 20:
        tick_marks = np.arange(num_classes)
        plt.xticks(tick_marks, tick_marks, rotation=90)
        plt.yticks(tick_marks, tick_marks)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


# -------------------------------
# Main script
# -------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="VizWiz Classification for Visually Impaired Project")

    # Data
    parser.add_argument("--data-root", type=str, default="data", help="Root folder for images")
    parser.add_argument("--train-csv", type=str, default="data/train.csv")
    parser.add_argument("--val-csv", type=str, default="data/val.csv")
    parser.add_argument("--test-csv", type=str, default="data/test.csv")
    parser.add_argument("--train-dir", type=str, default="data/train")
    parser.add_argument("--val-dir", type=str, default="data/val")
    parser.add_argument("--test-dir", type=str, default="data/test")
    parser.add_argument("--num-classes", type=int, required=True, help="Number of classes")

    # Model & training
    parser.add_argument("--model", type=str, default="resnet50_finetune",
                        choices=["resnet18_linear", "resnet50_finetune", "vit_b16_finetune"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    # General
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"])
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint for eval or resume")
    parser.add_argument("--out-dir", type=str, default="checkpoints")

    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model
    model = get_model(args.model, args.num_classes).to(device)

    # Data transforms
    train_tf = get_transforms(mode="train")
    eval_tf = get_transforms(mode="eval")

    if args.mode == "train":
        # Datasets
        train_ds = VizWizClassificationDataset(args.train_csv, args.train_dir, transform=train_tf)
        val_ds = VizWizClassificationDataset(args.val_csv, args.val_dir, transform=eval_tf)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

        # Loss & optimizer
        criterion = nn.CrossEntropyLoss()
        # Only trainable parameters
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        best_val_top1 = 0.0
        best_path = os.path.join(args.out_dir, f"best_{args.model}.pt")

        for epoch in range(1, args.epochs + 1):
            print(f"Epoch {epoch}/{args.epochs}")

            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, criterion, device
            )
            print(f"  Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}")

            val_metrics = evaluate(
                model, val_loader, device,
                num_classes=args.num_classes,
                compute_confusion=False,
            )
            print(
                f"  Val top1: {val_metrics['top1']:.4f}, "
                f"top5: {val_metrics['top5']:.4f}, "
                f"macro F1: {val_metrics['macro_f1']:.4f}"
            )

            # Save best model
            if val_metrics["top1"] > best_val_top1:
                best_val_top1 = val_metrics["top1"]
                torch.save(model.state_dict(), best_path)
                print(f"  Saved new best model to {best_path}")

        print(f"Training finished. Best val top-1 accuracy: {best_val_top1:.4f}")

    elif args.mode == "eval":
        assert args.checkpoint is not None, "You must provide --checkpoint in eval mode"
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))

        test_ds = VizWizClassificationDataset(args.test_csv, args.test_dir, transform=eval_tf)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

        metrics = evaluate(
            model, test_loader, device,
            num_classes=args.num_classes,
            compute_confusion=True,
        )

        print(f"Test top-1: {metrics['top1']:.4f}")
        print(f"Test top-5: {metrics['top5']:.4f}")
        print(f"Test macro F1: {metrics['macro_f1']:.4f}")

        if metrics["confusion_matrix"] is not None:
            cm_path = os.path.join(args.out_dir, f"cm_{args.model}.png")
            plot_confusion_matrix(metrics["confusion_matrix"], args.num_classes, cm_path)
            print(f"Saved confusion matrix to {cm_path}")


if __name__ == "__main__":
    main()
