"""
LIDC-IDRI 폐 결절 분류 — EfficientNetV2-S 베이스라인

Usage:
    python train_baseline.py [--csv PATH] [--out_dir DIR] [--epochs 50] [--batch 32]

전략 A: label 0=양성(score 1-3), label 1=악성(score 4-5)
Best model 저장 기준: Val AUC
"""

import os
import argparse
import random
import json
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms, models
from sklearn.metrics import roc_auc_score, classification_report

DATA_ROOT = '/home/kms/resnet_project/lidc-idri'


# ──────────────────────────────────────────────
# 재현성
# ──────────────────────────────────────────────
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────
def _make_transform(split: str, img_size: int):
    if split == 'train':
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((img_size, img_size), antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
            transforms.RandomErasing(p=0.2),  # Normalize 후 적용 (fill=0 → normalized space 중간값)
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((img_size, img_size), antialias=True),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])


class LIDCDataset(Dataset):
    def __init__(self, df: pd.DataFrame, split: str, img_size: int = 224):
        self.data = df[df['split'] == split].reset_index(drop=True)
        self.transform = _make_transform(split, img_size)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img = np.load(row['image_path']).astype(np.float32)  # (H, W), [0,1]
        img_3ch = np.stack([img, img, img], axis=2)           # (H, W, 3)
        img_tensor = self.transform(img_3ch)
        label = torch.tensor(int(row['label']), dtype=torch.long)
        return img_tensor, label


def get_dataloaders(csv_path: str, batch_size: int = 32,
                    img_size: int = 224, num_workers: int = 4):
    df = pd.read_csv(csv_path)
    train_ds = LIDCDataset(df, 'train', img_size)
    val_ds   = LIDCDataset(df, 'val',   img_size)
    test_ds  = LIDCDataset(df, 'test',  img_size)

    # 클래스 불균형 대응 WeightedRandomSampler (비율 1.21:1 이지만 적용)
    labels = train_ds.data['label'].values
    class_counts = np.bincount(labels)
    weights = 1.0 / class_counts[labels]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              sampler=sampler, num_workers=num_workers,
                              pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)
    return train_loader, val_loader, test_loader


# ──────────────────────────────────────────────
# 모델
# ──────────────────────────────────────────────
def get_model(arch: str = 'efficientnet_v2_s', num_classes: int = 2,
              pretrained: bool = True) -> nn.Module:
    weights_map = {
        'efficientnet_v2_s': 'IMAGENET1K_V1',
        'convnext_tiny':     'IMAGENET1K_V1',
        'resnet18':          'IMAGENET1K_V1',
    }
    w = weights_map[arch] if pretrained else None

    if arch == 'efficientnet_v2_s':
        model = models.efficientnet_v2_s(weights=w)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif arch == 'convnext_tiny':
        model = models.convnext_tiny(weights=w)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    elif arch == 'resnet18':
        model = models.resnet18(weights=w)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unknown arch: {arch}")
    return model


# ──────────────────────────────────────────────
# 평가
# ──────────────────────────────────────────────
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader,
             device: torch.device, num_classes: int = 2) -> dict:
    model.eval()
    all_labels, all_probs, all_preds = [], [], []

    for imgs, labels in loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(1).cpu().numpy()
        all_labels.extend(labels.numpy())
        all_probs.extend(probs)
        all_preds.extend(preds)

    all_probs = np.array(all_probs)

    if num_classes == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
        report = classification_report(
            all_labels, all_preds,
            target_names=['Benign', 'Malignant'],
            output_dict=True, zero_division=0,
        )
        return {
            'accuracy':    report['accuracy'],
            'auc':         auc,
            'sensitivity': report['Malignant']['recall'],
            'precision':   report['Malignant']['precision'],
            'f1':          report['Malignant']['f1-score'],
        }
    else:
        auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
        report = classification_report(
            all_labels, all_preds,
            target_names=['Benign', 'Indeterminate', 'Malignant'],
            output_dict=True, zero_division=0,
        )
        return {
            'accuracy':              report['accuracy'],
            'auc':                   auc,
            'sensitivity_malignant': report['Malignant']['recall'],
            'sensitivity_indet':     report['Indeterminate']['recall'],
            'f1_macro':              report['macro avg']['f1-score'],
        }


# ──────────────────────────────────────────────
# 학습
# ──────────────────────────────────────────────
def train_one_epoch(model: nn.Module, loader: DataLoader,
                    optimizer, criterion, device: torch.device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        correct += (logits.argmax(1) == labels).sum().item()
        total += len(labels)
    return total_loss / total, correct / total


def train(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
          epochs: int = 50, lr: float = 0.01, device: torch.device = None,
          num_classes: int = 2, out_dir: str = '.') -> tuple:
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    optimizer = SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_auc = 0.0
    best_weights = None
    history = defaultdict(list)

    print(f"\n학습 시작 | Device: {device} | Epochs: {epochs} | LR: {lr}")
    print("-" * 80)

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, device, num_classes)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_auc'].append(val_metrics['auc'])

        flag = ''
        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            ckpt_path = os.path.join(out_dir, 'best_model.pth')
            torch.save(best_weights, ckpt_path)
            flag = ' ← best'

        print(f"Epoch {epoch:3d}/{epochs} | "
              f"Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | "
              f"Val Acc: {val_metrics['accuracy']:.4f} | "
              f"Val AUC: {val_metrics['auc']:.4f}{flag}")

    model.load_state_dict(best_weights)
    return model, history


# ──────────────────────────────────────────────
# 최종 평가 출력
# ──────────────────────────────────────────────
def full_evaluation(model: nn.Module, test_loader: DataLoader,
                    device: torch.device, num_classes: int = 2) -> dict:
    metrics = evaluate(model, test_loader, device, num_classes)
    print("\n" + "=" * 50)
    print("[ Test Set 최종 평가 ]")
    print(f"  Accuracy : {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.2f}%)")
    print(f"  AUC      : {metrics['auc']:.4f}")
    if num_classes == 2:
        print(f"  Sensitivity (Malignant): {metrics['sensitivity']:.4f}")
        print(f"  Precision   (Malignant): {metrics['precision']:.4f}")
        print(f"  F1          (Malignant): {metrics['f1']:.4f}")
    else:
        print(f"  Sensitivity Malignant  : {metrics['sensitivity_malignant']:.4f}")
        print(f"  Sensitivity Indet.     : {metrics['sensitivity_indet']:.4f}")
        print(f"  F1 Macro               : {metrics['f1_macro']:.4f}")
    print("=" * 50)
    return metrics


# ──────────────────────────────────────────────
# 진입점
# ──────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--csv',        default=f'{DATA_ROOT}/labels.csv')
    p.add_argument('--out_dir',    default=f'{DATA_ROOT}/checkpoints/baseline')
    p.add_argument('--arch',       default='efficientnet_v2_s',
                   choices=['efficientnet_v2_s', 'convnext_tiny', 'resnet18'])
    p.add_argument('--num_classes', type=int, default=2)
    p.add_argument('--img_size',   type=int, default=224)
    p.add_argument('--epochs',     type=int, default=50)
    p.add_argument('--batch',      type=int, default=32)
    p.add_argument('--lr',         type=float, default=0.01)
    p.add_argument('--workers',    type=int, default=4)
    p.add_argument('--seed',       type=int, default=42)
    p.add_argument('--no_pretrain', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 데이터로더
    train_loader, val_loader, test_loader = get_dataloaders(
        args.csv, args.batch, args.img_size, args.workers)
    print(f"Train: {len(train_loader.dataset)}개 슬라이스")
    print(f"Val  : {len(val_loader.dataset)}개 슬라이스")
    print(f"Test : {len(test_loader.dataset)}개 슬라이스")

    # 모델
    model = get_model(args.arch, args.num_classes, pretrained=not args.no_pretrain)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"모델: {args.arch}  |  파라미터: {n_params:,}")

    # 학습
    model, history = train(
        model, train_loader, val_loader,
        epochs=args.epochs, lr=args.lr,
        device=device, num_classes=args.num_classes,
        out_dir=args.out_dir,
    )

    # 학습 곡선 저장
    history_path = os.path.join(args.out_dir, 'history.json')
    with open(history_path, 'w') as f:
        json.dump(dict(history), f, indent=2)
    print(f"\n학습 곡선 저장: {history_path}")

    # Best model 로드 후 테스트
    best_path = os.path.join(args.out_dir, 'best_model.pth')
    model.load_state_dict(torch.load(best_path, map_location=device))
    test_metrics = full_evaluation(model, test_loader, device, args.num_classes)

    # 결과 저장
    result_path = os.path.join(args.out_dir, 'test_results.json')
    with open(result_path, 'w') as f:
        json.dump({'args': vars(args), 'metrics': test_metrics}, f, indent=2)
    print(f"결과 저장: {result_path}")


if __name__ == '__main__':
    main()
