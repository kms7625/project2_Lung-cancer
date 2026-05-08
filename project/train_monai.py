# [목적] MONAI 라이브러리를 사용하여 3D 폐 결절 분류 모델을 학습한다.
# 직접 구현한 ResNet3D 대신 MONAI의 DenseNet121(3D)을 사용하고,
# MONAI의 의료 특화 데이터 증강을 적용하여 성능을 향상시킨다.

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score, classification_report

from monai.networks.nets import DenseNet121
from monai.transforms import (
    Compose,
    RandFlipd,
    RandRotate90d,
    RandGaussianNoised,
    RandZoomd,
    NormalizeIntensityd,
    ToTensord,
)

# ── 설정 ───────────────────────────────────────────
CSV_PATH = '/home/kms/resnet_project/lidc-idri/labels_3d.csv'
OUT_DIR  = '/home/kms/resnet_project/lidc-idri/checkpoints/monai_densenet'
EPOCHS   = 50
BATCH    = 8
LR       = 0.01
SEED     = 42
os.makedirs(OUT_DIR, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)

# ── MONAI Transform 정의 ────────────────────────────
train_transforms = Compose([
    RandFlipd(keys=['image'], prob=0.5, spatial_axis=0),
    RandFlipd(keys=['image'], prob=0.5, spatial_axis=1),
    RandFlipd(keys=['image'], prob=0.5, spatial_axis=2),
    RandRotate90d(keys=['image'], prob=0.5, spatial_axes=(1, 2)),
    RandGaussianNoised(keys=['image'], prob=0.3, std=0.01),
    RandZoomd(keys=['image'], prob=0.3, min_zoom=0.9, max_zoom=1.1),
    NormalizeIntensityd(keys=['image']),
    ToTensord(keys=['image']),
])

val_transforms = Compose([
    NormalizeIntensityd(keys=['image']),
    ToTensord(keys=['image']),
])

# ── Dataset ────────────────────────────────────────
class NoduleDataset(Dataset):
    def __init__(self, df, split, transforms=None):
        self.data       = df[df['split'] == split].reset_index(drop=True)
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        vol = np.load(row['image_path']).astype(np.float32)  # (64,64,64)
        vol = vol[np.newaxis]  # (1, 64, 64, 64)

        data = {'image': vol}
        if self.transforms:
            data = self.transforms(data)

        image = data['image']
        if not isinstance(image, torch.Tensor):
            image = torch.tensor(image)

        label = torch.tensor(int(row['label']), dtype=torch.long)
        return image, label

# ── DataLoader ─────────────────────────────────────
df = pd.read_csv(CSV_PATH)
train_ds = NoduleDataset(df, 'train', train_transforms)
val_ds   = NoduleDataset(df, 'val',   val_transforms)
test_ds  = NoduleDataset(df, 'test',  val_transforms)

labels       = train_ds.data['label'].values
class_counts = np.bincount(labels)
weights      = 1.0 / class_counts[labels]
sampler      = WeightedRandomSampler(weights, len(weights), replacement=True)

train_loader = DataLoader(train_ds, batch_size=BATCH, sampler=sampler,  num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False,    num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False,    num_workers=4, pin_memory=True)

print(f'Train: {len(train_ds)}개 | Val: {len(val_ds)}개 | Test: {len(test_ds)}개')

# ── 모델 ────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# MONAI DenseNet121 3D
model = DenseNet121(
    spatial_dims=3,   # 3D
    in_channels=1,    # grayscale CT
    out_channels=2,   # 양성/악성
).to(device)

n_params = sum(p.numel() for p in model.parameters())
print(f'파라미터 수: {n_params:,}')

# class weight
class_weights = torch.tensor(1.0 / class_counts, dtype=torch.float).to(device)
criterion     = nn.CrossEntropyLoss(weight=class_weights)
optimizer     = SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=1e-4)
scheduler     = CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ── 평가 함수 ───────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_labels, all_probs, all_preds = [], [], []
    for vols, labels in loader:
        vols   = vols.to(device)
        logits = model(vols)
        probs  = F.softmax(logits, dim=1).cpu().numpy()
        preds  = logits.argmax(1).cpu().numpy()
        all_labels.extend(labels.numpy())
        all_probs.extend(probs)
        all_preds.extend(preds)

    all_probs = np.array(all_probs)
    auc    = roc_auc_score(all_labels, all_probs[:, 1])
    report = classification_report(
        all_labels, all_preds,
        target_names=['Benign', 'Malignant'],
        output_dict=True, zero_division=0
    )
    return {
        'accuracy':    report['accuracy'],
        'auc':         auc,
        'sensitivity': report['Malignant']['recall'],
        'specificity': report['Benign']['recall'],
        'f1':          report['Malignant']['f1-score'],
    }

# ── 학습 ────────────────────────────────────────────
best_auc = 0.0
history  = {'train_loss': [], 'train_acc': [], 'val_acc': [], 'val_auc': []}

print(f'\n학습 시작 | Epochs: {EPOCHS} | Batch: {BATCH} | LR: {LR}')
print('-' * 70)

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for vols, labels in train_loader:
        vols, labels = vols.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(vols)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)