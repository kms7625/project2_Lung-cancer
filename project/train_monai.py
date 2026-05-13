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



# ── Dataset ────────────────────────────────────────
class NoduleDataset(Dataset):
    def __init__(self, df, split, augment=False):
        self.data    = df[df['split'] == split].reset_index(drop=True)
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        vol = np.load(row['image_path']).astype(np.float32)

        if self.augment:
            if np.random.rand() > 0.5:
                vol = np.flip(vol, axis=0).copy()
            if np.random.rand() > 0.5:
                vol = np.flip(vol, axis=1).copy()
            if np.random.rand() > 0.5:
                vol = np.flip(vol, axis=2).copy()
            k = np.random.randint(0, 4)
            vol = np.rot90(vol, k=k, axes=(1, 2)).copy()

        vol   = torch.tensor(vol).unsqueeze(0)
        label = torch.tensor(int(row['label']), dtype=torch.long)
        return vol, label

# ── DataLoader ─────────────────────────────────────
df = pd.read_csv(CSV_PATH)
train_ds = NoduleDataset(df, 'train', augment=True)
val_ds   = NoduleDataset(df, 'val',   augment=False)
test_ds  = NoduleDataset(df, 'test',  augment=False)

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

        correct += (logits.argmax(1) == labels).sum().item()
        total += len(labels)

    train_loss = total_loss / total
    train_acc = correct / total
    val_metrics = evaluate(model, val_loader, device)
    scheduler.step()

    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_metrics['accuracy'])
    history['val_auc'].append(val_metrics['auc'])

    flag = ''
    if val_metrics['auc'] > best_auc:
        best_auc = val_metrics['auc']
        torch.save(model.state_dict(), f'{OUT_DIR}/best_model_monai.pth')
        flag = ' ← best'

    print(f'Epoch {epoch:3d}/{EPOCHS} | '
          f'Loss: {train_loss:.4f} | '
          f'Train Acc: {train_acc * 100:.2f}% | '
          f'Val Acc: {val_metrics["accuracy"] * 100:.2f}% | '
          f'Val AUC: {val_metrics["auc"] * 100:.2f}%{flag}')

with open(f'{OUT_DIR}/history.json', 'w') as f:
    json.dump(history, f, indent=2)

print('\n' + '=' * 50)
model.load_state_dict(torch.load(f'{OUT_DIR}/best_model_monai.pth',
                                 map_location=device, weights_only=True))
test_metrics = evaluate(model, test_loader, device)
print('[ Test 최종 결과 ]')
print(f'  Accuracy   : {test_metrics["accuracy"]*100:.2f}%')
print(f'  AUC        : {test_metrics["auc"]*100:.2f}%')
print(f'  Sensitivity: {test_metrics["sensitivity"]*100:.2f}%')
print(f'  Specificity: {test_metrics["specificity"]*100:.2f}%')
print(f'  F1         : {test_metrics["f1"]*100:.2f}%')
print('=' * 50)

with open(f'{OUT_DIR}/test_results.json', 'w') as f:
    json.dump(test_metrics, f, indent=2)
print(f'결과 저장: {OUT_DIR}/test_results.json')