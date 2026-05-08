# [목적] 3D ResNet으로 폐 결절 양성/악성 이진 분류 모델을 학습한다.
# 입력: labels_3d.csv에서 train/val/test 분할된 64x64x64 ROI npy 파일
# 출력: best_model_3d.pth (val AUC 기준 최고 모델 저장)
# 평가: Accuracy, AUC, Sensitivity, Specificity, F1

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
import sys
sys.path.append('/home/kms/resnet_project')
from model import ResNet3D

# ── 설정 ───────────────────────────────────────────
CSV_PATH  = '/home/kms/resnet_project/lidc-idri/labels_3d.csv'
OUT_DIR   = '/home/kms/resnet_project/lidc-idri/checkpoints/3d_resnet'
EPOCHS    = 50
BATCH     = 8   # 3D라서 메모리 많이 써서 작게 설정
LR        = 0.01
SEED      = 42
os.makedirs(OUT_DIR, exist_ok=True)

# ── 재현성 ──────────────────────────────────────────
torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Dataset ────────────────────────────────────────
class NoduleDataset3D(Dataset):
    def __init__(self, df, split, augment=False):
        self.data    = df[df['split'] == split].reset_index(drop=True)
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        vol = np.load(row['image_path']).astype(np.float32)  # (64,64,64)

        # 데이터 증강 (train만)
        if self.augment:
            # 축 방향 랜덤 flip
            if np.random.rand() > 0.5:
                vol = np.flip(vol, axis=0).copy()
            if np.random.rand() > 0.5:
                vol = np.flip(vol, axis=1).copy()
            if np.random.rand() > 0.5:
                vol = np.flip(vol, axis=2).copy()
            # 90도 회전
            k = np.random.randint(0, 4)
            vol = np.rot90(vol, k=k, axes=(1, 2)).copy()

        vol = torch.tensor(vol).unsqueeze(0)  # (1, 64, 64, 64)
        label = torch.tensor(int(row['label']), dtype=torch.long)
        return vol, label

# ── DataLoader ─────────────────────────────────────
df = pd.read_csv(CSV_PATH)
train_ds = NoduleDataset3D(df, 'train', augment=True)
val_ds   = NoduleDataset3D(df, 'val',   augment=False)
test_ds  = NoduleDataset3D(df, 'test',  augment=False)

# 클래스 불균형 처리
labels = train_ds.data['label'].values
class_counts = np.bincount(labels)
weights = 1.0 / class_counts[labels]
sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

train_loader = DataLoader(train_ds, batch_size=BATCH, sampler=sampler, num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False,   num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False,   num_workers=4, pin_memory=True)

print(f'Train: {len(train_ds)}개 | Val: {len(val_ds)}개 | Test: {len(test_ds)}개')

# ── 평가 함수 ───────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_labels, all_probs, all_preds = [], [], []
    for vols, labels in loader:
        vols = vols.to(device)
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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

model     = ResNet3D(num_classes=2).to(device)
optimizer = SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
# criterion = nn.CrossEntropyLoss()
# 양성(0) 클래스가 적어서 모델이 악성으로 치우치는 문제를 해결
# 클래스 수에 반비례한 weight를 Loss에 적용해 균형맞춤
class_counts = np.bincount(df[df['split']=='train']['label'].values)
class_weights = torch.tensor(1.0 / class_counts, dtype=torch.float).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

best_auc = 0.0
history  = {'train_loss': [], 'train_acc': [], 'val_acc': [], 'val_auc': []}

print(f'\n학습 시작 | Epochs: {EPOCHS} | Batch: {BATCH} | LR: {LR}')
print('-' * 70)

for epoch in range(1, EPOCHS + 1):
    # train
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
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += len(labels)

    train_loss = total_loss / total
    train_acc  = correct / total
    val_metrics = evaluate(model, val_loader, device)
    scheduler.step()

    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_metrics['accuracy'])
    history['val_auc'].append(val_metrics['auc'])

    flag = ''
    if val_metrics['auc'] > best_auc:
        best_auc = val_metrics['auc']
        torch.save(model.state_dict(), f'{OUT_DIR}/best_model_3d.pth')
        flag = ' ← best'

    print(f'Epoch {epoch:3d}/{EPOCHS} | '
          f'Loss: {train_loss:.4f} | '
          f'Train Acc: {train_acc:.4f} | '
          f'Val Acc: {val_metrics["accuracy"]:.4f} | '
          f'Val AUC: {val_metrics["auc"]:.4f}{flag}')

# 학습 곡선 저장
with open(f'{OUT_DIR}/history.json', 'w') as f:
    json.dump(history, f, indent=2)

# 최종 테스트
print('\n' + '=' * 50)
model.load_state_dict(torch.load(f'{OUT_DIR}/best_model_3d.pth', map_location=device))
test_metrics = evaluate(model, test_loader, device)
print('[ Test 최종 결과 ]')
print(f'  Accuracy   : {test_metrics["accuracy"]:.4f}')
print(f'  AUC        : {test_metrics["auc"]:.4f}')
print(f'  Sensitivity: {test_metrics["sensitivity"]:.4f}')
print(f'  Specificity: {test_metrics["specificity"]:.4f}')
print(f'  F1         : {test_metrics["f1"]:.4f}')
print('=' * 50)

with open(f'{OUT_DIR}/test_results.json', 'w') as f:
    json.dump(test_metrics, f, indent=2)
print(f'결과 저장: {OUT_DIR}/test_results.json')