## 🫁 LIDC-IDRI 폐 결절 악성/양성 분류

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![MONAI](https://img.shields.io/badge/MONAI-1.0+-green)

LIDC-IDRI CT 데이터셋을 이용한 폐 결절 이진 분류 프로젝트.
3D ResNet-18 기반 모델로 AUC 0.899 달성.

---

## 분류 전략

**Score 1-2 = 양성(0), Score 3 = 제외, Score 4-5 = 악성(1)**

| Score | 의미 | 분류 |
|-------|------|------|
| 1 | Highly Unlikely | 양성 (0) |
| 2 | Moderately Unlikely | 양성 (0) |
| 3 | Indeterminate | **제외** |
| 4 | Moderately Suspicious | 악성 (1) |
| 5 | Highly Suspicious | 악성 (1) |

**Score 3 제외 근거:**
- 방사선 전문의도 양성/악성 판단이 어려운 결절 → 학습 신호에 노이즈 증가
- 정답 라벨 신뢰도가 낮아 모델이 애매한 패턴을 학습하게 됨
- 선행 연구(Shen et al., 2017)에서도 동일한 이유로 Score 3 제외가 일반적

---

## 데이터셋

### 원본 데이터 (LIDC-IDRI)

| 항목 | 값 |
|------|-----|
| 환자 수 | 1,010명 |
| CT 스캔 수 | 1,308개 |
| DICOM 파일 수 | 244,527개 |
| 어노테이션 방식 | 방사선 전문의 4명 독립 평가 → XML |

### 전처리 결과

| 파이프라인 | 데이터 수 | 비고 |
|-----------|----------|------|
| 2D 슬라이스 (score 3 제외) | ~5,332개 | 889명 환자 |
| **3D ROI 큐브 (통계적 필터링 추가)** | **494개** | **64×64×64 voxel** |

### 3D 통계적 필터링 (본 연구 차별화 포인트)

신뢰도 낮은 어노테이션을 3가지 기준으로 추가 제거:

```
조건 1: 채점자 수 < 3명 → 제외 (해당 결절을 인식한 전문의가 적음)
조건 2: 점수 표본분산 > 1 → 제외 (전문의 간 의견 불일치가 큰 경우)
조건 3: 점수 평균이 2.5~3.5 → 제외 (경계에 걸리는 결절 추가 제거)
```

| 단계 | 결절 수 |
|------|---------|
| 전체 XML 파싱 | 1,318개 |
| 필터링 통과 | 515개 |
| 최종 저장 ROI (경계 초과 제외) | 494개 |

### 3D Train/Val/Test 분할 (환자 단위)

| 분할 | 샘플 수 | 양성 | 악성 |
|------|--------|-----|-----|
| Train | 352 | - | - |
| Val | 70 | - | - |
| Test | 72 | - | - |
| **전체** | **494** | **199** | **295** |

> 슬라이스 단위 분할 시 data leakage 발생 → **반드시 환자 단위로 분할**

---

## 코드 구조.

```
project2_Lung-cancer/
├── project/
│   ├── prepare_csv.py       # slices/ 스캔 → score 3 제외 → 환자 단위 분할 → labels.csv
│   ├── prepare_3d.py        # DICOM → 통계적 필터링 → 3D ROI 추출 → labels_3d.csv
│   ├── model.py             # 3D ResNet18 모델 정의
│   ├── train_baseline.py    # 2D EfficientNetV2-S 학습 (베이스라인)
│   ├── train_3d.py          # 3D ResNet18 학습 + Ablation
│   ├── train_monai.py       # MONAI 기반 학습 (실험)
│   └── 1/
│       ├── eda_report.md       # EDA 결과 및 분류 전략 결정 근거
│       ├── baseline_report.md  # 2D 베이스라인 결과 분석
│       ├── progress_report.md  # 3D 접근 진행 현황
│       └── final_report.md     # 최종 보고서
├── data_explore.ipynb       # EDA 노트북
├── study/
│   └── resize_ct.py         # 리샘플링, XML 파싱, segment 매칭 참고 코드
└── CLAUDE.md                # 프로젝트 가이드 (Claude Code용)
```

---

## 실행 방법.

모든 명령은 **서버** (`kms@hkits`)에서 실행합니다.

```bash
# 서버 접속
ssh 사용자@서버

# 코드 최신화
cd ~/project2_Lung-cancer && git pull origin main

# [2D] Step 1: labels.csv 생성 (score 3 자동 제외)
cd ~/project2_Lung-cancer/project
python prepare_csv.py

# [2D] Step 2: 베이스라인 학습
python train_baseline.py
python train_baseline.py --epochs 100 --batch 64 --arch convnext_tiny

# [3D] Step 1: ROI 추출 + labels_3d.csv 생성 (최초 1회)
python prepare_3d.py

# [3D] Step 2: 3D 모델 학습
python train_3d.py
```

### 서버 데이터 경로

| 항목 | 경로 |
|------|------|
| 데이터 루트 | `/home/kms/resnet_project/lidc-idri/` |
| 전처리 슬라이스 | `.../slices/LIDC-IDRI-XXXX/slice_ZZZ_S.npy` |
| 3D ROI 큐브 | `.../rois_3d/` |
| 라벨 CSV | `.../labels.csv`, `.../labels_3d.csv` |
| 체크포인트 | `.../checkpoints/` |

---

## 전처리 파이프라인

### 2D 파이프라인
```
NPY 슬라이스 (score 3 제외)
  → HU 클리핑 [-1000, 400] → [0, 1] 정규화
  → 3채널 복제 (H, W) → (H, W, 3)
  → Resize(224×224) → Normalize([0.5]*3, [0.5]*3)
  → Train 전용: RandomFlip, RandomRotation(15°), RandomErasing(p=0.2)
```

### 3D 파이프라인
```
원본 DICOM
  → SimpleITK 읽기 + 1×1×1mm 리샘플링
  → 통계적 필터링 (채점자 수 / 분산 / 평균)
  → 결절 중심 ±32 voxel → 64×64×64 ROI 추출
  → HU 클리핑 [-1000, 400] → (roi + 1000) / 1400
  → Train: 3축 랜덤 Flip + 90도 회전
```

---

## 모델 구조

### 3D ResNet-18

2D ResNet-18의 모든 합성곱 연산을 3D로 확장. A팀(2D ResNet-18 + CBAM)과 직접 비교 가능하도록 동일 구조 채택.

```
입력: (B, 1, 64, 64, 64)
  ↓ Conv3D 7×7×7, 64ch, stride=2
  ↓ BatchNorm3D + ReLU + MaxPool3D
  ↓ Layer1-4: BasicBlock3D × 2 (64→128→256→512ch)
  ↓ AdaptiveAvgPool3D(1) → Flatten
  ↓ FC(512 → 2)
출력: (B, 2)  — 양성 / 악성
```

| 항목 | 2D ResNet-18 | 3D ResNet-18 (본 연구) |
|------|-------------|----------------------|
| 합성곱 | Conv2D | Conv3D |
| 파라미터 수 | 11,689,512 | 33,161,026 |
| 입력 | (B, 3, 224, 224) | (B, 1, 64, 64, 64) |
| 사전학습 | ImageNet 가능 | 해당 없음 |

---

## 실험 결과

### 2D 베이스라인 (EfficientNetV2-S, 전체 슬라이스 224×224)

| Epoch | Val Acc | Val AUC |
|-------|---------|---------|
| Best (17) | 63.5% | 0.660 |

**문제**: Train Acc 91% vs Val Acc 63% — 심각한 과적합  
**원인**: 전체 슬라이스 입력 시 결절이 차지하는 면적 < 1% → 배경 정보 학습

### 3D ResNet-18 Ablation Study

| 실험 | 주요 변경 | Accuracy | AUC | Sensitivity | Specificity |
|------|-----------|----------|-----|-------------|-------------|
| v1 (기본) | CrossEntropyLoss | 72.22% | 0.8909 | **97.06%** | 50.00% |
| **v2 (최종)** | **+ class weight** | **75.00%** | **0.8994** | 85.29% | **65.79%** |

v2 개선 포인트: Specificity +15.79%, AUC +0.85% (양성/악성 예측 균형 개선)

---

## 벤치마크 비교

| 팀 / 연구 | 모델 | 방식 | Accuracy | AUC |
|----------|------|------|----------|-----|
| **목표** | TBD | TBD | **>96.54%** | **>0.987** |
| C조 (최고) | EfficientNetV2-L + SAM | 2D ROI 크롭 | 96.54% | 0.987 |
| ProCAN (2022) | Progressive Channel Attention | 2D | 95.28% | 0.980 |
| A팀 | ResNet18 + CBAM + MGA | 2D 전체 슬라이스 | ~94% | 0.957 |
| MT-Swin (2025) | Multi-task Swin Transformer | 2D | 93.74% | 0.985 |
| 2팀 | ConvNeXt-Tiny + CBAM + CAM-Align | 2D | 91.50% | 0.950 |
| Xu et al. (2020) | 3D CNN | 3D | 90.90% | - |
| **본 연구 v2** | **3D ResNet-18 + class weight** | **3D ROI 큐브** | **75.00%** | **0.8994** |

### 성능 격차 분석

| 원인 | 상세 |
|------|------|
| 데이터 수 부족 | 3D 필터링 후 494개 vs 2D 수천 개 |
| 사전학습 가중치 없음 | 3D ResNet은 처음부터 학습 |
| 과적합 | Train Acc 97% vs Val Acc 91% |

---

## 향후 개선 방향

1. **데이터 증강 강화** — Elastic deformation, Gaussian noise 추가
2. **Epoch 확장** — 50 → 100 이상
3. **경량 3D 모델** — 데이터가 적으므로 3D MobileNet 등 시도
4. **Grad-CAM 3D** — 모델이 집중하는 부위 시각화

---

## 참고 문헌

- Armato et al. (2011). The Lung Image Database Consortium (LIDC) and Image Database Resource Initiative (IDRI). *Medical Physics*, 38(2), 915-931.
- Shen et al. (2017). Multi-crop CNN for lung nodule malignancy suspiciousness classification. *Pattern Recognition*, 61, 663-673.
- Al-Shabi et al. (2022). ProCAN: Progressive growing channel attentive non-local network. *Pattern Recognition*, 122, 108309.
- Jin et al. (2025). Multitask Swin Transformer for classification and characterization of pulmonary nodules. *Quantitative Imaging in Medicine and Surgery*, 15(3), 1845.
- He et al. (2016). Deep Residual Learning for Image Recognition. *CVPR*.
