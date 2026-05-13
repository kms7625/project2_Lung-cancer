# LIDC-IDRI 폐 결절 악성/양성 분류

LIDC-IDRI CT 데이터셋을 이용한 폐 결절 이진 분류 프로젝트.  
**목표: Accuracy > 96.54%, AUC > 0.987** (C조 최고 성능 상회)

---

## 분류 전략

**전략 A — 2클래스 (score 1-3 = 양성(0), score 4-5 = 악성(1))**

| 전략 | 클래스 | 샘플 수 | 불균형 비율 |
|------|--------|--------|-----------|
| **A (선택)** | score 1-3 → 양성 / score 4-5 → 악성 | 7,849 | **1.21:1** |
| B | score 3 제외 | 5,332 | 1.99:1 |
| C | 3클래스 (양성/불확실/악성) | 7,849 | 1.99:1 |

선택 근거:
- 클래스 불균형 최소 (1.21:1)로 전 샘플 활용 가능
- 이전 팀 및 논문과 동일 기준 → 직접 성능 비교 가능
- score 3(불확실)을 양성으로 분류하는 보수적 접근이 임상적으로 타당

---

## 데이터셋

| 항목 | 값 |
|------|-----|
| 총 슬라이스 수 | 7,849개 (2D) / 494개 (3D ROI) |
| 환자 수 | 889명 (2D) / 494명 (3D) |
| 이미지 크기 | 300 × 300 (원본 슬라이스) |
| 파일명 규칙 | `slice_{z_idx}_{malignancy_score}.npy` |
| HU 범위 | -3024 ~ 1403 |

### Train/Val/Test 분할 (환자 단위)

| Split | 환자 수 | 슬라이스 수 | 양성 비율 | 악성 비율 |
|-------|--------|-----------|----------|----------|
| Train | 623명  | 5,530개   | 54.4%   | 45.6%   |
| Val   | 133명  | 1,111개   | 58.4%   | 41.6%   |
| Test  | 133명  | 1,208개   | 53.1%   | 46.9%   |

> 슬라이스 단위 분할 시 data leakage 발생 → **반드시 환자 단위로 분할**

---

## 코드 구조

```
project2_Lung-cancer/
├── project/
│   ├── prepare_csv.py       # slices/ 스캔 → 환자 단위 분할 → labels.csv 생성
│   ├── prepare_3d.py        # DICOM → 3D ROI 추출 → labels_3d.csv 생성
│   ├── model.py             # 3D ResNet18 모델 정의
│   ├── train_baseline.py    # 2D EfficientNetV2-S 학습 (베이스라인)
│   ├── train_3d.py          # 3D ResNet18 학습
│   ├── train_monai.py       # MONAI 기반 학습 (실험)
│   └── 1/
│       ├── eda_report.md       # EDA 결과 및 전략 A 선택 근거
│       ├── baseline_report.md  # 2D 베이스라인 결과 분석
│       ├── progress_report.md  # 3D 접근 진행 현황
│       └── final_report.md     # 최종 보고서
├── data_explore.ipynb       # EDA 노트북
├── study/
│   └── resize_ct.py         # 리샘플링, XML 파싱, segment 매칭 참고 코드
└── CLAUDE.md                # 프로젝트 가이드 (Claude Code용)
```

---

## 실행 방법

모든 명령은 **서버** (`kms@hkits`)에서 실행합니다.

```bash
# 서버 접속
ssh kms@192.168.3.19

# 코드 최신화
cd ~/project2_Lung-cancer && git pull origin main

# Step 1: 환자 단위 CSV 생성 (최초 1회)
cd ~/project2_Lung-cancer/project
python prepare_csv.py

# Step 2-A: 2D 베이스라인 학습
python train_baseline.py
python train_baseline.py --epochs 100 --batch 64 --arch convnext_tiny

# Step 2-B: 3D ROI 추출 (최초 1회)
python prepare_3d.py

# Step 3: 3D 모델 학습
python train_3d.py
```

### 서버 환경

| 항목 | 경로 |
|------|------|
| 데이터 루트 | `/home/kms/resnet_project/lidc-idri/` |
| 전처리 슬라이스 (npy) | `.../slices/LIDC-IDRI-XXXX/slice_ZZZ_S.npy` |
| 3D ROI (npy) | `.../rois_3d/` |
| 라벨 CSV | `.../labels.csv`, `.../labels_3d.csv` |
| 체크포인트 | `.../checkpoints/` |

---

## 현재 성능

### 2D 베이스라인 (EfficientNetV2-S, 전체 슬라이스 300×300)

| Epoch | Val Acc | Val AUC |
|-------|---------|---------|
| Best (17) | 63.5%   | 0.660  |

**문제**: Train Acc 91% vs Val Acc 63% — 심각한 과적합  
**원인**: 300×300 전체 슬라이스 입력 시 결절이 전체 면적의 1% 미만 차지 → 배경 정보를 학습

### 3D ResNet18 (ROI 큐브 64×64×64, 494개)

| 지표 | 값 |
|------|-----|
| Test Accuracy | 72.22% |
| **Test AUC** | **0.8909** |
| Sensitivity (악성 탐지율) | 0.9706 |
| Specificity (양성 탐지율) | 0.5000 |

**분석**: AUC 0.89는 유의미하나 Specificity가 낮음 (악성 쪽으로 편향)  
**원인**: 소규모 데이터(494개), 클래스 불균형(양성 199 vs 악성 295)

---

## 벤치마크 비교

| 팀/연구 | 모델 | 방식 | Accuracy | AUC |
|---------|------|------|----------|-----|
| **목표** | TBD | TBD | **>96.54%** | **>0.987** |
| C조 (최고) | EfficientNetV2-L + SAM | 2D ROI 크롭 | 96.54% | 0.987 |
| ProCAN (논문) | Progressive Channel Attention | 2D | 95.28% | 0.980 |
| A팀 | ResNet18 + CBAM + MGA | 2D 전체 슬라이스 | ~94% | 0.957 |
| MT-Swin (논문) | Multi-task Swin Transformer | 2D | 93.74% | 0.985 |
| 2팀 | ConvNeXt-Tiny + CBAM + CAM-Align | 2D | 91.50% | 0.950 |
| **현재 (3D)** | 3D ResNet18 | 3D ROI 큐브 | 72.22% | 0.891 |
| **현재 (2D)** | EfficientNetV2-S | 2D 전체 슬라이스 | ~63% | ~0.66 |

---

## 개선 계획

1. **ROI 크롭** — XML 폴리곤 좌표로 결절 중심 계산 → 64×64 또는 128×128 크롭 (배경 노이즈 제거)
2. **CBAM** — 채널 + 공간 어텐션으로 결절 위치에 집중
3. **SAM** — Segment Anything Model로 결절 영역 자동 분할 (C조 핵심 요소)
4. **3D 데이터 확충** — class weight + 증강 강화로 Specificity 개선

---

## 참고 문헌

- Armato et al., "The Lung Image Database Consortium (LIDC) and Image Database Resource Initiative (IDRI)", *Medical Physics*, 2011
- ProCAN: Progressive Channel Attention Network for Lung Nodule Classification
- MT-Swin: Multi-task Swin Transformer for Pulmonary Nodule Classification
