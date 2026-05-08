# 폐 결절 악성/양성 분류 프로젝트 진행 현황

## 프로젝트 개요

| 항목 | 내용 |
|------|------|
| 목표 | LIDC-IDRI CT 데이터셋 기반 폐 결절 악성/양성 3D 이진 분류 |
| 데이터 | LIDC-IDRI (1010명 환자, 원본 DICOM) |
| 분류 기준 | score 1~2 = 양성(0), score 3 = 제외, score 4~5 = 악성(1) |
| 차별화 | 3D ROI 기반 + 통계적 필터링으로 신뢰도 높은 데이터만 학습 |
| 서버 | hkitserver.iptime.org:6303 (RTX GPU, Ubuntu 20.04) |
| 경로 | `/home/kms/resnet_project/lidc-idri/` |

---

## 타 팀 성능 비교 (참고용)

| 팀 | 모델 | 방식 | Accuracy | AUC |
|----|------|------|----------|-----|
| A팀 | ResNet18 + CBAM + MGA | 2D 전체 슬라이스 | 88.91% | 0.9400 |
| 2팀 | ConvNeXt + CAM-Alignment | 2D Axial 슬라이스 | - | - |
| C조 | EfficientNetV2(L) + SAM | 2D ROI 크롭 패치 | 96.54% | 0.987 |
| **우리 팀** | **3D ResNet18** | **3D ROI 큐브(64x64x64)** | **72.22%** | **0.8909** |

> C조가 가장 높은 성능. 우리 팀은 3D 방식으로 완전히 다른 접근.

---

## 데이터셋 구조

```
/home/kms/resnet_project/lidc-idri/
├── manifest-1600709154662/     # 원본 DICOM (환자 1010명)
│   └── LIDC-IDRI/
│       ├── LIDC-IDRI-0001/
│       │   └── .../1-001.dcm ~ 1-240.dcm
│       └── ...
├── LIDC-XML-only/              # 어노테이션 XML (1318개)
│   └── tcia-lidc-xml/
├── slices/                     # 전처리된 2D npy (889명, 5332개)
├── slices_png/                 # 시각화용 PNG
├── rois_3d/                    # 3D ROI npy (494개) ← 우리가 생성
├── nodule_malignancy_scores.json
├── labels.csv                  # 2D 학습용 CSV
├── labels_3d.csv               # 3D 학습용 CSV ← 우리가 생성
└── checkpoints/
    └── 3d_resnet/
        ├── best_model_3d.pth
        ├── history.json
        └── test_results.json
```

---

## 1단계: EDA (data_explore.ipynb)

### 확인 내용

| 항목 | 결과 |
|------|------|
| slices/ 환자 수 | 889명 |
| npy 파일 형식 | slice_{번호}_{악성도}.npy |
| 이미지 shape | (300, 300) |
| 픽셀값 범위 | -3024 ~ 1403 (HU 단위, 정규화 필요) |
| 원본 DICOM shape | (512, 512, 133~240) |
| Spacing | 환자마다 다름 → 리샘플링 필요 |
| XML 파일 수 | 1318개 |

### 핵심 발견
- XML의 noduleID는 채점자마다 달라서 ID 기반 매칭 불가
- 좌표(x, y, z) 기반으로 같은 결절 그룹핑 필요
- HU 클리핑(-1000 ~ 400) + 정규화(0~1) 필요

---

## 2단계: 필터링 전략 (팀 독자적 차별화 포인트)

### 필터링 조건 3가지

```python
# 조건 1: 채점자 3명 미만 제외
if n_readers < 3: exclude

# 조건 2: 점수 표본분산 > 1 제외
if variance(scores) > 1: exclude

# 조건 3: 점수 평균 2.5~3.5 제외 (경계 애매한 결절)
if 2.5 <= mean(scores) <= 3.5: exclude
```

### 필터링 결과

| 단계 | 수량 |
|------|------|
| 전체 XML | 1318개 |
| 추출된 유효 결절 | 515개 |
| metadata uid 매칭 | 515/515 (100%) |
| 최종 저장된 ROI | 494개 (21개 경계 외 제외) |

### 최종 데이터 분포

| 분할 | 양성(0) | 악성(1) | 합계 |
|------|---------|---------|------|
| train | - | - | 352 |
| val | - | - | 70 |
| test | - | - | 72 |
| **전체** | **199** | **295** | **494** |

---

## 3단계: 전처리 파이프라인 (prepare_3d.py)

```
원본 DICOM (.dcm 수백 장)
      ↓  SimpleITK로 읽기
3D 볼륨 합치기 (512×512×240)
      ↓  resample_to_1mm()
1×1×1mm 리샘플링 (300×300×300)
      ↓  extract_roi_v2()
결절 중심 기준 ±32 voxel 큐브 추출 (64×64×64)
      ↓  HU 클리핑(-1000~400) + 정규화(0~1)
npy 파일 저장
      ↓
labels_3d.csv 생성 (환자 단위 train/val/test 분할)
```

### 좌표 변환 방법
```python
# XML의 cx, cy는 픽셀 좌표 → world 좌표(mm)로 변환
cx_mm = orig_origin[0] + cx_pix * orig_spacing[0]
cy_mm = orig_origin[1] + cy_pix * orig_spacing[1]
# cz는 이미 mm 단위
voxel = resampled_image.TransformPhysicalPointToIndex((cx_mm, cy_mm, cz_mm))
```

---

## 4단계: 3D ResNet 모델 (model.py)

### 구조

```
입력: (B, 1, 64, 64, 64)
  ↓ Conv3D 7×7×7, stride=2
  ↓ MaxPool3D
  ↓ Layer1: BasicBlock3D × 2 (64ch)
  ↓ Layer2: BasicBlock3D × 2 (128ch, stride=2)
  ↓ Layer3: BasicBlock3D × 2 (256ch, stride=2)
  ↓ Layer4: BasicBlock3D × 2 (512ch, stride=2)
  ↓ AdaptiveAvgPool3D
  ↓ FC(512 → 2)
출력: (B, 2)  ← 양성/악성
```

| 항목 | 값 |
|------|-----|
| 파라미터 수 | 33,161,026 |
| 입력 채널 | 1 (grayscale CT) |
| 출력 클래스 | 2 (양성/악성) |
| A팀과 차이 | Conv2D → Conv3D, 2D→3D 전체 구조 |

---

## 5단계: 학습 결과 (train_3d.py)

### 학습 설정

| 항목 | 값 |
|------|-----|
| Optimizer | SGD (momentum=0.9, weight_decay=1e-4) |
| Scheduler | CosineAnnealingLR |
| Loss | CrossEntropyLoss |
| Epochs | 50 |
| Batch size | 8 |
| LR | 0.01 |
| 데이터 증강 | 3축 랜덤 Flip, 90도 회전 |
| 클래스 불균형 처리 | WeightedRandomSampler |

### Epoch별 주요 결과

| Epoch | Val Acc | Val AUC |
|-------|---------|---------|
| 1 | 0.3571 | 0.7493 |
| 10 | 0.8286 | 0.9156 |
| 20 | 0.8571 | 0.9271 |
| 36 | 0.9143 | 0.9520 ← best |
| 50 | 0.8714 | 0.9404 |

### 최종 Test 결과 (베이스라인 v1)

| 지표 | 값 |
|------|-----|
| Accuracy | 0.7222 |
| **AUC** | **0.8909** |
| Sensitivity (악성 탐지율) | 0.9706 |
| Specificity (양성 탐지율) | 0.5000 |
| F1-score | 0.7674 |

### 분석
- AUC 0.89는 괜찮은 수준
- Sensitivity 0.97 → 악성 결절을 거의 다 잡아냄
- Specificity 0.50 → 양성을 절반밖에 못 맞힘 → 모델이 악성 쪽으로 치우침
- 원인: 데이터 부족(494개), 클래스 불균형(양성199 vs 악성295)

---

## 다음 단계 (개선 계획)

### 즉시 적용 가능
- [ ] CrossEntropyLoss에 class weight 적용 → Specificity 개선
- [ ] Epoch 늘리기 (50 → 100)
- [ ] 데이터 증강 강화

### 코드 수정 내용
```python
# train_3d.py에서 수정
class_counts = np.bincount(df[df['split']=='train']['label'].values)
class_weights = torch.tensor(1.0 / class_counts, dtype=torch.float).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
```

---

## 파일 목록

| 파일 | 역할 |
|------|------|
| `data_explore.ipynb` | EDA, 데이터 구조 파악, 파이프라인 검증 |
| `project/prepare_csv.py` | 2D labels.csv 생성 |
| `project/prepare_3d.py` | 3D ROI 추출 + labels_3d.csv 생성 |
| `project/model.py` | 3D ResNet 모델 정의 |
| `project/train_3d.py` | 3D 모델 학습 + 평가 |

---

## GitHub 커밋 히스토리

| 커밋 | 내용 |
|------|------|
| `bb5667e` | EDA 및 데이터 파이프라인 검증 완료 |
| `a13b1ef` | prepare_3d.py: ROI 추출 + labels_3d.csv 생성 |
| 미커밋 | model.py, train_3d.py |

---

*작성일: 2026-05-08*
