# LIDC-IDRI 폐 결절 분류 프로젝트 — 1주차 진행 보고서

**작성일:** 2026-05-06  
**단계:** 1주차 — 데이터 준비 · 베이스라인 · CBAM · ROI 크롭

---

## 1. 전체 진행 요약

| 단계 | 내용 | 상태 |
|------|------|------|
| 데이터 탐색 (EDA) | 분포 분석, 라벨링 전략 결정 | 완료 |
| 전처리 | HU 윈도잉, 환자 단위 CSV 분할 | 완료 |
| 베이스라인 학습 | EfficientNetV2-S, 50 epoch | 완료 |
| CBAM 추가 | features[6], features[7] 삽입 | 완료 |
| ROI 크롭 (center) | 128×128 이미지 중심 크롭 | 완료 |
| ROI 크롭 (nodule) | nodule_malignancy_scores.json centroid 적용 | **진행 중** |
| 다음 단계 | Val AUC ≥ 0.93 달성 후 /lidc-experiment | 대기 |

---

## 2. 데이터 현황

### 2.1 슬라이스 데이터

| 항목 | 값 |
|------|-----|
| 총 슬라이스 | 7,849개 NPY |
| 환자 수 | 889명 |
| 이미지 크기 | 300 × 300 px |
| dtype | int32 (HU 값) |
| HU 범위 | -3024 ~ 1403 |
| 파일명 규칙 | `slice_{z_idx}_{malignancy_score}.npy` |

### 2.2 라벨링 전략 (전략 A — 2클래스)

| 클래스 | Score 범위 | 의미 |
|--------|-----------|------|
| 0 (양성) | 1 ~ 3 | 양성 또는 불확실 |
| 1 (악성) | 4 ~ 5 | 악성 |

### 2.3 환자 단위 분할

| Split | 환자 수 | 슬라이스 수 | 양성 비율 | 악성 비율 |
|-------|--------|-----------|----------|----------|
| Train | 623명  | 5,530개   | 54.4%   | 45.6%   |
| Val   | 133명  | 1,111개   | 58.4%   | 41.6%   |
| Test  | 133명  | 1,208개   | 53.1%   | 46.9%   |

Data leakage 검증 (3-way assert): **통과 ✓**

---

## 3. 실험 결과 비교

### 3.1 성능 추이

| 실험 | 설정 | Val AUC (best) | Test AUC | Test Acc |
|------|------|:--------------:|:--------:|:--------:|
| Exp 1 — 베이스라인 | EfficientNetV2-S, 전체 슬라이스 300×300, 50 epoch | 0.660 | 0.660 | 63% |
| Exp 2 — CBAM 추가 | + CBAM (features[6,7]), 50 epoch | 0.661 | 0.652 | 61% |
| Exp 3 — ROI 크롭 (center) | + CBAM + 128×128 center crop, 50 epoch | **0.772** | **0.790** | **74%** |

> Exp 3은 `labels.csv`에 cx/cy가 없어 이미지 중심 크롭으로 동작함 (nodule centroid 미적용)

### 3.2 목표 대비 현황

| 팀/연구 | 모델 | Accuracy | AUC |
|---------|------|:--------:|:---:|
| **현재 (Exp 3)** | EfficientNetV2-S + CBAM + center crop | 74.17% | 0.790 |
| MT-Swin (논문) | Multi-task Swin | 93.74% | 0.985 |
| ProCAN (논문) | Progressive Channel Attention | 95.28% | 0.980 |
| C조 (최고) | EfficientNetV2-L + SAM | 96.54% | 0.987 |
| **목표** | — | **>96.54%** | **>0.987** |

---

## 4. 실험별 분석

### 4.1 Exp 1 — 베이스라인 (전체 슬라이스)

**핵심 문제:** Train Acc 91% vs Val Acc 63% — 심각한 과적합

**원인 분석:**
- 전체 CT 단면 300×300 입력 시 결절 면적 비율 **< 1%**
- 결절 크기: 3~30mm → 3~30px (at 1.0mm/px)
- 모델이 결절 특성이 아닌 폐 전체 배경 패턴을 학습

```
이미지 중 결절 면적:
  결절 지름 30px → 면적 ≈ 707px²
  전체 이미지   → 면적 = 90,000px²
  비율          → 약 0.8%
```

### 4.2 Exp 2 — CBAM 추가

**결과:** 성능 변화 없음 (Test AUC 0.652, 베이스라인과 동등)

**원인 분석:**
- CBAM은 어텐션 가중치를 학습하지만, 결절이 feature map에서 극히 작은 영역을 차지
- 어텐션 메커니즘이 효과를 발휘하려면 먼저 ROI 국소화가 선행되어야 함
- CBAM 단독으로는 결절 위치 학습 불가

### 4.3 Exp 3 — ROI 크롭 128×128 (center crop)

**결과:** Test AUC 0.790 (+0.138), Accuracy 74.17% (+13.1%)

**효과 분석:**
- 300×300 → 128×128 크롭 → 224×224 리사이즈
- 유효 해상도 1.75× 향상 (0.75px/mm → 1.75px/mm)
- 불필요한 배경(폐 벽, 흉곽) 제거로 노이즈 감소
- center crop이었음에도 개선: 크롭 자체의 효과 확인

**한계:**
- 실제 nodule centroid가 아닌 이미지 중심(150, 150) 사용
- 결절은 대부분 폐 실질(periphery)에 위치 → center crop이 결절을 빗나갈 수 있음
- nodule_malignancy_scores.json 기반 centroid 적용 시 추가 개선 예상

---

## 5. 현재 코드 구조

```
project/
  prepare_csv.py        # slices/ 스캔 + JSON centroid 추출 → labels.csv 생성
                        # 출력 컬럼: patient_id, image_path, label, split, cx, cy
  train_baseline.py     # 전체 파이프라인 (Dataset·DataLoader·모델·학습·평가)
  1/
    eda_report.md       # EDA 결과
    baseline_report.md  # 1차 베이스라인 보고서
    progress_report.md  # 이 파일 (1주차 종합)
```

### 주요 파라미터

```bash
# 현재 best 설정
python train_baseline.py \
  --arch efficientnet_v2_s \
  --cbam \
  --crop_size 128 \
  --epochs 50 \
  --lr 0.001 \
  --batch 32
```

### 입력 파이프라인

```
NPY (HU int32)
  → clip[-1000, 400] → (x + 1000) / 1400  # [0, 1] 정규화
  → ROI 크롭 128×128 (cx, cy 기준, fallback: 이미지 중심)
  → 3채널 복제 (H, W, 3)
  → Resize(224×224) → Normalize([0.5]*3, [0.5]*3)
  → RandomErasing(p=0.2)  [train only]
```

---

## 6. 다음 단계

### 즉시 실행 (Val AUC 0.93 목표)

1. **nodule centroid 크롭 적용**
   ```bash
   python prepare_csv.py --json_path /home/kms/resnet_project/lidc-idri/nodule_malignancy_scores.json
   python train_baseline.py --cbam --crop_size 128 --epochs 50
   ```
   - 예상 효과: AUC +0.03~0.05

2. **에폭 증가 + crop_size 조정**
   ```bash
   python train_baseline.py --cbam --crop_size 96 --epochs 100
   ```
   - 64px (결절 집중), 96px, 128px 비교 실험

3. **옵티마이저 변경 (AdamW)**
   - SGD → AdamW(lr=1e-4, weight_decay=1e-2)
   - 예상 효과: 수렴 속도 향상

### Val AUC ≥ 0.93 달성 후 (/lidc-experiment)

| 기법 | 기대 효과 |
|------|----------|
| SAM (Segment Anything) | 결절 영역 정밀 마스킹 → C조 96.54% 핵심 요소 |
| CAM-Align | Class Activation Map 기반 크롭 정렬 |
| Multi-scale ensemble | 64/128/192px 크롭 앙상블 |
| Test Time Augmentation | flip/rotate 예측 평균 |

---

## 7. 단계별 성능 로드맵

```
현재: AUC 0.790 (74%)
  ↓ nodule centroid 크롭
예상: AUC ~0.83
  ↓ 하이퍼파라미터 튜닝
예상: AUC ~0.90
  ↓ /lidc-experiment (SAM, CAM-Align)
목표: AUC > 0.987 (96.54%)
```
