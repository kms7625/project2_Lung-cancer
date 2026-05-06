# LIDC-IDRI 폐 결절 분류 프로젝트 — 베이스라인 구축 보고서

**작성일:** 2026-05-06  
**단계:** 1주차 — 데이터 준비 및 베이스라인 모델 구축

---

## 1. 진행 요약

| 단계 | 상태 |
|------|------|
| 데이터 탐색 (EDA) | 완료 |
| 라벨링 전략 결정 | 완료 — 전략 A (2클래스) |
| 환자 단위 CSV 분할 | 완료 |
| 베이스라인 학습 코드 작성 | 완료 |
| 베이스라인 학습 실행 | 완료 (50 epoch) |
| 성능 분석 및 문제 진단 | 완료 |

---

## 2. 데이터 준비

### 2.1 metadata.csv ↔ XML UID 매칭

`resize_ct.py`에서 `metadata.csv`의 `Series UID`와 XML의 `SeriesInstanceUid`를 매칭하여 DICOM-XML 연결 및 환자 ID를 확정하였다.

```python
element = root.find('.//ns:SeriesInstanceUid', ns)
uid = element.text
subj_id = df.loc[df['Series UID'] == uid, 'Subject ID'].values[0]
```

이 과정을 통해 `slices/LIDC-IDRI-XXXX/` 폴더 구조로 전처리 결과를 저장하였다.

### 2.2 NPY 슬라이스 현황

| 항목 | 값 |
|------|-----|
| 총 슬라이스 수 | 7,849개 |
| 환자 수 | 889명 |
| 이미지 크기 | 300 × 300 |
| dtype | int32 (HU 값) |
| HU 범위 | -3024 ~ 1403 |
| 파일명 규칙 | `slice_{z_idx}_{malignancy_score}.npy` |

### 2.3 라벨링 전략

**전략 A — 2클래스 선택**

| 클래스 | Score 범위 | 의미 |
|--------|-----------|------|
| 0 (양성) | 1 ~ 3 | 양성 또는 불확실 |
| 1 (악성) | 4 ~ 5 | 악성 |

선택 근거:
- 클래스 불균형 비율 1.21:1 (세 전략 중 가장 균형적)
- 전체 7,849개 샘플 모두 활용
- 이전 팀 및 논문과 동일한 기준 → 직접 성능 비교 가능

### 2.4 환자 단위 Train/Val/Test 분할

슬라이스 단위 분할 시 data leakage가 발생하므로 **환자(patient) 단위**로 분할하였다.

| Split | 환자 수 | 슬라이스 수 | 양성 비율 | 악성 비율 |
|-------|--------|-----------|----------|----------|
| Train | 623명  | 5,530개   | 54.4%   | 45.6%   |
| Val   | 133명  | 1,111개   | 58.4%   | 41.6%   |
| Test  | 133명  | 1,208개   | 53.1%   | 46.9%   |

Data leakage 검증 (train/val/test 환자 ID 교집합): **통과 ✓**

---

## 3. 베이스라인 모델

### 3.1 모델 구성

| 항목 | 설정 |
|------|------|
| 아키텍처 | EfficientNetV2-S |
| 사전학습 | ImageNet |
| 파라미터 수 | 20,180,050 |
| 출력 클래스 | 2 (양성 / 악성) |
| Optimizer | SGD (lr=0.001, momentum=0.9, weight_decay=1e-4) |
| Scheduler | CosineAnnealingLR (T_max=50) |
| Loss | CrossEntropyLoss |
| Best model 기준 | Val AUC |

### 3.2 입력 전처리

```
NPY 로드 (HU int32)
  → HU 윈도잉: clip[-1000, 400] → [0, 1]
  → 3채널 복제 (H, W) → (H, W, 3)
  → ToTensor → Resize(224×224) → Normalize([0.5]*3, [0.5]*3)
```

Train 전용 augmentation: RandomHorizontalFlip, RandomVerticalFlip, RandomRotation(15°), RandomErasing(p=0.2)

### 3.3 클래스 불균형 대응

WeightedRandomSampler 적용 (불균형 비율 1.21:1)

---

## 4. 학습 결과

### 4.1 학습 곡선 (50 epoch)

| Epoch | Train Acc | Val Acc | Val AUC |
|-------|----------|---------|---------|
| 1     | 54.5%    | 57.6%   | 0.621  |
| 10    | 70.9%    | 59.8%   | 0.619  |
| 17    | 81.3%    | 63.5%   | **0.660** ← best |
| 30    | 89.0%    | 59.1%   | 0.600  |
| 50    | 91.2%    | 60.9%   | 0.615  |

### 4.2 문제점 진단

**현상**: Train Acc 91% vs Val Acc 63% — 심각한 과적합

**원인**: 전체 CT 단면(300×300) 입력 시 결절이 차지하는 면적이 전체의 1% 미만  
→ 모델이 결절 특성이 아닌 폐 전체 외관(배경 정보)을 학습

| 항목 | 값 |
|------|-----|
| 결절 크기 | 약 3~30mm (3~30 pixel at 1.0mm/px) |
| 이미지 크기 | 300 × 300 |
| 결절 면적 비율 | 약 0.01 ~ 1% |

---

## 5. 목표 성능 대비 현황

| 팀/연구 | 모델 | Accuracy | AUC |
|---------|------|----------|-----|
| **현재 베이스라인** | EfficientNetV2-S | ~63% | ~0.66 |
| MT-Swin (논문) | Multi-task Swin | 93.74% | 0.985 |
| ProCAN (논문) | Progressive Channel Attention | 95.28% | 0.980 |
| C조 (최고) | EfficientNetV2-L + SAM | 96.54% | 0.987 |
| **목표** | TBD | **>96.54%** | **>0.987** |

---

## 6. 다음 단계

성능 개선을 위한 계획:

1. **CBAM (Convolutional Block Attention Module) 추가**
   - 채널 + 공간 어텐션으로 결절 위치에 집중
   - EfficientNetV2-S 블록 사이에 삽입
   - 예상 효과: AUC 0.85 이상

2. **SAM (Segment Anything Model) 활용**
   - 결절 영역 자동 분할 → 집중 학습
   - C조 최고 성능(96.54%) 달성 핵심 요소

3. **ROI 크롭 전처리**
   - XML 폴리곤 좌표로 결절 중심 계산
   - 결절 주변 64×64 또는 128×128 크롭
   - 전처리 단계에서 사전 생성

```
현재: 전체 슬라이스(300×300) → 모델
목표: 결절 ROI 집중 → CBAM/SAM → 모델
```
