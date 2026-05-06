# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 프로젝트 개요

- **목표**: LIDC-IDRI 데이터셋으로 폐 결절 악성/양성 분류 모델 개발
- **분류 전략**: 전략 A — 2클래스 (score 1-3 = 양성(0), score 4-5 = 악성(1))
- **목표 성능**: Accuracy > 96.54%, AUC > 0.987 (C조 최고 성능 상회)
- **일정**: 5/13 ~ 6/10 (4주), 매주 수·금 진행상황 공유
- **GitHub**: https://github.com/kms7625/project2_Lung-cancer

## 서버 환경

| 항목 | 경로 |
|------|------|
| 서버 | `kms@hkits` |
| 데이터 루트 | `/home/kms/resnet_project/lidc-idri/` |
| 전처리 슬라이스 (npy) | `.../slices/LIDC-IDRI-XXXX/slice_ZZZ_S.npy` |
| 라벨 CSV | `.../labels.csv` |
| 체크포인트 저장 | `.../checkpoints/baseline/` |
| 프로젝트 코드 | `~/project2_Lung-cancer/project/` |

> `/data1/wellness_data/project`는 위 데이터 루트의 심볼릭 링크

## 실행 명령어

모든 명령은 **서버 터미널** (`kms@hkits`)에서 실행한다.

```bash
# 서버 접속 (로컬 터미널에서)
ssh kms@192.168.3.19

# 코드 최신화
cd ~/project2_Lung-cancer && git pull origin main

# Step 1: 환자 단위 CSV 생성 (최초 1회)
cd ~/project2_Lung-cancer/project
python prepare_csv.py

# Step 2: 베이스라인 학습
python train_baseline.py

# 옵션 지정 예시
python train_baseline.py --epochs 100 --batch 64 --arch convnext_tiny
```

## 데이터 현황 (완료)

- **슬라이스**: 7,849개 NPY 파일 (889명 환자)
- **파일명 규칙**: `slice_{z_idx}_{malignancy_score}.npy` — score가 파일명에 내장
- **분할**: Train 623명(5,530개) / Val 133명(1,111개) / Test 133명(1,208개)
- **클래스 비율**: 양성 54% / 악성 46% (전 split 균일, leakage 검증 완료)

## 코드 구조

```
project/
  prepare_csv.py     # slices/ 스캔 → patient 단위 분할 → labels.csv 생성
  train_baseline.py  # Dataset·DataLoader·모델·학습루프·평가 일체형 스크립트
  1/eda_report.md    # EDA 결과 (분포 분석, 전략 A 선택 근거)
```

### train_baseline.py 핵심 구조

- `LIDCDataset` — NPY 로드 → 3채널 복제 → transforms 적용
- `get_dataloaders` — WeightedRandomSampler로 클래스 불균형 대응
- `get_model(arch, num_classes)` — EfficientNetV2-S / ConvNeXt-Tiny / ResNet18 지원
- `evaluate` — 2클래스: `roc_auc_score(labels, probs[:,1])` / 3클래스: `multi_class='ovr'`
- `train` — SGD(lr=0.01, momentum=0.9) + CosineAnnealingLR, **Val AUC 기준** best model 저장

## 핵심 제약사항

- **환자 단위 분할 필수** — 슬라이스 단위 분할은 data leakage
- **Best model 기준: AUC** — Accuracy는 불균형 시 착시 가능
- **3클래스 AUC**: `roc_auc_score(labels, probs, multi_class='ovr', average='macro')` 사용
- **RandomErasing 위치**: Normalize 이후에 배치

## 이전 팀 벤치마크

| 팀/연구 | 모델 | Accuracy | AUC |
|---------|------|----------|-----|
| C조 (최고) | EfficientNetV2-L + SAM | 96.54% | 0.987 |
| ProCAN (논문) | Progressive Channel Attention | 95.28% | 0.980 |
| A팀 | ResNet18 + CBAM + MGA | ~94% | 0.957 |
| MT-Swin (논문) | Multi-task Swin Transformer | 93.74% | 0.985 |
| 2팀 | ConvNeXt-Tiny + CBAM + CAM-Align | 91.50% | 0.950 |

## 스킬셋

| 스킬 | 용도 |
|------|------|
| `/lidc-eda` | 데이터 분포 확인, malignancy 분포, 클래스 불균형 분석 |
| `/lidc-preprocess` | DICOM 로딩, XML 파싱, HU 윈도잉, ROI 크롭, CSV 생성 |
| `/lidc-baseline` | 베이스라인 모델 구축, Dataset/DataLoader, 학습 루프 |
| `/lidc-experiment` | Ablation study, CBAM/SAM/CAM-Align 실험 설계 |
| `/lidc-analyze` | 성능 비교 표, 이전 연구 대비 분석 |
| `/lidc-report` | 보고서/발표 자료 작성, 과제 답변 가이드 |
| `/lidc-verify` | 코드 완성 후 검증 → 통과 시 GitHub 자동 push |

## GitHub 워크플로우

코드 수정 → `/lidc-verify` 실행 → 검증 통과 → 자동 `git commit` + `git push origin main`

서버에서 최신 코드 반영:
```bash
cd ~/project2_Lung-cancer && git pull origin main
```
