# 폐 결절 판별 프로젝트 (LIDC-IDRI)

## 프로젝트 개요

- **목표**: LIDC-IDRI 데이터셋을 이용한 폐 결절 악성/양성 분류 모델 개발
- **데이터**: `/data1/wellness_data/project` (1018명 환자 CT, DICOM + XML)
- **일정**: 5/13 ~ 6/10 (4주), 매주 수·금 진행상황 공유
  - 1주차 (5/13~5/19): 발표 방향 구상, 작업 분배, 데이터 전처리, 베이스라인 확보
  - 2주차 (5/20~5/26): 실험 (5/25 대체공휴일)
  - 3주차 (5/27~6/2): 실험 + 보고서/발표 자료 준비
  - 4주차 (6/3~6/10): 추가 실험 + 보고서/발표 마무리 (6/10 지방선거)
- **참고 코드**: `study/resize_ct.py` — 리샘플링, XML 파싱, segment 매칭 구현

## 과제 요구사항 (study/start.md)

1. LIDC-IDRI로 분류 수행 (2D slice 또는 3D volume) — **직접 라벨링 불필요, XML 어노테이션 사용**
2. Malignancy Score 분류 기준과 근거 명시 — **몇 클래스로 나눌지도 결정해야 함 (2클래스 vs 3클래스)**
3. 성능을 이전 연구 또는 다른 모델들과 비교 + 이유 분석 (왜 해당 클래스 수로 했는지 포함)

## 스킬셋

| 스킬 | 용도 |
|------|------|
| `/lidc-eda` | 데이터 분포 확인, malignancy 분포, 클래스 불균형 분석 |
| `/lidc-preprocess` | DICOM 로딩, XML 파싱, HU 윈도잉, ROI 크롭, CSV 생성 |
| `/lidc-baseline` | 베이스라인 모델 구축, Dataset/DataLoader, 학습 루프 |
| `/lidc-experiment` | Ablation study, CBAM/SAM/CAM-Align 실험 설계 |
| `/lidc-analyze` | 성능 비교 표, 이전 연구 대비 분석 |
| `/lidc-report` | 보고서/발표 자료 작성, 과제 답변 가이드 |
| `/lidc-verify` | 코드 완성 후 검증 (data leakage, AUC, sensitivity) |

## 이전 팀 성능 (벤치마크)

| 팀/연구 | 모델 | Accuracy | AUC |
|---------|------|----------|-----|
| C조 (최고) | EfficientNetV2-L + SAM | 96.54% | 0.987 |
| A팀 | ResNet18 + CBAM + MGA | ~94% | 0.957 |
| 2팀 | ConvNeXt-Tiny + CBAM + CAM-Align | 91.50% | 0.950 |
| ProCAN (논문) | Progressive Channel Attention | 95.28% | 0.980 |
| MT-Swin (논문) | Multi-task Swin Transformer | 93.74% | 0.985 |

**목표: C조 성능(96.54% / AUC 0.987) 상회**

## 핵심 주의사항

- **직접 라벨링 불필요**: polygon + malignancy score는 XML에 이미 제공됨 (study/resize_ct.py로 추출)
- **클래스 수 결정 필요**: 2클래스(악성/양성) 또는 3클래스(악성/불확실/양성) — 근거와 함께 결정
- **반드시 환자(patient) 단위로 train/val/test 분할** — 슬라이스 단위 분할은 data leakage
- **Best model 저장 기준: AUC** — Accuracy는 클래스 불균형 시 착시 가능
- **XML 네임스페이스**: `{'ns': 'http://www.nih.gov'}` (study/resize_ct.py 참조)
- **리샘플링**: (1.0, 1.0, 3.0) mm spacing → DICOM을 numpy/SimpleITK로 변환 후 사용
