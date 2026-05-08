"""
환자 단위 train/val/test 분할 CSV 생성.

Usage:
    python prepare_csv.py [--slices_dir DIR] [--out_csv PATH] [--seed 42]
                          [--json_path PATH]

출력 CSV 컬럼:
    patient_id, image_path, label, split, cx, cy
    label: 0=양성(score 1-3), 1=악성(score 4-5)
    split: train / val / test
    cx, cy: 결절 centroid (pixel 좌표), -1이면 JSON 미매칭
"""

import os
import glob
import argparse
import random
import csv
import json
from collections import defaultdict

import numpy as np

DATA_ROOT = '/home/kms/resnet_project/lidc-idri'


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--slices_dir', default=f'{DATA_ROOT}/slices')
    p.add_argument('--out_csv',    default=f'{DATA_ROOT}/labels.csv')
    p.add_argument('--json_path',  default=f'{DATA_ROOT}/nodule_malignancy_scores.json')
    p.add_argument('--seed',       type=int,   default=42)
    p.add_argument('--val_ratio',  type=float, default=0.15) # 15%
    p.add_argument('--test_ratio', type=float, default=0.15) # 15%
    return p.parse_args()


def score_to_label(score: int) -> int:
    # 전략 A: score 1-2 = 양성(0), score 4-5 = 악성(1)
    return 0 if score <= 2 else 1


def load_centroids(json_path: str) -> dict:
    """
    JSON → {patient_id: {z_idx(int): (cx_float, cy_float)}}
    cx = 폴리곤 x 좌표 평균 (column), cy = y 좌표 평균 (row)
    같은 슬라이스의 모든 reader 폴리곤 점들의 평균을 centroid로 사용.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    centroids = {}
    for patient_id, slices in data.items():
        patient_centroids = {}
        for slice_key, readers in slices.items():
            # slice_key 예: "slice_5", "slice_48"
            try:
                z_idx = int(slice_key.split('_')[-1])
            except ValueError:
                continue
            all_points = []
            for reader_id, nodules in readers.items():
                for nodule in nodules:
                    all_points.extend(nodule.get('polygon', []))
            if all_points:
                arr = np.array(all_points, dtype=np.float32)
                cx = float(arr[:, 0].mean())  # x = column
                cy = float(arr[:, 1].mean())  # y = row
                patient_centroids[z_idx] = (cx, cy)
        centroids[patient_id] = patient_centroids
    return centroids


def main():
    args = parse_args()
    random.seed(args.seed)

    print(f"슬라이스 디렉토리: {args.slices_dir}")
    npy_files = glob.glob(os.path.join(args.slices_dir, '**', '*.npy'), recursive=True)
    print(f"총 NPY 파일: {len(npy_files)}개")

    # JSON centroid 로드
    centroids = {}
    if os.path.exists(args.json_path):
        centroids = load_centroids(args.json_path)
        print(f"JSON 로드 완료: {len(centroids)}명 환자 centroid")
    else:
        print(f"경고: JSON 없음 ({args.json_path}) → cx/cy = -1 로 설정")

    # 환자별 슬라이스 목록 수집
    patient_slices = defaultdict(list)  # patient_id -> [(path, label, z_idx)]
    skip_count = 0
    for path in npy_files:
        fname = os.path.basename(path)          # slice_048_4.npy
        patient_id = os.path.basename(os.path.dirname(path))  # LIDC-IDRI-XXXX
        parts = fname.replace('.npy', '').split('_')
        try:
            z_idx = int(parts[1])
            score = int(parts[-1])
        except (ValueError, IndexError):
            skip_count += 1
            continue
        if score < 1 or score > 5 or score == 3:
            skip_count += 1
            continue
        label = score_to_label(score)
        patient_slices[patient_id].append((path, label, z_idx))

    if skip_count:
        print(f"  파싱 실패 / 범위 외 파일 {skip_count}개 제외")

    patients = sorted(patient_slices.keys())
    random.shuffle(patients)
    n = len(patients)
    n_val = int(n * args.val_ratio)
    n_test = int(n * args.test_ratio)

    val_patients = set(patients[:n_val])
    test_patients = set(patients[n_val:n_val + n_test])
    train_patients = set(patients[n_val + n_test:])

    print(f"\n환자 분할 (seed={args.seed})")
    print(f"  Train: {len(train_patients)}명")
    print(f"  Val  : {len(val_patients)}명")
    print(f"  Test : {len(test_patients)}명")

    rows = []
    no_centroid_count = 0
    for patient_id, slices in patient_slices.items():
        if patient_id in train_patients:
            split = 'train'
        elif patient_id in val_patients:
            split = 'val'
        else:
            split = 'test'

        patient_cx = centroids.get(patient_id, {})
        for path, label, z_idx in slices:
            if z_idx in patient_cx:
                cx, cy = patient_cx[z_idx]
            else:
                cx, cy = -1.0, -1.0
                no_centroid_count += 1
            rows.append({
                'patient_id': patient_id,
                'image_path': path,
                'label':      label,
                'split':      split,
                'cx':         round(cx, 2),
                'cy':         round(cy, 2),
            })

    # 분포 출력
    from collections import Counter
    for sp in ['train', 'val', 'test']:
        sp_rows = [r for r in rows if r['split'] == sp]
        cnt = Counter(r['label'] for r in sp_rows)
        total = len(sp_rows)
        print(f"\n  [{sp}] {total}개 슬라이스 — "
              f"양성(0): {cnt[0]} ({cnt[0]/total*100:.1f}%), "
              f"악성(1): {cnt[1]} ({cnt[1]/total*100:.1f}%)")

    centroid_total = len(rows) - no_centroid_count
    print(f"\n  Centroid 매칭: {centroid_total}/{len(rows)} "
          f"({centroid_total/len(rows)*100:.1f}%)")

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    fieldnames = ['patient_id', 'image_path', 'label', 'split', 'cx', 'cy']
    with open(args.out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Data leakage 검증
    train_pids = {r['patient_id'] for r in rows if r['split'] == 'train'}
    val_pids   = {r['patient_id'] for r in rows if r['split'] == 'val'}
    test_pids  = {r['patient_id'] for r in rows if r['split'] == 'test'}
    assert not train_pids & val_pids,  "Data leakage: train-val 환자 중복!"
    assert not train_pids & test_pids, "Data leakage: train-test 환자 중복!"
    assert not val_pids & test_pids,   "Data leakage: val-test 환자 중복!"
    print("\nData leakage 검증 통과 ✓")

    print(f"\nCSV 저장 완료: {args.out_csv}")
    print(f"총 행 수: {len(rows)}")


if __name__ == '__main__':
    main()
