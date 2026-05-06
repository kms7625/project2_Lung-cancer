"""
환자 단위 train/val/test 분할 CSV 생성.

Usage:
    python prepare_csv.py [--slices_dir DIR] [--out_csv PATH] [--seed 42]

출력 CSV 컬럼:
    patient_id, image_path, label, split
    label: 0=양성(score 1-3), 1=악성(score 4-5)
    split: train / val / test
"""

import os
import glob
import argparse
import random
import csv
from collections import defaultdict

DATA_ROOT = '/home/kms/resnet_project/lidc-idri'


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--slices_dir', default=f'{DATA_ROOT}/slices')
    p.add_argument('--out_csv', default=f'{DATA_ROOT}/labels.csv')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--val_ratio', type=float, default=0.15)
    p.add_argument('--test_ratio', type=float, default=0.15)
    return p.parse_args()


def score_to_label(score: int) -> int:
    # 전략 A: score 1-3 = 양성(0), score 4-5 = 악성(1)
    return 0 if score <= 3 else 1


def main():
    args = parse_args()
    random.seed(args.seed)

    print(f"슬라이스 디렉토리: {args.slices_dir}")
    npy_files = glob.glob(os.path.join(args.slices_dir, '**', '*.npy'), recursive=True)
    print(f"총 NPY 파일: {len(npy_files)}개")

    # 환자별 슬라이스 목록 수집
    patient_slices = defaultdict(list)  # patient_id -> [(path, label)]
    skip_count = 0
    for path in npy_files:
        fname = os.path.basename(path)          # slice_048_4.npy
        patient_id = os.path.basename(os.path.dirname(path))  # LIDC-IDRI-XXXX
        try:
            score = int(fname.replace('.npy', '').split('_')[-1])
        except ValueError:
            skip_count += 1
            continue
        if score < 1 or score > 5:
            skip_count += 1
            continue
        label = score_to_label(score)
        patient_slices[patient_id].append((path, label))

    if skip_count:
        print(f"  파싱 실패 / 범위 외 파일 {skip_count}개 제외")

    patients = sorted(patient_slices.keys())
    random.shuffle(patients)
    n = len(patients)
    n_val = int(n * args.val_ratio)
    n_test = int(n * args.test_ratio)
    n_train = n - n_val - n_test

    val_patients = set(patients[:n_val])
    test_patients = set(patients[n_val:n_val + n_test])
    train_patients = set(patients[n_val + n_test:])

    print(f"\n환자 분할 (seed={args.seed})")
    print(f"  Train: {len(train_patients)}명")
    print(f"  Val  : {len(val_patients)}명")
    print(f"  Test : {len(test_patients)}명")

    rows = []
    for patient_id, slices in patient_slices.items():
        if patient_id in train_patients:
            split = 'train'
        elif patient_id in val_patients:
            split = 'val'
        else:
            split = 'test'
        for path, label in slices:
            rows.append({
                'patient_id': patient_id,
                'image_path': path,
                'label':      label,
                'split':      split,
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

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['patient_id', 'image_path', 'label', 'split'])
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
