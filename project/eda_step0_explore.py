import os
import glob

DATA_ROOT = '/home/kms/resnet_project/lidc-idri'

print("=" * 60)
print("Step 0: 데이터 경로 구조 탐색")
print("=" * 60)

# 1. 최상위 구조
print("\n[1] 최상위 항목")
try:
    for item in sorted(os.listdir(DATA_ROOT)):
        full = os.path.join(DATA_ROOT, item)
        kind = 'DIR ' if os.path.isdir(full) else 'FILE'
        size = ''
        if os.path.isfile(full):
            size = f'  ({os.path.getsize(full):,} bytes)'
        print(f"  [{kind}] {item}{size}")
except Exception as e:
    print(f"  ERROR: {e}")

# 2. 환자 폴더 패턴 탐색
print("\n[2] 환자 폴더 패턴 탐색")
for pattern in ['LIDC-IDRI-*', '*/LIDC-IDRI-*', 'manifest*', '*/manifest*']:
    found = glob.glob(f'{DATA_ROOT}/{pattern}')
    if found:
        print(f"  패턴 '{pattern}': {len(found)}개")
        print(f"    예시: {found[0]}")
        if len(found) > 1:
            print(f"    예시2: {found[1]}")

# 3. XML 위치 탐색
print("\n[3] XML 파일 위치")
xml_candidates = glob.glob(f'{DATA_ROOT}/**/*.xml', recursive=True)
print(f"  총 XML: {len(xml_candidates)}개")
if xml_candidates:
    dirs = sorted(set(os.path.dirname(x) for x in xml_candidates[:30]))
    for d in dirs[:8]:
        cnt = len([x for x in xml_candidates if x.startswith(d)])
        print(f"  {d}  ({cnt}개)")

# 4. DICOM 파일 탐색
print("\n[4] DICOM 파일 위치")
dcm_candidates = glob.glob(f'{DATA_ROOT}/**/*.dcm', recursive=True)
print(f"  총 DCM: {len(dcm_candidates)}개 (첫 5개만)")
for d in dcm_candidates[:5]:
    print(f"  {d}")

# 5. 핵심 파일 존재 여부
print("\n[5] 핵심 파일 체크")
key_files = [
    'metadata.csv',
    'tcia-diagnosis-data-2012-04-20.xls',
    'lidc-idri-nodule-counts-6-23-2015.xlsx',
    'nodule_malignancy_scores.json',
]
for f in key_files:
    path = os.path.join(DATA_ROOT, f)
    if os.path.exists(path):
        print(f"  [O] {f}")
    else:
        found = glob.glob(f'{DATA_ROOT}/**/{f}', recursive=True)
        if found:
            print(f"  [O] {f}  → {found[0]}")
        else:
            print(f"  [X] {f}")

# 6. 전처리 결과물 탐색 (이미 처리된 데이터 있는지 확인)
print("\n[6] 전처리 결과물 체크")
preprocess_patterns = [
    ('nii.gz', '**/*.nii.gz'),
    ('slices npy', '**/slice_*.npy'),
    ('slices png', '**/slice_*.png'),
]
for name, pattern in preprocess_patterns:
    found = glob.glob(f'{DATA_ROOT}/{pattern}', recursive=True)
    print(f"  {name}: {len(found)}개")
    if found:
        print(f"    예시: {found[0]}")

print("\n" + "=" * 60)
print("탐색 완료 — 위 결과를 Claude에게 공유해주세요")
print("=" * 60)
