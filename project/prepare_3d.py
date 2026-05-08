# [목적] 전체 515개 결절에 대해 아래 파이프라인을 자동으로 실행한다.
# XML 필터링 → DICOM 읽기 → 리샘플링(1x1x1mm) → ROI 추출(64x64x64) → npy 저장
# 저장 경로: /home/kms/resnet_project/lidc-idri/rois_3d/{subject_id}/nodule_{n}.npy
# 완료 후 labels_3d.csv 생성 (image_path, label, subject_id, split)

import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
import xml.etree.ElementTree as ET
from statistics import variance, mean
from scipy.spatial.distance import euclidean
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ── 경로 설정 ──────────────────────────────────────
BASE       = '/home/kms/resnet_project/lidc-idri'
XML_DIR    = f'{BASE}/LIDC-XML-only/tcia-lidc-xml'
META_CSV   = f'{BASE}/manifest-1600709154662/metadata.csv'
DICOM_BASE = f'{BASE}/manifest-1600709154662'
SAVE_DIR   = f'{BASE}/rois_3d'
OUT_CSV    = f'{BASE}/labels_3d.csv'
HALF       = 32   # ROI 크기: 64x64x64
NS         = {'ns': 'http://www.nih.gov'}

os.makedirs(SAVE_DIR, exist_ok=True)

# ── XML 파싱 함수 ───────────────────────────────────
def parse_xml_nodules(xml_path):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except:
        return [], None

    header = root.find('.//ns:ResponseHeader', NS)
    uid = header.find('ns:SeriesInstanceUid', NS).text if header is not None else None

    sessions = root.findall('.//ns:readingSession', NS)
    all_nodules = []
    for reader_idx, session in enumerate(sessions):
        for nodule in session.findall('ns:unblindedReadNodule', NS):
            char = nodule.find('ns:characteristics', NS)
            if char is None:
                continue
            mal = char.find('ns:malignancy', NS)
            if mal is None or mal.text is None:
                continue
            score = int(mal.text)

            rois = nodule.findall('ns:roi', NS)
            if not rois:
                continue

            z_positions, xy_points = [], []
            for roi in rois:
                z = roi.find('ns:imageZposition', NS)
                if z is not None:
                    z_positions.append(float(z.text))
                for edge in roi.findall('ns:edgeMap', NS):
                    x = float(edge.find('ns:xCoord', NS).text)
                    y = float(edge.find('ns:yCoord', NS).text)
                    xy_points.append([x, y])

            if xy_points:
                cx = np.mean([p[0] for p in xy_points])
                cy = np.mean([p[1] for p in xy_points])
                cz = np.mean(z_positions) if z_positions else 0
                all_nodules.append((reader_idx, score, cz, cx, cy))

    # 위치 기반 그룹핑
    groups = []
    used = [False] * len(all_nodules)
    for i, n1 in enumerate(all_nodules):
        if used[i]:
            continue
        group = [n1]
        used[i] = True
        for j, n2 in enumerate(all_nodules):
            if used[j] or i == j:
                continue
            if euclidean([n1[2], n1[3], n1[4]], [n2[2], n2[3], n2[4]]) < 15:
                group.append(n2)
                used[j] = True
        groups.append(group)

    # 필터링
    valid = []
    for group in groups:
        scores = [n[1] for n in group]
        n_r = len(group)
        avg = mean(scores)
        var = variance(scores) if n_r >= 2 else 0
        if n_r < 3 or var > 1 or 2.5 <= avg <= 3.5:
            continue
        valid.append({
            'uid': uid,
            'cx': float(np.mean([n[3] for n in group])),
            'cy': float(np.mean([n[4] for n in group])),
            'cz': float(np.mean([n[2] for n in group])),
            'label': 0 if avg <= 2 else 1
        })
    return valid, uid

# ── 리샘플링 함수 ───────────────────────────────────
def resample_to_1mm(image):
    orig_spacing = image.GetSpacing()
    orig_size = image.GetSize()
    new_size = [int(round(orig_size[i] * orig_spacing[i])) for i in range(3)]
    f = sitk.ResampleImageFilter()
    f.SetOutputSpacing([1.0, 1.0, 1.0])
    f.SetSize(new_size)
    f.SetOutputDirection(image.GetDirection())
    f.SetOutputOrigin(image.GetOrigin())
    f.SetTransform(sitk.Transform())
    f.SetDefaultPixelValue(-1000)
    f.SetInterpolator(sitk.sitkLinear)
    return f.Execute(image)

# ── ROI 추출 함수 ───────────────────────────────────
def extract_roi(orig_image, resampled, cx_pix, cy_pix, cz_mm, half=32):
    orig_spacing = orig_image.GetSpacing()
    orig_origin  = orig_image.GetOrigin()
    cx_mm = orig_origin[0] + cx_pix * orig_spacing[0]
    cy_mm = orig_origin[1] + cy_pix * orig_spacing[1]
    vx, vy, vz = resampled.TransformPhysicalPointToIndex((cx_mm, cy_mm, cz_mm))
    size = resampled.GetSize()
    if not (half <= vx < size[0]-half and half <= vy < size[1]-half and half <= vz < size[2]-half):
        return None  # 경계 근처 결절 제외
    arr = sitk.GetArrayFromImage(resampled)
    roi = arr[vz-half:vz+half, vy-half:vy+half, vx-half:vx+half]
    roi = np.clip(roi, -1000, 400)
    roi = (roi + 1000) / 1400
    return roi.astype(np.float32)

# ── 메인 파이프라인 ─────────────────────────────────
# 1. XML 전체 파싱
print('1단계: XML 파싱 중...')
xml_files = []
for r, d, files in os.walk(XML_DIR):
    for f in files:
        if f.endswith('.xml'):
            xml_files.append(os.path.join(r, f))

all_valid = []
for xml_path in xml_files:
    nodules, _ = parse_xml_nodules(xml_path)
    all_valid.extend(nodules)

df = pd.DataFrame(all_valid)
print(f'유효 결절: {len(df)}개')

# 2. metadata 매칭
meta = pd.read_csv(META_CSV)
uid_to_info = {
    row['Series UID']: {
        'subject_id': row['Subject ID'],
        'file_location': row['File Location'].replace('\\', '/')
    }
    for _, row in meta.iterrows()
}
df['subject_id'] = df['uid'].map(lambda x: uid_to_info.get(x, {}).get('subject_id'))
df['dicom_dir'] = df['uid'].map(
    lambda x: f"{DICOM_BASE}/{uid_to_info[x]['file_location']}" if x in uid_to_info else None
)
df = df.dropna(subset=['subject_id', 'dicom_dir'])

# 3. ROI 추출 및 저장
print('2단계: ROI 추출 중...')
rows = []
errors = 0

for i, row in df.iterrows():
    try:
        reader = sitk.ImageSeriesReader()
        series_IDs = reader.GetGDCMSeriesIDs(row['dicom_dir'])
        file_names = reader.GetGDCMSeriesFileNames(row['dicom_dir'], series_IDs[0])
        reader.SetFileNames(file_names)
        image = reader.Execute()
        resampled = resample_to_1mm(image)
        roi = extract_roi(image, resampled, row['cx'], row['cy'], row['cz'])

        if roi is None or roi.shape != (64, 64, 64):
            errors += 1
            continue

        save_subdir = f'{SAVE_DIR}/{row["subject_id"]}'
        os.makedirs(save_subdir, exist_ok=True)
        n = len([f for f in os.listdir(save_subdir) if f.endswith('.npy')])
        save_path = f'{save_subdir}/nodule_{n:03d}_{row["label"]}.npy'
        np.save(save_path, roi)

        rows.append({
            'image_path': save_path,
            'label': row['label'],
            'subject_id': row['subject_id']
        })

        if (len(rows) + errors) % 50 == 0:
            print(f'  진행: {len(rows)}개 저장, {errors}개 실패')

    except Exception as e:
        errors += 1
        continue

print(f'\n저장 완료: {len(rows)}개, 실패: {errors}개')

# 4. train/val/test 분할 (환자 단위)
result_df = pd.DataFrame(rows)
subjects = result_df['subject_id'].unique()
train_s, temp_s = train_test_split(subjects, test_size=0.3, random_state=42)
val_s, test_s   = train_test_split(temp_s,    test_size=0.5, random_state=42)

def get_split(sid):
    if sid in train_s: return 'train'
    if sid in val_s:   return 'val'
    return 'test'

result_df['split'] = result_df['subject_id'].map(get_split)
result_df.to_csv(OUT_CSV, index=False)

print(f'\nCSV 저장: {OUT_CSV}')
print(result_df['label'].value_counts())
print(result_df['split'].value_counts())