import os
import glob
import json
from collections import Counter, defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_ROOT = '/home/kms/resnet_project/lidc-idri'
SLICES_DIR = f'{DATA_ROOT}/slices'
JSON_PATH = f'{DATA_ROOT}/nodule_malignancy_scores.json'
OUT_DIR = '/tmp/eda_outputs'
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 60)
print("Step 1: Malignancy 분포 분석")
print("=" * 60)

# --- 슬라이스 파일에서 환자ID, score 추출 ---
# 파일명 패턴: slices/LIDC-IDRI-XXXX/slice_ZZZ_S.npy
npy_files = glob.glob(f'{SLICES_DIR}/**/*.npy', recursive=True)
print(f"\n총 슬라이스 수: {len(npy_files)}")

records = []
for path in npy_files:
    parts = path.split('/')
    patient_id = parts[-2]             # LIDC-IDRI-XXXX
    fname = parts[-1]                  # slice_048_4.npy
    score = int(fname.rstrip('.npy').split('_')[-1])
    records.append({'patient': patient_id, 'score': score, 'path': path})

patients = sorted(set(r['patient'] for r in records))
scores = [r['score'] for r in records]
score_counts = Counter(scores)

print(f"환자 수: {len(patients)}")
print(f"\n[Malignancy Score 분포]")
for s in range(1, 6):
    n = score_counts.get(s, 0)
    bar = '█' * (n // 50)
    print(f"  Score {s}: {n:5d}개 ({n/len(scores)*100:.1f}%)  {bar}")

# --- 전략별 라벨링 분포 ---
print("\n[라벨링 전략 비교]")

def assign_label(score, strategy):
    if strategy == 'A':    # 2클래스: 1-3=양성(0), 4-5=악성(1)
        return 0 if score <= 3 else 1
    elif strategy == 'B':  # 2클래스 (3 제외): 1-2=양성(0), 4-5=악성(1), 3=제외
        if score <= 2: return 0
        if score >= 4: return 1
        return None
    elif strategy == 'C':  # 3클래스: 1-2=양성(0), 3=불확실(1), 4-5=악성(2)
        if score <= 2: return 0
        if score == 3: return 1
        return 2

strategies = {
    'A': '2클래스 (1-3=양성, 4-5=악성)',
    'B': '2클래스 (score=3 제외)',
    'C': '3클래스 (1-2=양성, 3=불확실, 4-5=악성)',
}

strategy_results = {}
for strat, desc in strategies.items():
    labels = [assign_label(s, strat) for s in scores]
    labels = [l for l in labels if l is not None]
    cnt = Counter(labels)
    total = len(labels)
    ratio_str = ' : '.join(f"{cnt.get(i,0)}" for i in range(max(cnt.keys())+1))
    strategy_results[strat] = {'cnt': cnt, 'total': total}
    print(f"\n  전략 {strat} — {desc}")
    print(f"  총 {total}개  비율 = {ratio_str}")
    for cls, n in sorted(cnt.items()):
        label_names = {0: '양성', 1: '악성' if strat=='A' else ('불확실' if strat=='C' else '악성'), 2: '악성'}
        print(f"    클래스 {cls} ({label_names.get(cls, '')}): {n:5d}개 ({n/total*100:.1f}%)")

# --- 환자 단위 분포 ---
print("\n[환자 단위 score 분포]")
patient_scores = defaultdict(list)
for r in records:
    patient_scores[r['patient']].append(r['score'])

patient_avg_scores = {p: np.mean(s) for p, s in patient_scores.items()}
avg_vals = list(patient_avg_scores.values())
print(f"  환자 평균 score: min={min(avg_vals):.2f}, max={max(avg_vals):.2f}, mean={np.mean(avg_vals):.2f}")

# 환자별 슬라이스 수 분포
slice_counts = [len(v) for v in patient_scores.values()]
print(f"  환자당 슬라이스 수: min={min(slice_counts)}, max={max(slice_counts)}, mean={np.mean(slice_counts):.1f}")

# --- 시각화 ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 1. Score 분포
ax = axes[0]
vals = [score_counts.get(s, 0) for s in range(1, 6)]
ax.bar(range(1, 6), vals, color='steelblue', edgecolor='black')
ax.set_xlabel('Malignancy Score')
ax.set_ylabel('Count')
ax.set_title(f'Score Distribution\n(총 {len(scores)}개 슬라이스)')
ax.set_xticks(range(1, 6))
for i, v in enumerate(vals):
    ax.text(i+1, v+10, str(v), ha='center', fontsize=9)

# 2. 전략 A vs B (2클래스)
ax = axes[1]
for idx, strat in enumerate(['A', 'B']):
    res = strategy_results[strat]
    x = [c + idx * 0.35 for c in res['cnt'].keys()]
    y = list(res['cnt'].values())
    ax.bar(x, y, width=0.35, label=f'전략 {strat}', alpha=0.8)
ax.set_xlabel('클래스')
ax.set_title('2클래스 전략 비교\n(A: score3=양성, B: score3 제외)')
ax.legend()
ax.set_xticks([0, 1])
ax.set_xticklabels(['양성(0)', '악성(1)'])

# 3. 전략 C (3클래스)
ax = axes[2]
res = strategy_results['C']
class_names = ['양성(0)', '불확실(1)', '악성(2)']
colors = ['#2196F3', '#FF9800', '#F44336']
bars = ax.bar(range(3), [res['cnt'].get(i, 0) for i in range(3)], color=colors, edgecolor='black')
ax.set_xlabel('클래스')
ax.set_title(f"3클래스 전략 C\n(총 {res['total']}개)")
ax.set_xticks(range(3))
ax.set_xticklabels(class_names)
for bar, v in zip(bars, [res['cnt'].get(i, 0) for i in range(3)]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
            str(v), ha='center', fontsize=10)

plt.tight_layout()
save_path = f'{OUT_DIR}/eda_distribution.png'
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n시각화 저장: {save_path}")

# --- 권고사항 ---
print("\n" + "=" * 60)
print("EDA 결과 요약 및 권고사항")
print("=" * 60)
a_cnt = strategy_results['A']['cnt']
imbalance_a = max(a_cnt.values()) / min(a_cnt.values()) if a_cnt else 0
c_cnt = strategy_results['C']['cnt']
imbalance_c = max(c_cnt.values()) / min(c_cnt.values()) if c_cnt else 0
print(f"  전략 A 불균형 비율: {imbalance_a:.2f}:1")
print(f"  전략 C 불균형 비율: {imbalance_c:.2f}:1")
if imbalance_a >= 1.5:
    print("  → WeightedRandomSampler 또는 class_weight 적용 필요")
print(f"\n  [권고] 클래스 수 결정 기준:")
print(f"    - 2클래스(전략A): 성능 벤치마크와 직접 비교 가능, 임상 해석 명확")
print(f"    - 2클래스(전략B): score=3 제거로 경계 모호성 줄임, 샘플 감소")
print(f"    - 3클래스(전략C): 더 세분화, 불확실 클래스 분리 → 불균형 심화 가능")
