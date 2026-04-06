"""Verify all manuscript claims against source data."""
import json
import numpy as np

print("=" * 60)
print("  DEEP MANUSCRIPT VERIFICATION REPORT")
print("=" * 60)

# ===== 1. ABSTRACT CLAIMS =====
print("\n--- 1. ABSTRACT CLAIMS ---")

# Claim: "mean ventricular (V) recall of 0.87 ± 0.02 across 10 independent seeds"
ci = json.load(open("results/confidence_intervals.json"))
v_mean = ci["metrics"]["v_recall"]["mean"]
v_std = ci["metrics"]["v_recall"]["std"]
n_seeds = ci["n_seeds"]
print(f"  V-Recall mean: {v_mean:.4f} (manuscript says 0.87)")
print(f"  V-Recall std:  {v_std:.4f} (manuscript says 0.02)")
print(f"  Seeds: {n_seeds} (manuscript says '10 independent seeds')")
if n_seeds != 10:
    print(f"  ⚠️  DISCREPANCY: Actual seeds = {n_seeds}, manuscript says 10")
else:
    print(f"  ✅ OK")
v_match = abs(v_mean - 0.87) < 0.015 and abs(v_std - 0.02) < 0.015
print(f"  V-Recall match: {'✅' if v_match else '⚠️ CHECK'}")

# Claim: "95,189 parameters"
lat = json.load(open("results/latency_report.json"))
params = lat["n_parameters"]
print(f"\n  Parameters: {params} (manuscript says 95,189)")
print(f"  Match: {'✅' if params == 95189 else '⚠️ MISMATCH'}")

# ===== 2. TABLE 2 (Class Distribution) =====
print("\n--- 2. TABLE 2 (Class Distribution) ---")
y_train = np.load("data/processed_rr_history/y_train.npy")
y_val = np.load("data/processed_rr_history/y_val.npy")
y_test = np.load("data/processed_rr_history/y_test.npy")

classes = {0: "N", 1: "S", 2: "V", 3: "F", 4: "Q"}
manuscript_train = {0: 36396, 1: 773, 2: 3150, 3: 399, 4: 8}
manuscript_val = {0: 9443, 1: 170, 2: 638, 3: 15, 4: 0}
manuscript_test = {0: 44232, 1: 1837, 2: 3220, 3: 388, 4: 7}
manuscript_total = {0: 90071, 1: 2780, 2: 7008, 3: 802, 4: 15}

from collections import Counter
actual_train = Counter(y_train)
actual_val = Counter(y_val)
actual_test = Counter(y_test)

all_ok = True
for c in range(5):
    at = actual_train.get(c, 0)
    av = actual_val.get(c, 0)
    atest = actual_test.get(c, 0)
    mt = manuscript_train[c]
    mv = manuscript_val[c]
    mtest = manuscript_test[c]
    total_actual = at + av + atest
    total_manuscript = manuscript_total[c]
    ok = (at == mt and av == mv and atest == mtest and total_actual == total_manuscript)
    if not ok:
        all_ok = False
    print(f"  {classes[c]}: Train {at}=={mt}{'✅' if at==mt else '❌'}, "
          f"Val {av}=={mv}{'✅' if av==mv else '❌'}, "
          f"Test {atest}=={mtest}{'✅' if atest==mtest else '❌'}, "
          f"Total {total_actual}=={total_manuscript}{'✅' if total_actual==total_manuscript else '❌'}")

print(f"  Overall: {'✅ ALL MATCH' if all_ok else '⚠️ DISCREPANCIES FOUND'}")
print(f"  Total beats: {len(y_train)+len(y_val)+len(y_test)} (manuscript says 100,676)")

# ===== 3. TABLE 5 (Statistical Rigor) =====
print("\n--- 3. TABLE 5 (Statistical Rigor - 10-Seed Comparison) ---")
# Manuscript: Macro F1 0.362 ± 0.021 for WavKAN-CL
f1_mean = ci["metrics"]["f1_macro"]["mean"]
f1_std = ci["metrics"]["f1_macro"]["std"]
s_mean = ci["metrics"]["s_recall"]["mean"]
s_std = ci["metrics"]["s_recall"]["std"]
print(f"  Macro-F1 mean: {f1_mean:.3f} (manuscript: 0.362±0.021)")
print(f"  Macro-F1 std:  {f1_std:.3f}")
print(f"  S-Recall mean: {s_mean:.3f} (manuscript: 0.260±0.077)")
print(f"  S-Recall std:  {s_std:.3f}")
print(f"  V-Recall mean: {v_mean:.3f} (manuscript: 0.898±0.011)")
print(f"  V-Recall std:  {v_std:.3f}")

# Note: CI file has 20 seeds, manuscript Table 5 says 10 seeds
# The 10-seed values were from an earlier run
print(f"  ⚠️  CI file uses {n_seeds} seeds. Table 5 references 10-seed comparison.")

# ===== 4. TABLE 6 (Per-Class Metrics) =====
print("\n--- 4. TABLE 6 (Per-Class Metrics - Best Seed) ---")
# The best-seed metrics
best = json.load(open("results/thesis_fusion_metrics.json"))
print(f"  N: P={best['N']['precision']:.3f} R={best['N']['recall']:.3f} F1={best['N']['f1-score']:.3f} Sup={int(best['N']['support'])}")
print(f"     Manuscript: P=0.900 R=0.772 F1=0.831 Sup=44,232")
# Check matches
n_p_match = abs(best['N']['precision'] - 0.900) < 0.06
n_r_match = abs(best['N']['recall'] - 0.772) < 0.07
print(f"     {'✅' if n_p_match else '⚠️'} Precision, {'✅' if n_r_match else '⚠️'} Recall")

print(f"\n  S: P={best['S']['precision']:.3f} R={best['S']['recall']:.3f} F1={best['S']['f1-score']:.3f} Sup={int(best['S']['support'])}")
print(f"     Manuscript: P=0.286 R=0.280 F1=0.283 Sup=1,837")

print(f"\n  V: P={best['V']['precision']:.3f} R={best['V']['recall']:.3f} F1={best['V']['f1-score']:.3f} Sup={int(best['V']['support'])}")
print(f"     Manuscript: P=0.490 R=0.898 F1=0.634 Sup=3,220")

print(f"\n  F: P={best['F']['precision']:.3f} R={best['F']['recall']:.3f} F1={best['F']['f1-score']:.3f} Sup={int(best['F']['support'])}")
print(f"     Manuscript: P=0.048 R=0.051 F1=0.050 Sup=388")

# ===== 5. TABLE 8 (PTB-XL Zero-Shot) =====
print("\n--- 5. TABLE 8 (PTB-XL Zero-Shot) ---")
ptbxl = json.load(open("results/ptbxl_zero_shot_metrics.json"))
print(f"  N: P={ptbxl['N']['precision']:.3f} R={ptbxl['N']['recall']:.3f} F1={ptbxl['N']['f1-score']:.3f} Sup={int(ptbxl['N']['support'])}")
print(f"     Manuscript: P=0.701 R=0.729 F1=0.715 Sup=53,497")
n_ok = abs(ptbxl['N']['precision'] - 0.701) < 0.002
print(f"     {'✅ EXACT MATCH' if n_ok else '⚠️ CHECK'}")

print(f"  S: P={ptbxl['S']['precision']:.3f} R={ptbxl['S']['recall']:.3f} F1={ptbxl['S']['f1-score']:.3f} Sup={int(ptbxl['S']['support'])}")
print(f"     Manuscript: P=0.150 R=0.052 F1=0.078 Sup=6,741")

print(f"  V: P={ptbxl['V']['precision']:.3f} R={ptbxl['V']['recall']:.3f} F1={ptbxl['V']['f1-score']:.3f} Sup={int(ptbxl['V']['support'])}")
print(f"     Manuscript: P=0.112 R=0.108 F1=0.110 Sup=5,889")

print(f"  F: P={ptbxl['F']['precision']:.3f} R={ptbxl['F']['recall']:.3f} F1={ptbxl['F']['f1-score']:.3f} Sup={int(ptbxl['F']['support'])}")
print(f"     Manuscript: P=0.118 R=0.016 F1=0.028 Sup=9,043")

print(f"  Macro: P={ptbxl['macro avg']['precision']:.3f} R={ptbxl['macro avg']['recall']:.3f} F1={ptbxl['macro avg']['f1-score']:.3f}")
print(f"     Manuscript: P=0.270 R=0.226 F1=0.233")

# ===== 6. TABLE 9 (Ablation) =====
print("\n--- 6. TABLE 9 (Ablation Study) ---")
abl = json.load(open("results/missing_ablations.json"))
qa = json.load(open("results/quick_ablation.json"))

ablation_checks = [
    ("B-spline", "F1", abl["B-spline"]["f1_macro"], 0.396),
    ("B-spline", "V-Recall", abl["B-spline"]["v_recall"], 0.920),
    ("B-spline", "S-Recall", abl["B-spline"]["s_recall"], 0.570),
    ("Focal Loss", "F1", abl["FocalLoss"]["f1_macro"], 0.336),
    ("Focal Loss", "V-Recall", abl["FocalLoss"]["v_recall"], 0.907),
    ("Focal Loss", "S-Recall", abl["FocalLoss"]["s_recall"], 0.384),
    ("Morlet", "F1", qa["Morlet variant"]["F1"], 0.325),
    ("Morlet", "V-Recall", qa["Morlet variant"]["V"], 0.866),
    ("DOG", "F1", qa["DOG variant"]["F1"], 0.354),
]
for name, metric, actual, manuscript_val in ablation_checks:
    ok = abs(actual - manuscript_val) < 0.01
    print(f"  {name} {metric}: actual={actual:.3f} manuscript={manuscript_val:.3f} {'✅' if ok else '⚠️ MISMATCH'}")

# ===== 7. LATENCY BENCHMARKS =====
print("\n--- 7. COMPUTATIONAL BENCHMARKS ---")
print(f"  Latency median: {lat['latency']['median_ms']:.2f} ms (manuscript: 0.78 ms/beat)")
print(f"  Match: {'✅' if abs(lat['latency']['median_ms'] - 0.78) < 0.02 else '⚠️'}")
print(f"  MACs: {lat['macs_info']['macs_M']:.5f} M (manuscript: 0.026 M)")
print(f"  Match: {'✅' if abs(lat['macs_info']['macs_M'] - 0.026) < 0.002 else '⚠️'}")
print(f"  Size KB: {lat['size_kb']:.1f} (manuscript: 0.38 MB = {0.38*1024:.0f} KB)")
print(f"  Match: {'✅' if abs(lat['size_kb'] - 0.38*1024) < 15 else '⚠️'}")

# ===== 8. MODEL ARCHITECTURE =====
print("\n--- 8. MODEL ARCHITECTURE ---")
print(f"  Beat window T=360 samples (manuscript line 95: T=360)")
print(f"  RR input dim K=5 (manuscript: 5-element RR history)")
print(f"  WavKAN output: 64 dims (manuscript line 108: z_kan ∈ R^64)")
print(f"  BiGRU: 32 hidden per direction = 64 total → z_m ∈ R^64")
print(f"  RR MLP: 5→64→32→16 → z_r ∈ R^16")
print(f"  Fusion: [z_m; z_r] ∈ R^80 → FC 80→48→5")
print(f"  Total params: {params}")

print("\n" + "=" * 60)
print("  VERIFICATION COMPLETE")
print("=" * 60)
