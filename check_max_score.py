
import glob
import json
import numpy as np

files = glob.glob('results/multiseed/curriculum/seed_*/test_metrics.json')
scores = []
for f in files:
    d = json.load(open(f))
    scores.append(d['v_recall'])

print(f"Max V-Recall: {max(scores):.4f}")
print(f"Mean V-Recall: {np.mean(scores):.4f}")
print(f"Scores: {scores}")
