import json
import torch
import numpy as np
from src.run_ablation_study import FullModel, train_one_config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Running B-spline...")
m_bspline = FullModel('b_spline').to(device)
res_bspline = train_one_config(m_bspline, 42, 50, device)

print("Running Focal...")
m_focal = FullModel('mexican_hat').to(device)
res_focal = train_one_config(m_focal, 42, 50, device, use_focal_loss=True)

out = {
    "B-spline": res_bspline,
    "FocalLoss": res_focal
}

with open("results/missing_ablations.json", "w") as f:
    json.dump(out, f, indent=4)
print("Saved to missing_ablations.json")
