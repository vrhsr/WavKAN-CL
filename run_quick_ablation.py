import json
import torch
from src.run_ablation_study import FullModel, NoGRU, NoRR, PureWavKAN, train_one_config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

configs = [
    ("Full (Mexican Hat)", lambda: FullModel('mexican_hat'), False),
    ("Morlet variant",     lambda: FullModel('morlet'),      False),
    ("DOG variant",        lambda: FullModel('dog'),         False),
    ("B-spline variant",   lambda: FullModel('b_spline'),    False),
    ("Focal Loss",         lambda: FullModel('mexican_hat'), True),
    ("No GRU",             lambda: NoGRU('mexican_hat'),     False),
    ("No RR",              lambda: NoRR('mexican_hat'),      False),
    ("Pure WavKAN",        lambda: PureWavKAN('mexican_hat'),False)
]

out = {}
for name, model_fn, fl in configs:
    m = model_fn().to(device)
    print(f"Running {name}...")
    res = train_one_config(m, 42, 50, device, use_focal_loss=fl)
    out[name] = {"F1": res['f1_macro'], "V": res['v_recall'], "S": res['s_recall']}
    print(f"  {name}: {out[name]}")

with open("results/quick_ablation.json", "w") as f:
    json.dump(out, f, indent=4)
print("Saved to results/quick_ablation.json")
