import torch
import traceback
from src.run_ablation_study import FullModel, train_one_config, set_seed

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
try:
    print('Starting B-spline...')
    m = FullModel('b_spline').to(device)
    print('B-spline initialized...')
    r = train_one_config(m, 42, 1, device)
    print('B-SPLINE OK:', r)

    print('Starting Focal Loss...')
    m2 = FullModel('mexican_hat').to(device)
    print('Focal initialized...')
    r2 = train_one_config(m2, 42, 1, device, use_focal_loss=True)
    print('FOCAL OK:', r2)
except Exception as e:
    print('ERROR ENCOUNTERED!')
    traceback.print_exc()
