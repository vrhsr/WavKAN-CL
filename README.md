# Wavelet Kolmogorov–Arnold Networks for Inter-Patient ECG Heartbeat Classification

Code, per-seed results and checkpoints for the manuscript *"Wavelet Kolmogorov–Arnold Networks for Inter-Patient ECG Heartbeat Classification: A Controlled Multi-Seed Evaluation"* (`Submission_Array/manuscript.tex`). The repository name (WavKAN-CL) is historical.

## What the study is

A controlled evaluation, not a method proposal. A 153,045-parameter wavelet KAN (PC-WavKAN) is compared with four baselines on MIT-BIH under the inter-patient DS1/DS2 protocol of de Chazal et al. The comparison uses 20 paired seeds, Holm–Bonferroni correction, effect sizes and confidence intervals, and architecture selection on validation data only. Main findings:

- No statistically detectable Macro-F1 difference from the baselines (every paired 95% CI within ±0.03). This is not an equivalence claim.
- A prior-centred initialisation of the wavelet parameters helped relative to an isotropic one (base configuration). The mother wavelet did not measurably matter. Self-attention over RR intervals was harmful.
- The wavelet parameters do not support a physiological interpretation (exchangeable channel blocks; a trained null model).
- Poor supraventricular operating point for all models.

## Layout

- `src/`: preprocessing, training, evaluation, statistics, figures. Key entry points:
  - `process_data.py`: MIT-BIH arrays.
  - `train_pca.py`: PC-WavKAN.
  - `baselines_extended.py`: the baselines.
  - `extract_matched.py`, `eval_inference_sensitivity.py`: the external and RR-sensitivity analyses.
  - `verify_manuscript_numbers.py`: re-derives every number in the manuscript.
- `models/wavkan_v2.py`: model definition. The class is still named `WavKAN_v2`; the evaluated configuration is `WavKAN_v2(use_rr_attn=False)`.
- `results/`: per-seed results and checkpoints. See `results/README.md` for which directories back the paper.
- `tests/`: regression tests (`python -m pytest -q tests`).
- `AUDIT_FINDINGS.md`, `CHANGELOG.md`: the project's integrity audit and fix record.

## Reproducing

1. `pip install -r requirements.txt`. The verification and inference-only analyses ran on Python 3.13, PyTorch 2.6 (CPU) and NeuroKit2 0.2.13. The published models were trained on a separate CUDA machine without cuDNN determinism flags, so retraining reproduces the results statistically, not bit for bit.
2. Download the 44 non-paced MIT-BIH records into `data/raw/` (PhysioNet `mitdb` 1.0.0), then `python src/process_data.py`. The expected counts are in `configs/mitbih_split_counts.json`.
3. Training: `python src/train_pca.py --seed S --epochs 100 --no-rr-attn` and `python src/baselines_extended.py --model M --seeds S`.
4. Verify the manuscript against the stored results: `python src/verify_manuscript_numbers.py` (exits non-zero on any mismatch).

No data ships with the repository (PhysioNet licence and size).
