"""
verify_manuscript_numbers.py -- re-derive every quantitative claim in
Submission_Array/manuscript.tex from the real result files in results/
(repointed 2026-09-30 from the superseded Submission_JBHI/ieee_manuscript_v2.tex)
and report agreement or disagreement, claim by claim.

Written 2026-09-03 during the publication-readiness audit. This is the honest
replacement for the quarantine-adjacent verify_manuscript.py at the repo root,
which compared result JSONs against hardcoded Python literals, never read the
.tex at all, and could not run because 5 of its 6 required files are absent.

How this one differs, and why it can be trusted:
  * Every "real" value is recomputed here from per-seed files on disk
    (test_metrics.json, training_history.json, confusion_matrix.npy, the
    trained .pth checkpoints) -- never read back from a summary another
    script produced, except where that summary IS the artefact under test.
  * Paired statistics are recomputed from scratch (paired Wilcoxon,
    Holm-Bonferroni within the declared family, Cohen's d, 95% CI), so a
    pairing or correction bug in the original analysis would surface as a
    mismatch rather than being reproduced.
  * The only hardcoded numbers are the CLAIMS being tested, i.e. what the
    manuscript currently prints. A mismatch therefore means the manuscript
    and the data disagree, and one of them must be corrected.

Run after ANY change to a reported number, and before any resubmission:
    python src/verify_manuscript_numbers.py

Exits non-zero if any claim fails, so it can gate a commit.
"""
import json, glob, os, sys
import numpy as np
from scipy.stats import wilcoxon, mannwhitneyu, t as tdist

ROOT = r"E:\study-buddy\WavKAN-CL-main"
os.chdir(ROOT)
OK = []
BAD = []


def chk(label, claimed, actual, tol):
    good = abs(claimed - actual) <= tol
    (OK if good else BAD).append((label, claimed, actual))
    print("%-58s claim=%-10s real=%-12s %s" %
          (label, round(claimed, 5), round(actual, 5), "ok" if good else "MISMATCH"))


def load(d, key):
    o = {}
    for f in glob.glob(os.path.join(d, "seed_*", "test_metrics.json")):
        o[os.path.basename(os.path.dirname(f))] = json.load(open(f))[key]
    return o


def ms(d, key):
    v = list(load(d, key).values())
    return np.mean(v), np.std(v, ddof=1), len(v)


print("=" * 100)
print("TABLE: Primary comparison (DS2 Macro-F1)")
print("=" * 100)
m, s, n = ms("results/ablation_no_rr_attn", "macro_f1")
chk("PC-WavKAN Macro-F1 mean", 0.357, m, 0.0005)
chk("PC-WavKAN Macro-F1 std", 0.019, s, 0.0005)
chk("PC-WavKAN n seeds", 20, n, 0)
bstats = json.load(open("results/no_rr_attn_vs_baselines_stats.json"))["comparisons"]
for k, cm, cs, cd, chp, cdiff in [
        ("resnet1d", 0.351, 0.019, 0.19, 1.00, 0.006),
        ("transformer", 0.353, 0.040, 0.12, 1.00, 0.004),
        ("cnn_focal", 0.362, 0.015, -0.22, 1.00, -0.005),
        ("bspline_kan", 0.371, 0.024, -0.43, 0.36, -0.014)]:
    v = bstats[k]
    chk(k + " mean", cm, v["mean_b"], 0.0005)
    chk(k + " std", cs, v["std_b"], 0.0005)
    chk(k + " cohens d", cd, v["cohens_d"], 0.005)
    chk(k + " holm p", chp, v["holm_p"], 0.005)
    chk(k + " diff", cdiff, v["mean_diff"], 0.0005)

print("\n" + "=" * 100)
print("TABLE: 95% CIs of paired differences (primary)")
print("=" * 100)
ref = load("results/ablation_no_rr_attn", "macro_f1")
for name, d, lo, hi in [("ResNet1D", "results/baseline_resnet1d", -0.008, 0.020),
                        ("Transformer", "results/baseline_transformer", -0.013, 0.022),
                        ("CNN+Focal", "results/baseline_cnn_focal", -0.017, 0.006),
                        ("B-Spline KAN", "results/baseline_bspline_kan", -0.029, 0.001)]:
    b = load(d, "macro_f1")
    sd = sorted(set(ref) & set(b))
    dz = np.array([ref[k] - b[k] for k in sd])
    ci = tdist.ppf(0.975, len(dz) - 1) * dz.std(ddof=1) / np.sqrt(len(dz))
    chk(name + " CI low", lo, dz.mean() - ci, 0.0006)
    chk(name + " CI high", hi, dz.mean() + ci, 0.0006)

print("\n" + "=" * 100)
print("TABLE: RAC vs no-RAC on adopted config (5-metric Holm family)")
print("=" * 100)
A, B = "results/ablation_no_rr_attn_no_augment", "results/ablation_no_rr_attn_no_curriculum_no_augment"
raw = {}
for key in ("macro_f1", "v_recall", "s_recall", "f_recall", "n_recall"):
    a, b = load(A, key), load(B, key)
    sd = sorted(set(a) & set(b))
    av, bv = [a[k] for k in sd], [b[k] for k in sd]
    dz = np.array(av) - np.array(bv)
    raw[key] = (np.mean(bv), np.std(bv, ddof=1), np.mean(av), np.std(av, ddof=1),
                dz.mean() / dz.std(ddof=1), wilcoxon(av, bv)[1], dz, len(sd))
order = sorted(raw, key=lambda k: raw[k][5])
prev, holm = 0.0, {}
for i, k in enumerate(order):
    h = min(1.0, max(prev, raw[k][5] * (5 - i)))
    prev = holm[k] = h
claims = {"macro_f1": (0.351, 0.014, 0.364, 0.021, 0.52, 0.098),
          "s_recall": (0.137, 0.037, 0.222, 0.057, 1.47, 2.9e-5),
          "f_recall": (0.0003, None, 0.0022, None, 0.88, 0.013),
          "v_recall": (0.880, 0.024, 0.880, 0.017, -0.00, 0.756),
          "n_recall": (0.927, 0.021, 0.917, 0.024, -0.32, 0.228)}
for k, c in claims.items():
    r = raw[k]
    chk("RAC " + k + " no-RAC mean", c[0], r[0], 0.0006)
    if c[1] is not None:
        chk("RAC " + k + " no-RAC std", c[1], r[1], 0.0006)
    chk("RAC " + k + " RAC mean", c[2], r[2], 0.0006)
    if c[3] is not None:
        chk("RAC " + k + " RAC std", c[3], r[3], 0.0006)
    chk("RAC " + k + " cohens d", c[4], r[4], 0.006)
    chk("RAC " + k + " holm p", c[5], holm[k], max(0.001, c[5] * 0.05))
dz = raw["s_recall"][6]
ci = tdist.ppf(0.975, 19) * dz.std(ddof=1) / np.sqrt(20)
chk("RAC S-Rec CI low", 0.058, dz.mean() - ci, 0.0006)
chk("RAC S-Rec CI high", 0.112, dz.mean() + ci, 0.0006)

print("\n" + "=" * 100)
print("TABLE: DS1-validation ablation (selection)")
print("=" * 100)


def pv(d):
    o = {}
    for p in sorted(glob.glob(os.path.join(d, "seed_*", "training_history.json"))):
        h = json.load(open(p))
        o[os.path.basename(os.path.dirname(p))] = max(e["val_macro_f1"] for e in h)
    return o


base = pv("results/wavkan_v2_curriculum")
chk("base val Macro-F1 mean", 0.4145, np.mean(list(base.values())), 0.0006)
chk("base val Macro-F1 std", 0.0201, np.std(list(base.values()), ddof=1), 0.0006)
alts = {"No PCWI": ("results/ablation_no_pcwi", 0.3932, 0.0129, -0.0213, -1.05, 0.0001),
        "No PWAM": ("results/ablation_no_pwam", 0.3976, 0.0184, -0.0169, -0.54, 0.164),
        "No RAC": ("results/wavkan_v2_baseline", 0.4053, 0.0173, -0.0092, -0.37, 0.624),
        "Morlet": ("results/ablation_wavelet_morlet", 0.4219, 0.0187, 0.0074, 0.31, 0.568),
        "DOG": ("results/ablation_wavelet_dog", 0.4041, 0.0123, -0.0104, -0.45, 0.164),
        "B-spline": ("results/ablation_wavelet_bspline", 0.4116, 0.0151, -0.0029, -0.14, 0.624),
        "No RR-attn": ("results/ablation_no_rr_attn", 0.4397, 0.0140, 0.0252, 1.15, 0.0008)}
res = {}
for nm, (d, *_) in alts.items():
    v = pv(d)
    sd = sorted(set(base) & set(v))
    dz = np.array([v[k] - base[k] for k in sd])
    res[nm] = (np.mean([v[k] for k in sd]), np.std([v[k] for k in sd], ddof=1),
               dz.mean(), dz.mean() / dz.std(ddof=1), wilcoxon([v[k] for k in sd], [base[k] for k in sd])[1])
order = sorted(res, key=lambda k: res[k][4])
prev, holm2 = 0.0, {}
for i, k in enumerate(order):
    h = min(1.0, max(prev, res[k][4] * (7 - i)))
    prev = holm2[k] = h
for nm, (d, cm, cs, cdf, cd, chp) in alts.items():
    r = res[nm]
    chk("abl " + nm + " val mean", cm, r[0], 0.0006)
    chk("abl " + nm + " val std", cs, r[1], 0.0006)
    chk("abl " + nm + " delta", cdf, r[2], 0.0006)
    chk("abl " + nm + " d", cd, r[3], 0.006)
    chk("abl " + nm + " holm p", chp, holm2[nm], max(0.0002, chp * 0.06))

print("\n" + "=" * 100)
print("TABLE: PPR")
print("=" * 100)
sys.path.insert(0, ROOT)
import torch
from src.wavkan_pcwi import ECG_PRIORS
from models.wavkan_v2 import WavKAN_v2
MU_R, GA_R = 0.40, 0.20


def eppr(kan, per_comp=False):
    v = {}
    for comp, p_ in ECG_PRIORS.items():
        c0, c1 = p_["channels"]
        mu = kan.translation.data[c0:c1]
        ga = kan.scale.data[c0:c1].abs()
        emu = (mu - p_["mu_center"]).abs().mean().item()
        ega = (ga - p_["gamma"]).abs().mean().item()
        bm = abs(mu.mean().item() - p_["mu_center"])
        v[comp] = (max(0.0, 1 - (emu / MU_R + ega / GA_R) / 2), emu, ega, bm)
    if per_comp:
        return v
    return tuple(float(np.mean([x[i] for x in v.values()])) for i in range(4))


def ppr_rows(pattern, **kw):
    out = []
    for ck in sorted(glob.glob(pattern)):
        m_ = WavKAN_v2(**kw)
        m_.load_state_dict(torch.load(ck, map_location="cpu"))
        out.append(eppr(m_.kan))
    return np.array(out)


def ppr_untrained(pcwi):
    out = []
    for sd_ in range(1000, 1020):
        torch.manual_seed(sd_)
        out.append(eppr(WavKAN_v2(use_pcwi=pcwi, use_pwam=True, use_rr_attn=False).kan))
    return np.array(out)


T_ = ppr_rows("results/ablation_no_rr_attn/seed_*/best_model.pth",
              use_pcwi=True, use_pwam=True, use_rr_attn=False)
U_pc, U_iso = ppr_untrained(True), ppr_untrained(False)
# Trained isotropic null: the 20 checkpoints of the no-PCWI ablation arm (base config).
R_ = ppr_rows("results/ablation_no_pcwi/seed_*/best_model.pth",
              use_pcwi=False, use_pwam=True, use_rr_attn=True)
chk("PPR trained n", 20, len(T_), 0)
chk("PPR trained-isotropic n", 20, len(R_), 0)
for lbl, A_, cm, cs, cmu, cga in [("untrained PCWI", U_pc, 0.9729, 0.0001, 0.013, 0.004),
                                  ("trained PCWI", T_, 0.7826, 0.0351, 0.064, 0.055),
                                  ("untrained isotropic", U_iso, 0.6698, 0.0008, 0.161, 0.051),
                                  ("trained isotropic", R_, 0.5794, 0.0173, 0.177, 0.080)]:
    chk("PPR " + lbl + " mean", cm, A_[:, 0].mean(), 0.0012)
    chk("PPR " + lbl + " std", cs, A_[:, 0].std(ddof=1), 0.0006)
    chk("PPR " + lbl + " |dmu|", cmu, A_[:, 1].mean(), 0.0015)
    chk("PPR " + lbl + " |dgamma|", cga, A_[:, 2].mean(), 0.0015)
chk("PPR trained block-mean |dmu| (0.004)", 0.004, T_[:, 3].mean(), 0.0006)
chk("PPR |dmu| growth ~fivefold (0.064/0.013)", 5.0, T_[:, 1].mean() / U_pc[:, 1].mean(), 0.3)
for lbl, X_, Y_ in [("trained PCWI vs untrained iso", T_, U_iso),
                    ("trained iso vs untrained iso", R_, U_iso)]:
    p_ = mannwhitneyu(X_[:, 0], Y_[:, 0], alternative="two-sided")[1]
    sep = (X_[:, 0].min() > Y_[:, 0].max()) or (X_[:, 0].max() < Y_[:, 0].min())
    chk("MW two-sided p (7e-8) " + lbl, 7e-8, p_, 0.5e-8)
    chk("complete separation " + lbl, 1, 1 if sep else 0, 0)
chk("trained isotropic moves AWAY from priors", 1,
    1 if R_[:, 0].mean() < U_iso[:, 0].mean() else 0, 0)

# Exchangeability of the 64 KAN channels (Methods): permuting hidden units with the
# matching downstream weights leaves the function unchanged, and the permuted
# physiological model carries exactly the swap_pt priors (AUDIT_FINDINGS.md H62).
import copy as _copy  # noqa: E402
torch.manual_seed(0)
_ph = WavKAN_v2(use_rr_attn=False, prior_assignment="physiological").eval()
torch.manual_seed(0)
_sw = WavKAN_v2(use_rr_attn=False, prior_assignment="swap_pt").eval()
_perm = torch.tensor(list(range(32)) + list(range(48, 64)) + list(range(32, 48)))
_p2 = _copy.deepcopy(_ph)
with torch.no_grad():
    for _n in ["weights", "translation", "scale", "linear_w"]:
        getattr(_p2.kan, _n).copy_(getattr(_ph.kan, _n)[_perm])
    _p2.kan_norm.weight.copy_(_ph.kan_norm.weight[_perm])
    _p2.kan_norm.bias.copy_(_ph.kan_norm.bias[_perm])
    for _n in ["weight_ih_l0", "weight_ih_l0_reverse"]:
        getattr(_p2.bigru, _n).copy_(getattr(_ph.bigru, _n)[:, _perm])
    _x = torch.randn(256, 360)
    _r = torch.rand(256, 5) + 0.5
    _md = (_ph(_x, _r) - _p2(_x, _r)).abs().max().item()
chk("exchangeability: max |f - f_perm| < 1e-6 (reported ~1e-7)", 1, 1 if _md < 1e-6 else 0, 0)


def _blk(m, a, b):
    return (round(m.kan.translation[a:b].mean().item(), 3), round(m.kan.scale[a:b].mean().item(), 3))


chk("exchangeability: permuted priors == swap_pt priors", 1,
    1 if all(_blk(_p2, a, b) == _blk(_sw, a, b) for a, b in [(0, 32), (32, 48), (48, 64)]) else 0, 0)

print("\n" + "=" * 100)
print("TABLE: Cross-dataset + confusion diagnostic + fewshot + augmentation + deployment")
print("=" * 100)
# The superseded cross-dataset table (multidataset_final_stats.json, from an
# unfiltered, lead-I pipeline) and the few-shot section were removed from the
# manuscript on 2026-09-30 (AUDIT_FINDINGS.md H58). The matched-pipeline external
# results are checked in the EXTERNAL section below.

ag = json.load(open("results/noise_augmentation_final/augmentation_comparison.json"))["summary"]
chk("aug isolated none S", 0.316, ag["none"]["s_recall"]["mean"], 0.0006)
chk("aug isolated smote S", 0.648, ag["smote"]["s_recall"]["mean"], 0.0006)
chk("aug isolated smote largest MF1 cost", 1,
    1 if ag["smote"]["macro_f1"]["mean"] == min(v["macro_f1"]["mean"] for v in ag.values()) else 0, 0)
chk("aug isolated smote largest S gain", 1,
    1 if ag["smote"]["s_recall"]["mean"] == max(v["s_recall"]["mean"] for v in ag.values()) else 0, 0)
for _st in ("gaussian", "baseline_wander", "combined"):
    chk("aug isolated %s raises V" % _st, 1,
        1 if ag[_st]["v_recall"]["mean"] > ag["none"]["v_recall"]["mean"] else 0, 0)
    chk("aug isolated %s lowers MF1" % _st, 1,
        1 if ag[_st]["macro_f1"]["mean"] < ag["none"]["macro_f1"]["mean"] else 0, 0)

h36 = json.load(open("results/h36_no_augment_ablation_report.json"))["headline_curriculum_augment_vs_no_augment"]["per_metric"]
chk("h36 MF1 with-aug", 0.357, h36["macro_f1"]["with_augment_mean"], 0.0006)
chk("h36 MF1 no-aug", 0.364, h36["macro_f1"]["no_augment_mean"], 0.0006)
chk("h36 MF1 holm p", 0.46, h36["macro_f1"]["holm_adjusted_p"], 0.006)
chk("h36 F-Rec with-aug", 0.0061, h36["f_recall"]["with_augment_mean"], 0.0002)
chk("h36 F-Rec no-aug", 0.0022, h36["f_recall"]["no_augment_mean"], 0.0002)
chk("h36 F-Rec holm p", 0.043, h36["f_recall"]["holm_adjusted_p"], 0.002)

dep = [json.load(open("results/deployment_final_runs/run%d/benchmark_report.json" % r)) for r in (1, 2, 3)]
chk("dep FP32 size MB", 0.596, dep[0]["fp32_size_mb"], 0.0006)
chk("dep INT8 size MB", 0.551, dep[0]["int8_size_mb"], 0.0006)
chk("dep compression", 1.08, dep[0]["compression_ratio"], 0.005)
f32 = np.mean([d["latency_fp32"]["mean_ms"] for d in dep])
i8 = np.mean([d["latency_int8"]["mean_ms"] for d in dep])
chk("dep FP32 lat mean", 2.93, f32, 0.006)
chk("dep FP32 lat spread", 0.38, np.std([d["latency_fp32"]["mean_ms"] for d in dep], ddof=1), 0.006)
chk("dep FP32 median", 2.78, np.mean([d["latency_fp32"]["median_ms"] for d in dep]), 0.006)
chk("dep INT8 lat", 4.00, i8, 0.006)
chk("dep INT8 spread", 0.35, np.std([d["latency_int8"]["mean_ms"] for d in dep], ddof=1), 0.006)
chk("dep speedup", 0.73, f32 / i8, 0.006)
chk("dep batch64", 0.429, np.mean([d["throughput_fp32"]["64"]["latency_ms"] for d in dep]), 0.0006)
chk("dep peak rss", 235, dep[0]["peak_rss_mb"], 0.6)

rr = json.load(open("results/rr_ablation_real/rr_ablation_report.json"))
chk("rr baseline S-Rec", 0.1225, rr["baseline"]["s_recall"]["mean"], 0.0002)
chk("rr baseline S-Rec std", 0.0524, rr["baseline"]["s_recall"]["std"], 0.0002)
pp = rr["per_position"]
srt = sorted(pp, key=lambda k: pp[k]["p_value_s"])
prev, hrr = 0.0, {}
for i, k in enumerate(srt):
    h = min(1.0, max(prev, pp[k]["p_value_s"] * (5 - i)))
    prev = hrr[k] = h
for k, dl, ds_, hp in [("RR-2", -0.0020, 0.0077, 0.221), ("RR-1", -0.0016, 0.0075, 0.221),
                       ("RR0 (Pre)", -0.0323, 0.0492, 3e-4), ("RR+1 (Post)", 0.0022, 0.0095, 0.358),
                       ("RR+2 (Post)", -0.0045, 0.0103, 0.014)]:
    chk("rr " + k + " delta", dl, pp[k]["s_recall_delta_mean"], 0.0001)
    chk("rr " + k + " delta std", ds_, pp[k]["s_recall_delta_std"], 0.0001)
    chk("rr " + k + " holm p", hp, hrr[k], max(0.0002, hp * 0.06))

cmx = np.load("results/ablation_no_rr_attn/seed_42/confusion_matrix.npy")
rn = cmx / cmx.sum(1, keepdims=True)
chk("DS2 cm S->V", 0.42, rn[1, 2], 0.005)
chk("DS2 cm S->N", 0.30, rn[1, 0], 0.005)
chk("DS2 cm F->N", 0.87, rn[3, 0], 0.005)
chk("DS2 cm V-Rec", 0.905, rn[2, 2], 0.0009)
chk("DS2 cm N->V (<6%)", 0.0498, rn[0, 2], 0.0009)
chk("DS2 cm V->N (<6%)", 0.0512, rn[2, 0], 0.0009)

old = json.load(open("results/wavkan_v2_20seed_comparison.json"))["metrics"]
chk("replication S-Rec no-RAC", 0.079, old["s_recall"]["baseline_mean"], 0.0006)
chk("replication S-Rec no-RAC std", 0.016, old["s_recall"]["baseline_std"], 0.0006)
chk("replication S-Rec RAC", 0.123, old["s_recall"]["curriculum_mean"], 0.0006)
chk("replication S-Rec RAC std", 0.052, old["s_recall"]["curriculum_std"], 0.0006)
chk("replication holm p", 0.0016, old["s_recall"]["holm_adjusted_p"], 0.0002)
chk("replication d", 0.77, abs(old["s_recall"]["cohens_d"]), 0.006)

vs = {d: None for d in ("results/wavkan_v2_curriculum", "results/wavkan_v2_baseline")}
a = {k: v for k, v in pv("results/wavkan_v2_curriculum").items()}
sc, sb = {}, {}
for d, o in (("results/wavkan_v2_curriculum", sc), ("results/wavkan_v2_baseline", sb)):
    for p in sorted(glob.glob(os.path.join(d, "seed_*", "training_history.json"))):
        h = json.load(open(p))
        o[os.path.basename(os.path.dirname(p))] = max(h, key=lambda e: e["val_macro_f1"])["val_s_recall"]
sd = sorted(set(sc) & set(sb))
dz = np.array([sc[k] - sb[k] for k in sd])
chk("val S-Rec no-RAC", 0.218, np.mean([sb[k] for k in sd]), 0.0006)
chk("val S-Rec no-RAC std", 0.048, np.std([sb[k] for k in sd], ddof=1), 0.0006)
chk("val S-Rec RAC", 0.305, np.mean([sc[k] for k in sd]), 0.0006)
chk("val S-Rec RAC std", 0.047, np.std([sc[k] for k in sd], ddof=1), 0.0006)
chk("val S-Rec d", 1.36, dz.mean() / dz.std(ddof=1), 0.006)
chk("val S-Rec raw p", 2.4e-4, wilcoxon([sc[k] for k in sd], [sb[k] for k in sd])[1], 3e-5)

pset = set()
for f in glob.glob("results/ablation_no_rr_attn/seed_*/test_metrics.json"):
    pset.add(json.load(open(f))["model_params"])
chk("PC-WavKAN params", 153045, list(pset)[0], 0)
pset2 = set()
for f in glob.glob("results/wavkan_v2_curriculum/seed_*/test_metrics.json"):
    pset2.add(json.load(open(f))["model_params"])
chk("base config params", 154325, list(pset2)[0], 0)
for d, cp in [("results/baseline_resnet1d", 62869), ("results/baseline_transformer", 29653),
              ("results/baseline_cnn_focal", 71061), ("results/baseline_bspline_kan", 118229)]:
    ps = {json.load(open(f))["n_params"] for f in glob.glob(os.path.join(d, "seed_*", "test_metrics.json"))}
    chk(os.path.basename(d) + " params", cp, list(ps)[0], 0)

# base config DS2 disclosure figure
m2, s2, _ = ms("results/wavkan_v2_curriculum", "macro_f1")
chk("base config DS2 Macro-F1", 0.323, m2, 0.0006)
chk("base config DS2 std", 0.018, s2, 0.0006)

# PTB-XL excluded number
px = json.load(open("results/ptbxl_zero_shot_metrics_multiseed_final.json"))
chk("PTB-XL excluded MF1", 0.200, px["macro_f1"]["mean"], 0.0006)
chk("PTB-XL excluded std", 0.005, px["macro_f1"]["std"], 0.0006)

print("\n" + "=" * 100)
print("CROSS-IMPLEMENTATION CHECK: this script's own statistics vs src/paired_stats.py")
print("=" * 100)
print("This script recomputes every statistic independently rather than reading it back")
print("from a summary. That independence is only useful if it agrees with the canonical")
print("implementation the project's own analysis scripts use, so we check that here:")
print("a disagreement means one of the two is wrong and no number below can be trusted.")
from src.paired_stats import paired_compare, holm, holm_family  # noqa: E402

_ref = load("results/ablation_no_rr_attn", "macro_f1")
_cmp = {
    "resnet1d": "results/baseline_resnet1d",
    "transformer": "results/baseline_transformer",
    "cnn_focal": "results/baseline_cnn_focal",
    "bspline_kan": "results/baseline_bspline_kan",
}
_res, _praw = {}, []
for _k, _d in _cmp.items():
    _r = paired_compare(_ref, load(_d, "macro_f1"), metric_name="macro_f1")
    _res[_k] = _r
    _praw.append(_r["p_value"])
_holm = holm(_praw)
_published = json.load(open("results/no_rr_attn_vs_baselines_stats.json"))["comparisons"]
for _i, (_k, _r) in enumerate(_res.items()):
    _pub = _published[_k]
    chk("xcheck " + _k + " p (paired_stats vs published)", _pub["p_value"], _r["p_value"], 1e-9)
    chk("xcheck " + _k + " d (paired_stats vs published)", _pub["cohens_d"], _r["cohens_d"], 1e-6)
    chk("xcheck " + _k + " holm (recomputed vs published)", _pub["holm_p"], _holm[_i], 1e-9)
    chk("xcheck " + _k + " n_paired is 20", 20, _r["n_paired"], 0)

# The RAC comparison, through the canonical helper, against this script's own
# independently computed values above.
_A = "results/ablation_no_rr_attn_no_augment"
_B = "results/ablation_no_rr_attn_no_curriculum_no_augment"
for _m, _d_expect, _ci_lo, _ci_hi in [("s_recall", 1.47, 0.058, 0.112),
                                      ("macro_f1", 0.52, 0.0013, 0.0233)]:
    _r = paired_compare(load(_A, _m), load(_B, _m), metric_name=_m)
    chk("xcheck RAC " + _m + " d via paired_stats", _d_expect, _r["cohens_d"], 0.006)
    chk("xcheck RAC " + _m + " CI low via paired_stats", _ci_lo, _r["ci_low"], 0.0006)
    chk("xcheck RAC " + _m + " CI high via paired_stats", _ci_hi, _r["ci_high"], 0.0006)
    chk("xcheck RAC " + _m + " two-sided", 1, 1 if _r["alternative"] == "two-sided" else 0, 0)


# ---------------------------------------------------------------------------
# Per-class operating points (Table: tab:perclass) and the S-class comparison.
# Recomputed here from results/per_class_metrics.json, which itself is derived
# from the stored per-seed prediction arrays by src/per_class_metrics.py.
# ---------------------------------------------------------------------------
print("\n" + "=" * 100)
print("TABLE: per-class precision / recall / F1")
print("=" * 100)

_pc_path = "results/per_class_metrics.json"
if os.path.exists(_pc_path):
    _pc = json.load(open(_pc_path))["arms"]

    # (a) PC-WavKAN per class, as printed in the manuscript
    for _c, _p, _r, _f in [("N", 0.971, 0.905, 0.936),
                           ("S", 0.175, 0.198, 0.180),
                           ("V", 0.528, 0.898, 0.663),
                           ("F", 0.005, 0.006, 0.005),
                           ("Q", 0.000, 0.000, 0.000)]:
        _g = _pc["PC-WavKAN"]["aggregate"][_c]
        chk("perclass %s precision" % _c, _p, _g["precision"]["mean"], 0.0006)
        chk("perclass %s recall" % _c, _r, _g["recall"]["mean"], 0.0006)
        chk("perclass %s f1" % _c, _f, _g["f1"]["mean"], 0.0006)

    # The per-class F1 values must average to the published Macro-F1: an
    # independent route to the headline number, so a per-class error cannot
    # pass unnoticed.
    for _name, _macro in [("PC-WavKAN", 0.357), ("ResNet1D", 0.351),
                          ("Transformer", 0.353), ("CNN+Focal", 0.362),
                          ("B-Spline KAN", 0.371)]:
        _f1s = [_pc[_name]["aggregate"][_c]["f1"]["mean"] for _c in "NSVFQ"]
        chk("perclass %s F1 -> Macro-F1" % _name, _macro,
            float(np.mean([0.0 if not np.isfinite(x) else x for x in _f1s])), 0.0006)

    # (b) S-class F1 across models, and the paired comparison against ours
    for _name, _sf1 in [("ResNet1D", 0.124), ("Transformer", 0.207),
                        ("CNN+Focal", 0.115), ("B-Spline KAN", 0.244)]:
        chk("S-F1 %s" % _name, _sf1,
            _pc[_name]["aggregate"]["S"]["f1"]["mean"], 0.0006)

    def _sf1_by_seed(_n):
        return {s: _pc[_n]["per_seed"][s]["S"]["f1"] for s in _pc[_n]["seeds"]}

    _comps = {b: paired_compare(_sf1_by_seed("PC-WavKAN"), _sf1_by_seed(b),
                                metric_name="s_f1")
              for b in ["ResNet1D", "Transformer", "CNN+Focal", "B-Spline KAN"]}
    _fam = holm_family(_comps)
    for _b, _d, _p in [("ResNet1D", 0.77, 0.004), ("Transformer", -0.35, 0.312),
                       ("CNN+Focal", 0.85, 0.002), ("B-Spline KAN", -0.66, 0.027)]:
        chk("S-F1 %s d" % _b, _d, _fam[_b]["cohens_d"], 0.006)
        chk("S-F1 %s holm p" % _b, _p, _fam[_b]["holm_p"], max(0.0006, _p * 0.06))
else:
    print("  results/per_class_metrics.json absent -- run src/per_class_metrics.py")

# ---------------------------------------------------------------------------
# Abstract: "outperformed all seven alternatives (Holm-adjusted p <= 0.0032)".
# This is a DIFFERENT family from Table tab:family_ablation (which compares each
# variant against the BASE). Registering it here because an earlier draft quoted
# the base-comparison p-value for this claim, which no check would have caught.
# ---------------------------------------------------------------------------
print("\n" + "=" * 100)
print("ABSTRACT: adopted configuration vs every alternative (validation)")
print("=" * 100)

_adopted = pv("results/ablation_no_rr_attn")
_alts = {"base": "results/wavkan_v2_curriculum",
         "no_pcwi": "results/ablation_no_pcwi",
         "no_pwam": "results/ablation_no_pwam",
         "no_rac": "results/wavkan_v2_baseline",
         "morlet": "results/ablation_wavelet_morlet",
         "dog": "results/ablation_wavelet_dog",
         "bspline": "results/ablation_wavelet_bspline"}
chk("abstract: number of alternatives", 7, len(_alts), 0)
_ac = {k: paired_compare(_adopted, pv(v), metric_name="val_macro_f1")
       for k, v in _alts.items()}
_af = holm_family(_ac)
_worst = max(r["holm_p"] for r in _af.values())
_all_better = all(r["mean_diff"] > 0 for r in _af.values())
chk("abstract: adopted beats every alternative", 1, 1 if _all_better else 0, 0)
chk("abstract: weakest Holm p <= 0.0032", 0.0032, _worst, 0.0004)

# ---------------------------------------------------------------------------
# Method: parameter counts asserted in prose.
# ---------------------------------------------------------------------------
print("\n" + "=" * 100)
print("METHOD: parameter counts")
print("=" * 100)
try:
    import torch.nn as _nn
    _gru = _nn.GRU(64, 32, batch_first=True, bidirectional=True)
    chk("BiGRU parameter count", 18816,
        sum(p.numel() for p in _gru.parameters()), 0)
    chk("BiGRU share of model (%)", 12.3,
        100.0 * 18816 / 153045, 0.06)
except Exception as _e:
    print("  (torch unavailable: %s)" % _e)

print("\n" + "=" * 100)
print("CLASS-DISTRIBUTION TABLE (tab:class_dist) vs PhysioNet-derived split counts")
print("=" * 100)
print("Parsed from the .tex itself, not typed here, and checked against")
print("configs/mitbih_split_counts.json (src/derive_split_counts.py). This table carried")
print("pre-H16 counts for weeks while every other number was verified, because nothing")
print("checked it (AUDIT_FINDINGS.md H49).")
import re as _re  # noqa: E402
_tex = open("Submission_Array/manuscript.tex", encoding="utf-8").read()
_tab = _tex.split(r"\label{tab:class_dist}")[1].split(r"\end{tabular}")[0]
_gt = json.load(open("configs/mitbih_split_counts.json", encoding="utf-8"))


def _num(s):
    return int(_re.sub(r"[^0-9]", "", s))


for _i, _cls in enumerate(["N", "S", "V", "F", "Q"]):
    _row = [ln for ln in _tab.split("\n") if ln.strip().startswith(r"\textbf{%s}" % _cls)]
    chk("class table row %s present" % _cls, 1, len(_row), 0)
    if not _row:
        continue
    _cells = [c.strip() for c in _row[0].split("&")]
    _tr, _va, _te, _tot = (_num(_cells[2]), _num(_cells[3]), _num(_cells[4]), _num(_cells[5]))
    chk("class table %s train" % _cls, _tr, _gt["counts"]["train"][str(_i)], 0)
    chk("class table %s val" % _cls, _va, _gt["counts"]["val"][str(_i)], 0)
    chk("class table %s test" % _cls, _te, _gt["counts"]["test"][str(_i)], 0)
    chk("class table %s row total" % _cls, _tot, _tr + _va + _te, 0)
_trow = [ln for ln in _tab.split("\n") if ln.strip().startswith(r"\textbf{Total}")][0]
_tc = [c.strip() for c in _trow.split("&")]
for _k, _split in ((2, "train"), (3, "val"), (4, "test")):
    chk("class table total %s" % _split, _num(_tc[_k]), _gt["totals"][_split], 0)
# the prose statement added with the correction
chk("prose: record 208 fusion beats", 372, _gt["per_record"]["208"][3], 0)
chk("prose: DS1 fusion beats", 414, _gt["counts"]["train"]["3"] + _gt["counts"]["val"]["3"], 0)
chk("prose: training fusion beats", 28, _gt["counts"]["train"]["3"], 0)
# and the table must describe the split the published arms were SELECTED on:
# the forensic reproduction of 2026-09-28 is the evidence; the prose record
# lists must name the same records the ground truth uses.
_prose = _tex.split(r"\subsection{Inter-Patient Split}")[1].split(r"\subsection")[0]
_LIST = r"((?:\d{3}, )*\d{3},? and \d{3})"
for _split, _pat in (("train", r"training uses the (\d+) records " + _LIST),
                     ("val", r"validation (?:uses )?the (\d+) records " + _LIST),
                     ("test", r"held out, (?:comprises )?the (\d+) records " + _LIST)):
    _m = _re.search(_pat, _prose)
    chk("split prose lists the %s partition" % _split, 1, 1 if _m else 0, 0)
    if not _m:
        continue
    _recs = sorted(_re.findall(r"\d{3}", _m.group(2)))
    chk("split prose %s record count as stated" % _split, int(_m.group(1)), len(_recs), 0)
    chk("split prose %s records == ground truth" % _split, 1,
        1 if _recs == sorted(_gt["records"][_split]) else 0, 0)

# ---------------------------------------------------------------------------
# Final-audit claims (2026-09-30): training procedure, validation evidence,
# protocol artefacts. AUDIT_FINDINGS.md H58-H67.
# ---------------------------------------------------------------------------
print("\n" + "=" * 100)
print("FINAL AUDIT: class-balanced phase, validation evidence, artefacts")
print("=" * 100)


def _best_rows(d):
    o = {}
    for pth in sorted(glob.glob(os.path.join(d, "seed_*", "training_history.json"))):
        h = json.load(open(pth))
        b_ = max(h, key=lambda e: e["val_macro_f1"])      # first maximum = train_pca's strict '>'
        o[os.path.basename(os.path.dirname(pth))] = (b_, h[-1])
    return o


# (1) every CBS checkpoint was selected in the class-balanced (WARMUP) phase
_cbs_arms = ["results/ablation_no_rr_attn", "results/ablation_no_rr_attn_no_augment",
             "results/wavkan_v2_curriculum"]
_rows = [r for d in _cbs_arms for r in _best_rows(d).values()]
chk("CBS runs counted (3 arms x 20)", 60, len(_rows), 0)
chk("CBS: best epoch in WARMUP phase (all)", 60, sum(1 for b_, _ in _rows if b_.get("phase") == "WARMUP"), 0)
chk("CBS: earliest best epoch", 2, min(b_["epoch"] for b_, _ in _rows), 0)
chk("CBS: latest best epoch", 25, max(b_["epoch"] for b_, _ in _rows), 0)
chk("CBS: earliest stop epoch", 17, min(l_["epoch"] for _, l_ in _rows), 0)
chk("CBS: latest stop epoch", 40, max(l_["epoch"] for _, l_ in _rows), 0)

# (2) Table tab:sampling, validation rows (CBS vs reweighting, adopted config, no augmentation)
_cb = _best_rows("results/ablation_no_rr_attn_no_augment")
_rw = _best_rows("results/ablation_no_rr_attn_no_curriculum_no_augment")
_sd = sorted(set(_cb) & set(_rw))
_vp = []
for _k, _cm_rw, _cs_rw, _cm_cb, _cs_cb, _cd, _cp in [
        ("val_macro_f1", 0.4285, 0.0125, 0.4195, 0.0098, -0.57, 0.027),
        ("val_s_recall", 0.2639, 0.0510, 0.2391, 0.0425, -0.53, 0.025)]:
    a_ = np.array([_cb[k][0][_k] for k in _sd]); b_ = np.array([_rw[k][0][_k] for k in _sd])
    d_ = a_ - b_
    pr = wilcoxon(a_, b_)[1]
    _vp.append(pr)
    chk("sampling val " + _k + " reweighting mean", _cm_rw, b_.mean(), 0.00006)
    chk("sampling val " + _k + " reweighting std", _cs_rw, b_.std(ddof=1), 0.00006)
    chk("sampling val " + _k + " CBS mean", _cm_cb, a_.mean(), 0.00006)
    chk("sampling val " + _k + " CBS std", _cs_cb, a_.std(ddof=1), 0.00006)
    chk("sampling val " + _k + " d_z", _cd, d_.mean() / d_.std(ddof=1), 0.006)
    chk("sampling val " + _k + " raw p", _cp, pr, 0.0006)
_hv = holm(_vp)
chk("sampling val: Holm over 2 metrics ~0.05 (not < 0.05)", 0.05, min(_hv), 0.002)
chk("sampling val: neither significant after Holm", 1, 1 if min(_hv) >= 0.05 else 0, 0)

# (3) augmentation is supported on validation (CBS both arms)
_ad = _best_rows("results/ablation_no_rr_attn")
_sd2 = sorted(set(_ad) & set(_cb))
a_ = np.array([_ad[k][0]["val_macro_f1"] for k in _sd2]); b_ = np.array([_cb[k][0]["val_macro_f1"] for k in _sd2])
d_ = a_ - b_
chk("aug val: MF1 gain", 0.020, d_.mean(), 0.0006)
chk("aug val: d_z", 1.16, d_.mean() / d_.std(ddof=1), 0.006)
chk("aug val: raw p", 1.7e-4, wilcoxon(a_, b_)[1], 0.05e-4)

# (4) seed-1001 exclusion: unadjusted CI vs B-Spline KAN narrowly excludes zero
_r1 = {k: v for k, v in load("results/ablation_no_rr_attn", "macro_f1").items() if k != "seed_1001"}
_b1 = load("results/baseline_bspline_kan", "macro_f1")
_s1 = sorted(set(_r1) & set(_b1))
_d1 = np.array([_r1[k] - _b1[k] for k in _s1])
_h1 = tdist.ppf(0.975, len(_d1) - 1) * _d1.std(ddof=1) / np.sqrt(len(_d1))
chk("seed-1001-excluded n", 19, len(_d1), 0)
chk("seed-1001-excluded CI low", -0.031, _d1.mean() - _h1, 0.0006)
chk("seed-1001-excluded CI high", -0.0002, _d1.mean() + _h1, 0.00006)

# (5) architecture facts stated in Methods
from src.pwam import PWaveAttentionModule  # noqa: E402
_pw = PWaveAttentionModule(main_dim=64, p_hidden=32, attn_heads=4, dropout=0.2)
chk("side branch parameters", 34816, sum(q.numel() for q in _pw.parameters()), 0)
import src.pwam as _pwm  # noqa: E402
chk("side branch reads samples 80..160", 1, 1 if (_pwm.P_START, _pwm.P_END) == (80, 160) else 0, 0)
chk("side branch start = -28 ms (R at sample 90)", -28, round((80 - 90) / 360 * 1000), 1)
chk("side branch end = +194 ms", 194, round((160 - 90) / 360 * 1000), 1)
_ra = WavKAN_v2(use_rr_attn=True).rr_branch.self_attn
chk("alternative rhythm encoder: two attention heads", 2, _ra.num_heads, 0)
chk("record 208 share of DS1 fusion beats (~90%)", 0.90, 372 / 414, 0.005)

# (6) protocol artefacts (src/protocol_artefacts_report.py, from the annotation files)
_pa = json.load(open("results/rr_sensitivity/protocol_artefacts.json"))["test"]
chk("DS2 beats in artefact report", 49684, _pa["beats"], 0)
chk("DS2 beats with any RR element changed", 4578, _pa["any_rr_element_changed"], 0)
chk("DS2 any-RR changed pct (9.2)", 9.2, _pa["any_rr_element_changed_pct"], 0.05)
chk("DS2 beats with RR_0 changed", 1433, _pa["rr0_changed"], 0)
chk("DS2 RR_0 changed pct (2.9)", 2.9, _pa["rr0_changed_pct"], 0.05)
chk("DS2 next beat inside window pct (48.8)", 48.8, _pa["next_beat_inside_window_pct"], 0.05)

# (7) four-class N/S/V/F Macro-F1 (exploratory), from the per-seed confusion
# matrices of the published-RR re-evaluation, which is also the beat-for-beat
# reproduction check of every checkpoint's saved DS2 predictions.
_ps_path = "results/rr_sensitivity/published_rr"
if os.path.exists(os.path.join(_ps_path, "per_seed.json")):
    _pr = json.load(open(os.path.join(_ps_path, "report.json")))
    _mm = _pr["published_prediction_mismatches"]
    chk("reproduction: checkpoints with any differing DS2 prediction", 12, len(_mm), 0)
    chk("reproduction: all differing checkpoints are CNN+Focal", 1, 1 if all(m[0] == "CNN+Focal" for m in _mm) else 0, 0)
    chk("reproduction: at most 4 of 49,684 beats differ", 4, max(m[2] for m in _mm), 0)
    chk("reproduction: at least 1 beat differs where listed", 1, min(m[2] for m in _mm), 0)
    _pss = json.load(open(os.path.join(_ps_path, "per_seed.json")))

    def _f1_nsvf(cm):
        cm = np.asarray(cm, float)
        tp = np.diag(cm)
        pr_ = np.divide(tp, cm.sum(0), out=np.zeros_like(tp), where=cm.sum(0) > 0)
        rc_ = np.divide(tp, cm.sum(1), out=np.zeros_like(tp), where=cm.sum(1) > 0)
        f_ = np.divide(2 * pr_ * rc_, pr_ + rc_, out=np.zeros_like(tp), where=(pr_ + rc_) > 0)
        return f_[:4].mean(), f_.mean()

    _nsvf = {m: {s_: _f1_nsvf(v["confusion_matrix"]) for s_, v in d.items()} for m, d in _pss.items()}
    for _m, _cm_, _cs_, _c5 in [("PC-WavKAN", 0.446, 0.023, 0.357), ("ResNet1D", 0.439, 0.023, 0.351),
                                ("Transformer", 0.441, 0.050, 0.353), ("CNN+Focal", 0.453, 0.019, 0.362),
                                ("B-Spline KAN", 0.463, 0.030, 0.371)]:
        _v = np.array([x[0] for x in _nsvf[_m].values()])
        _v5 = np.array([x[1] for x in _nsvf[_m].values()])
        chk("4-class MF1 %s mean" % _m, _cm_, _v.mean(), 0.0006)
        chk("4-class MF1 %s std" % _m, _cs_, _v.std(ddof=1), 0.0006)
        chk("re-inferred 5-class MF1 %s == published" % _m, _c5, _v5.mean(), 0.0006)
    _fam4 = holm_family({_m: paired_compare({k: v[0] for k, v in _nsvf["PC-WavKAN"].items()},
                                            {k: v[0] for k, v in _nsvf[_m].items()}, "mf1_nsvf")
                         for _m in ["ResNet1D", "Transformer", "CNN+Focal", "B-Spline KAN"]})
    chk("4-class: min Holm p (>= 0.36)", 0.36, min(r["holm_p"] for r in _fam4.values()), 0.005)
else:
    chk("results/rr_sensitivity/published_rr present", 1, 0, 0)

# (8) RR artefact: per-class RR_0 rates and direction (protocol_artefacts.json)
_pa_c = _pa["per_class"]
chk("RR_0 changed, V beats pct (6.4)", 6.4, _pa_c["V"]["rr0_pct"], 0.05)
chk("RR_0 changed, N beats pct (2.7)", 2.7, _pa_c["N"]["rr0_pct"], 0.05)
chk("RR_0 changed, S beats pct (1.3)", 1.3, _pa_c["S"]["rr0_pct"], 0.05)
chk("RR_0 changed values almost always shorter (>=99%)", 1,
    1 if _pa["rr0_changed_and_shorter"] >= 0.99 * _pa["rr0_changed"] else 0, 0)

# (9) A2: beat-to-beat intervals at inference, same checkpoints (Colab GPU run, same backend
# for both passes; results/rr_sensitivity/{published_rr,beats_only_rr})
_A2p = json.load(open("results/rr_sensitivity/published_rr/per_seed.json"))
_A2b = json.load(open("results/rr_sensitivity/beats_only_rr/per_seed.json"))
_incs = {}
for _m in _A2p:
    _d = np.array([_A2b[_m][k]["macro_f1"] - _A2p[_m][k]["macro_f1"] for k in _A2p[_m]])
    _incs[_m] = _d.mean()
    chk("A2 %s Macro-F1 rises in all 20 seeds" % _m, 20, int((_d > 0).sum()), 0)
chk("A2 smallest mean model increase (~+0.002)", 0.002, min(_incs.values()), 0.0006)
chk("A2 largest mean model increase (~+0.005)", 0.005, max(_incs.values()), 0.0006)
chk("A2 PC-WavKAN published-RR Macro-F1 (0.357)", 0.357,
    np.mean([v["macro_f1"] for v in _A2p["PC-WavKAN"].values()]), 0.0006)
chk("A2 PC-WavKAN beat-to-beat Macro-F1 (0.360)", 0.360,
    np.mean([v["macro_f1"] for v in _A2b["PC-WavKAN"].values()]), 0.0006)
chk("A2 PC-WavKAN V-recall change (-0.001)", -0.001,
    np.mean([_A2b["PC-WavKAN"][k]["v_recall"] - _A2p["PC-WavKAN"][k]["v_recall"] for k in _A2p["PC-WavKAN"]]), 0.0005)
_fb = holm_family({_m: paired_compare({k: v["macro_f1"] for k, v in _A2b["PC-WavKAN"].items()},
                                      {k: v["macro_f1"] for k, v in _A2b[_m].items()}, "macro_f1")
                   for _m in ["ResNet1D", "Transformer", "CNN+Focal", "B-Spline KAN"]})
chk("A2 beat-to-beat primary: min Holm p >= 0.38", 1, 1 if min(r["holm_p"] for r in _fb.values()) >= 0.38 else 0, 0)
chk("A2 beat-to-beat primary: every CI includes zero", 1,
    1 if all(r["ci_low"] <= 0 <= r["ci_high"] for r in _fb.values()) else 0, 0)

# (10) A1: matched-pipeline external evaluation (Table tab:multidataset). Statistics are
# recomputed here from the per-seed confusion matrices and cross-checked with the report.
_fpc = json.load(open("results/colab_fingerprint_check.json"))
for _ds in ("incart_matched", "svdb_matched", "mitdb_test_published_rr", "mitdb_test_beats_only_rr"):
    for _k in ("n", "y_sha256", "rr_sha256", "X_round4_sha256"):
        chk("inputs identical to local extraction: %s %s" % (_ds, _k), 1, 1 if _fpc[_ds][_k] else 0, 0)
_TAB = {  # model: (MF1, std, d_z, holm_p or '<0.001', V, S)
    "incart": {"PC-WavKAN": (0.390, 0.017, None, None, 0.832, 0.671),
               "ResNet1D": (0.431, 0.017, -1.80, "<0.001", 0.849, 0.760),
               "Transformer": (0.398, 0.032, -0.24, 0.245, 0.852, 0.817),
               "CNN+Focal": (0.431, 0.020, -1.53, "<0.001", 0.807, 0.560),
               "B-Spline KAN": (0.367, 0.017, 0.83, 0.003, 0.820, 0.606)},
    "svdb": {"PC-WavKAN": (0.275, 0.012, None, None, 0.788, 0.126),
             "ResNet1D": (0.350, 0.026, -2.49, "<0.001", 0.832, 0.256),
             "Transformer": (0.316, 0.031, -1.36, "<0.001", 0.820, 0.305),
             "CNN+Focal": (0.354, 0.013, -4.86, "<0.001", 0.769, 0.179),
             "B-Spline KAN": (0.267, 0.016, 0.33, 0.231, 0.791, 0.125)}}
for _ds, _rows in _TAB.items():
    _ps_ = json.load(open("results/external_matched/%s/per_seed.json" % _ds))
    _rep = json.load(open("results/external_matched/%s/report.json" % _ds))
    chk("%s beats" % _ds, {"incart": 175785, "svdb": 184486}[_ds], _rep["n_beats"], 0)
    _fam_e = holm_family({_m: paired_compare({k: v["macro_f1"] for k, v in _ps_["PC-WavKAN"].items()},
                                             {k: v["macro_f1"] for k, v in _ps_[_m].items()}, "macro_f1")
                          for _m in ["ResNet1D", "Transformer", "CNN+Focal", "B-Spline KAN"]})
    for _m, (_mf, _sd, _dz, _hp, _vr, _sr) in _rows.items():
        _v = np.array([x["macro_f1"] for x in _ps_[_m].values()])
        chk("%s %s Macro-F1" % (_ds, _m), _mf, _v.mean(), 0.0006)
        chk("%s %s Macro-F1 std" % (_ds, _m), _sd, _v.std(ddof=1), 0.0006)
        chk("%s %s V-recall" % (_ds, _m), _vr, np.mean([x["v_recall"] for x in _ps_[_m].values()]), 0.0006)
        chk("%s %s S-recall" % (_ds, _m), _sr, np.mean([x["s_recall"] for x in _ps_[_m].values()]), 0.0006)
        if _dz is None:
            continue
        _r = _fam_e[_m]
        chk("%s %s d_z" % (_ds, _m), _dz, _r["cohens_d"], 0.006)
        if _hp == "<0.001":
            chk("%s %s Holm p < 0.001" % (_ds, _m), 1, 1 if _r["holm_p"] < 0.001 else 0, 0)
        else:
            chk("%s %s Holm p" % (_ds, _m), _hp, _r["holm_p"], max(0.0006, _hp * 0.02))
        chk("%s %s Holm p matches stored report" % (_ds, _m),
            _rep["macro_f1_vs_reference_holm_family"][_m]["holm_p"], _r["holm_p"], 1e-9)
_sv = json.load(open("results/external_matched/svdb/per_seed.json"))
_svm = {m: (np.mean([x["s_to_v_rate"] for x in d.values()]), np.mean([x["v_pred_share"] for x in d.values()]))
        for m, d in _sv.items()}
chk("SVDB S->V PC-WavKAN (0.431)", 0.431, _svm["PC-WavKAN"][0], 0.0006)
chk("SVDB S->V B-Spline KAN (0.451)", 0.451, _svm["B-Spline KAN"][0], 0.0006)
chk("SVDB S->V others min (0.231)", 0.231, min(_svm[m][0] for m in ("ResNet1D", "Transformer", "CNN+Focal")), 0.0006)
chk("SVDB S->V others max (0.395)", 0.395, max(_svm[m][0] for m in ("ResNet1D", "Transformer", "CNN+Focal")), 0.0006)
chk("SVDB V share PC-WavKAN (18.7%)", 0.187, _svm["PC-WavKAN"][1], 0.0006)
chk("SVDB V share B-Spline KAN (20.7%)", 0.207, _svm["B-Spline KAN"][1], 0.0006)
chk("SVDB V share others min (8.4%)", 0.084, min(_svm[m][1] for m in ("ResNet1D", "Transformer", "CNN+Focal")), 0.0006)
chk("SVDB V share others max (17.6%)", 0.176, max(_svm[m][1] for m in ("ResNet1D", "Transformer", "CNN+Focal")), 0.0006)
# the superseded, unmatched INCART evaluation quoted as a preprocessing-sensitivity example
_old = json.load(open("results/multidataset_final_stats.json"))["INCART"]
chk("unmatched INCART PC-WavKAN (0.374)", 0.374, _old["wavkan_v2_final_macro_f1"]["mean"], 0.0006)
chk("unmatched INCART baselines min (0.316)", 0.316, min(v["mean_b"] for v in _old["comparisons"].values()), 0.0006)
chk("unmatched INCART baselines max (0.365)", 0.365, max(v["mean_b"] for v in _old["comparisons"].values()), 0.0006)

print("\n" + "=" * 100)
print("RESULT:  %d verified,  %d MISMATCHED" % (len(OK), len(BAD)))
if BAD:
    print("\nMISMATCHES REQUIRING CORRECTION:")
    for l, c, a_ in BAD:
        print("  %-56s manuscript=%s  real=%s" % (l, c, a_))
print("=" * 100)

sys.exit(1 if BAD else 0)
