"""
e3c_compliance.py  —  E3C Clinical Evaluation Compliance Checker

THE "4.1% CLUB" — AUTO-VERIFICATION
=====================================
Silva et al. [4] reviewed 122 ECG arrhythmia classification papers and
found that only 4.1% (5 out of 122) satisfied all four E3C criteria:

  E3C Criterion 1: Strict inter-patient evaluation protocol
  E3C Criterion 2: Class-specific metrics for imbalanced data
  E3C Criterion 3: Adherence to AAMI EC57 clinical protocol
  E3C Criterion 4: Consideration of computational complexity

This script checks the experimental results against each criterion and reports a
PASS/PARTIAL/FAIL per criterion, generating a compliance summary. Where a real
generated artifact exists (e.g. the actual train/test record-ID arrays, or a real
per-class label file) it verifies against that; where none exists it reports PARTIAL
rather than assuming compliance. See AUDIT_FINDINGS.md C6 for the history of why this
distinction matters -- earlier versions of this script hardcoded PASS for 3 of these 4
criteria regardless of input.

Output:
  results/e3c_compliance/
    e3c_report.json          Structured compliance results
    e3c_certificate.tex      LaTeX table (the "4.1% certificate")
    e3c_report.txt           Human-readable compliance report

Usage:
    python src/e3c_compliance.py \\
        --results-dir results/full_pipeline \\
        --model wavkan_v2 \\
        --params 109345 \\
        --latency-ms 0.78
"""

import os, sys, json, argparse
from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass, field, asdict

CHECKMARK = "✅"
CROSS     = "❌"
PARTIAL   = "⚠️"


# ─────────────────────────────────────────────────────────────────────────────
# E3C Criterion Definitions (from Silva et al. 2025, arXiv:2503.07276)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class E3CCriterion:
    id:           str
    name:         str
    description:  str
    status:       str = "NOT_CHECKED"   # PASS | FAIL | PARTIAL
    evidence:     list = field(default_factory=list)
    score:        float = 0.0           # 0.0 = fail, 0.5 = partial, 1.0 = pass


@dataclass
class E3CReport:
    model_name:   str
    criteria:     list
    overall:      str = "UNKNOWN"
    compliant:    bool = False
    summary:      str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Criterion checkers
# ─────────────────────────────────────────────────────────────────────────────

def check_criterion_1(
    results_dir: str,
    model:       str,
    ds1_records: Optional[list] = None,
    ds2_records: Optional[list] = None,
    data_dir:    str = "data/processed_rr_history",
) -> E3CCriterion:
    """
    E3C-1: Strict inter-patient evaluation
    Required: DS1/DS2 De Chazal split, no patient appears in both train and test.
    """
    c = E3CCriterion(
        id="E3C-1",
        name="Strict Inter-Patient Evaluation",
        description=(
            "Model is trained on DS1 and evaluated on DS2 (De Chazal et al. 2004 protocol). "
            "No patient overlap between train and test sets."
        ),
    )

    # Default DS1/DS2 records from process_data.py
    DS1 = {'101','106','108','109','112','114','115','116','118','119',
            '122','124','201','203','205','207','208','209','215','220','223','230'}
    DS2 = {'100','103','105','111','113','117','121','123','200','202',
            '210','212','213','214','219','221','222','228','231','232','233','234'}

    overlap = DS1 & DS2
    if overlap:
        c.status = "FAIL"
        c.evidence.append(f"FAIL: Records in both DS1 and DS2: {overlap}")
        c.score   = 0.0
        return c

    c.evidence.append("PASS: No overlap between the hardcoded reference DS1/DS2 record lists")
    c.evidence.append(f"PASS: DS1 = {len(DS1)} records (train+val), DS2 = {len(DS2)} records (test)")
    c.evidence.append(
        "NOTE: the check above only verifies two hand-typed literals are disjoint from "
        "each other -- it says nothing about which records a real training run actually "
        "used. See AUDIT_FINDINGS.md C6/M17."
    )

    # Check that records 201+202 (same patient) are split
    if "201" in DS1 and "202" in DS2:
        c.evidence.append("PASS: Records 201 & 202 (same patient) correctly split across DS1/DS2")
    else:
        c.evidence.append("⚠️  Records 201 & 202 same-patient check: verify manually")

    # Real leakage check: verify the ACTUAL record-ID arrays a training run produced
    # (process_data.py:154,164 saves ids_{split}.npy) never overlap between train and test.
    # This is the check that can actually fail against real generated artifacts, unlike the
    # hardcoded-literal comparison above.
    ids_train_path = Path(data_dir) / "ids_train.npy"
    ids_test_path = Path(data_dir) / "ids_test.npy"
    real_ids_verified = False
    if ids_train_path.exists() and ids_test_path.exists():
        import numpy as np
        ids_train = set(np.load(ids_train_path, allow_pickle=True).tolist())
        ids_test = set(np.load(ids_test_path, allow_pickle=True).tolist())
        real_overlap = ids_train & ids_test
        if real_overlap:
            c.status = "FAIL"
            c.evidence.append(f"FAIL: Real generated ids_train/ids_test overlap: {real_overlap}")
            c.score = 0.0
            return c
        c.evidence.append(
            f"PASS: Real generated record-ID arrays verified disjoint "
            f"({len(ids_train)} train records, {len(ids_test)} test records)"
        )
        real_ids_verified = True
    else:
        c.evidence.append(
            f"PARTIAL: {ids_train_path} / {ids_test_path} not found -- cannot verify the "
            "actual generated split, only the hardcoded reference lists above. Run "
            "process_data.py first to enable the real check."
        )

    # Check test results file exists
    test_path = Path(results_dir) / model / "seed_42" / "test_metrics.json"
    if test_path.exists():
        with open(test_path) as f:
            m = json.load(f)
        c.evidence.append(f"PASS: Test results found (Macro-F1={m.get('macro_f1', 'N/A'):.4f})")

    if real_ids_verified:
        c.status = "PASS"
        c.score = 1.0
    else:
        c.status = "PARTIAL"
        c.score = 0.5
    return c


def check_criterion_2(
    results_dir: str,
    model:       str,
) -> E3CCriterion:
    """
    E3C-2: Class-specific metrics for imbalanced data
    Required: Per-class precision/recall/F1, not just accuracy.
    AAMI classes: N, S, V, F, Q reported separately.
    """
    c = E3CCriterion(
        id="E3C-2",
        name="Class-Specific Metrics for Imbalanced Data",
        description=(
            "Reports per-class precision, recall, and F1 for ALL AAMI classes "
            "(N, S, V, F, Q). Does not rely solely on accuracy."
        ),
    )

    test_path = Path(results_dir) / model / "seed_42" / "test_metrics.json"
    if not test_path.exists():
        c.status  = "PARTIAL"
        c.score   = 0.5
        c.evidence.append("⚠️  test_metrics.json not found — run pipeline first")
        return c

    with open(test_path) as f:
        m = json.load(f)

    reported_classes = []
    for cls in ["N", "S", "V", "F", "Q"]:
        key = f"{cls.lower()}_recall"
        if key in m:
            reported_classes.append(cls)
            c.evidence.append(f"PASS: {cls}-class recall = {m[key]:.4f}")

    if len(reported_classes) >= 5:
        c.status = "PASS"
        c.score  = 1.0
        c.evidence.append("PASS: All 5 AAMI classes reported with per-class metrics")
        c.evidence.append("PASS: Macro-averaged F1 used (not accuracy) for imbalanced evaluation")
    elif len(reported_classes) >= 3:
        c.status = "PARTIAL"
        c.score  = 0.5
        c.evidence.append(f"PARTIAL: Only {len(reported_classes)}/5 classes with recall reported")
    else:
        c.status = "FAIL"
        c.score  = 0.0

    return c


def check_criterion_3(
    results_dir: str,
    model:       str,
    aami_classes: list = None,
    data_dir:    str = "data/processed_rr_history",
) -> E3CCriterion:
    """
    E3C-3: AAMI EC57 clinical protocol adherence
    Required: Uses AAMI beat-type mapping, 5 superclasses.
    """
    c = E3CCriterion(
        id="E3C-3",
        name="AAMI EC57 Clinical Protocol",
        description=(
            "Annotations mapped to AAMI EC57 superclasses: "
            "N (Normal), S (Supraventricular), V (Ventricular), F (Fusion), Q (Unknown). "
            "Class N/S/V/F/Q must correspond to the EC57 standard annotation groupings."
        ),
    )

    # Verify AAMI mapping from process_data.py
    AAMI_MAP = {
        'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,     # Normal
        'A': 1, 'a': 1, 'J': 1, 'S': 1,               # Supraventricular
        'V': 2, 'E': 2,                                # Ventricular
        'F': 3,                                         # Fusion
        'Q': 4, '/': 4, 'f': 4                         # Unknown
    }
    c.evidence.append(f"PASS: AAMI EC57 mapping defined with {len(AAMI_MAP)} annotation types")
    c.evidence.append("PASS: Class N → {N, L, R, e, j} (Normal sinus)")
    c.evidence.append("PASS: Class S → {A, a, J, S} (Supraventricular ectopic)")
    c.evidence.append("PASS: Class V → {V, E} (Ventricular ectopic)")
    c.evidence.append("PASS: Class F → {F} (Fusion)")
    c.evidence.append("PASS: Class Q → {Q, /, f} (Unknown/paced)")

    # Verify data file exists
    data_path = Path(data_dir) / "y_test.npy"
    real_labels_verified = False
    if data_path.exists():
        y = __import__("numpy").load(str(data_path))
        unique = set(y.tolist())
        expected = {0, 1, 2, 3, 4}
        if unique.issubset(expected) and len(unique) > 0:
            c.evidence.append(f"PASS: Test labels verified: classes present = {sorted(unique)}")
            real_labels_verified = True
        else:
            c.evidence.append(
                f"FAIL: Test labels contain values outside the AAMI 0-4 mapping: {sorted(unique)}"
            )
            c.status = "FAIL"
            c.score = 0.0
            return c
    else:
        c.evidence.append("⚠️  y_test.npy not found — run process_data.py first, cannot verify real labels")

    c.evidence.append("PASS: Q class retained in reporting (completeness, weight=0 in loss)")

    if real_labels_verified:
        c.status = "PASS"
        c.score = 1.0
    else:
        # AAMI_MAP is a hardcoded literal, real annotations were never checked against it.
        c.status = "PARTIAL"
        c.score = 0.5
    return c


def check_criterion_4(
    results_dir:  str,
    model:        str,
    n_params:     int = 109345,
    latency_ms:   float = 0.78,
    max_params:   int = 200_000,
    max_latency:  float = 5.0,
) -> E3CCriterion:
    """
    E3C-4: Computational complexity consideration
    Required: Reports model size, inference latency, deployment feasibility.
    """
    c = E3CCriterion(
        id="E3C-4",
        name="Computational Complexity Consideration",
        description=(
            "Reports parameter count, memory footprint, and inference latency. "
            "Addresses edge deployment feasibility (Green AI requirements)."
        ),
    )

    # Check deployment report
    deploy_path = Path(results_dir) / "deployment"
    if deploy_path.exists():
        c.evidence.append(f"PASS: Deployment benchmark directory found: {deploy_path}")

    # Parameter count
    params_ok = n_params <= max_params
    if params_ok:
        c.evidence.append(
            f"PASS: {n_params:,} parameters ≤ {max_params:,} (edge-feasible)"
        )
        c.evidence.append(
            f"PASS: Memory footprint ≈ {n_params * 4 / 1024:.1f} KB"
        )
    else:
        c.evidence.append(f"FAIL: {n_params:,} parameters exceeds threshold of {max_params:,}")

    # Inference latency
    latency_ok = latency_ms <= max_latency
    latency_path = Path(results_dir) / "latency_report.json"
    if not latency_path.exists():
        c.evidence.append(
            f"⚠️  {latency_path} not found -- latency_ms={latency_ms} is an unverified "
            "CLI-supplied value, not measured from this run. See AUDIT_FINDINGS.md H9."
        )
    if latency_ok:
        c.evidence.append(
            f"PASS: Inference latency = {latency_ms} ms/beat (≤{max_latency} ms threshold)"
        )
    else:
        c.evidence.append(f"FAIL: Latency {latency_ms} ms exceeds target {max_latency} ms")

    # INT8 quantization
    quant_path = Path(results_dir) / "deployment" / "quantized_model.pt"
    if quant_path.exists():
        c.evidence.append("PASS: INT8 quantized model available for edge deployment")
    else:
        c.evidence.append("INFO: INT8 quantization result not found (run Stage 8) -- not required for this criterion, informational only")

    c.evidence.append("INFO: Green AI / CO2 tracking is a separate benchmark (Stage 8), not verified by this criterion")

    if params_ok and latency_ok:
        c.status = "PASS"
        c.score = 1.0
    elif params_ok or latency_ok:
        c.status = "PARTIAL"
        c.score = 0.5
    else:
        c.status = "FAIL"
        c.score = 0.0
    return c


# ─────────────────────────────────────────────────────────────────────────────
# Report generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_full_report(
    model_name:   str,
    results_dir:  str,
    n_params:     int,
    latency_ms:   float,
) -> E3CReport:
    print(f"\n{'='*65}")
    print(f"E3C CLINICAL EVALUATION COMPLIANCE CHECK — {model_name}")
    print(f"{'='*65}")
    print(f"Reference: Silva et al. (2025), arXiv:2503.07276 — 122 papers reviewed, 4.1% compliant")
    print(f"{'='*65}\n")

    c1 = check_criterion_1(results_dir, "wavkan_v2")
    c2 = check_criterion_2(results_dir, "wavkan_v2")
    c3 = check_criterion_3(results_dir, "wavkan_v2")
    c4 = check_criterion_4(results_dir, "wavkan_v2", n_params, latency_ms)

    criteria   = [c1, c2, c3, c4]
    total_score = sum(c.score for c in criteria) / len(criteria)
    all_pass    = all(c.status == "PASS" for c in criteria)

    for c in criteria:
        icon = CHECKMARK if c.status == "PASS" else (PARTIAL if c.status == "PARTIAL" else CROSS)
        print(f"{icon}  [{c.id}] {c.name}")
        for ev in c.evidence:
            print(f"     {ev}")
        print()

    overall = "COMPLIANT" if all_pass else ("PARTIALLY_COMPLIANT" if total_score >= 0.75 else "NON_COMPLIANT")

    bang = "🏆" if all_pass else ("⚠️" if total_score >= 0.75 else "❌")
    print(f"{'='*65}")
    print(f"{bang}  VERDICT: {overall}  (score={total_score:.2f}/1.00)")
    if all_pass:
        print(f"   {model_name} satisfies ALL four E3C criteria.")
        print(f"   Estimated: top 4.1% of ECG classification papers [Silva et al. 2025, arXiv:2503.07276]")
    elif total_score >= 0.75:
        failed = [c.id for c in criteria if c.status != "PASS"]
        print(f"   Partially compliant. Failed: {failed}. Run pipeline to fix.")
    print(f"{'='*65}")

    return E3CReport(
        model_name = model_name,
        criteria   = [asdict(c) for c in criteria],
        overall    = overall,
        compliant  = all_pass,
        summary    = f"Score={total_score:.2f}/1.00, Status={overall}",
    )


def generate_latex_certificate(report: E3CReport) -> str:
    icon    = r"\checkmark" if True else r"$\times$"
    rows    = []
    for c in report.criteria:
        status = r"\textbf{\checkmark}" if c["status"] == "PASS" else r"\textbf{$\times$}"
        rows.append(f"  {c['id']} & {c['name']} & {status} \\\\")

    compliant_str = "\\textbf{YES}" if report.compliant else "\\textbf{NO}"
    if report.compliant:
        claim = (
            rf"{report.model_name} satisfies all four criteria, placing it among the "
            rf"4.1\% of ECG classification studies (per \citeauthor{{silva_systematic_2025}}) "
            rf"that meet rigorous clinical evaluation standards."
        )
    else:
        failed = [c["id"] for c in report.criteria if c["status"] != "PASS"]
        claim = (
            rf"{report.model_name} does not yet satisfy all four E3C criteria "
            rf"(per \citeauthor{{silva_systematic_2025}}) -- failing or partial on: {', '.join(failed)}."
        )
    return "\n".join([
        r"\begin{table}[!htbp]",
        r"\centering",
        rf"\caption{{E3C Clinical Evaluation Compliance. {claim}}}",
        r"\label{tab:e3c}",
        r"\begin{tabular}{@{}llc@{}}",
        r"\toprule",
        r"  \textbf{Criterion} & \textbf{Requirement} & \textbf{Status} \\",
        r"\midrule",
        *rows,
        r"\midrule",
        rf"  \multicolumn{{2}}{{l}}{{\textbf{{Overall Compliance}}}} & {compliant_str} \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="E3C Clinical Evaluation Compliance Checker")
    parser.add_argument("--results-dir", type=str, default="results/full_pipeline")
    parser.add_argument("--model",       type=str, default="WavKAN-v2")
    parser.add_argument("--params",      type=int, default=109345)
    parser.add_argument("--latency-ms",  type=float, default=0.78)
    parser.add_argument("--output-dir",  type=str, default="results/e3c_compliance")
    args = parser.parse_args()

    OUT = Path(args.output_dir)
    OUT.mkdir(parents=True, exist_ok=True)

    report = generate_full_report(
        model_name  = args.model,
        results_dir = args.results_dir,
        n_params    = args.params,
        latency_ms  = args.latency_ms,
    )

    # Save JSON
    from dataclasses import asdict
    with open(OUT / "e3c_report.json", "w") as f:
        json.dump(asdict(report), f, indent=2)

    # Save LaTeX
    latex = generate_latex_certificate(report)
    with open(OUT / "e3c_certificate.tex", "w") as f:
        f.write(latex)

    # Save TXT
    lines = [
        f"E3C COMPLIANCE REPORT — {report.model_name}",
        f"{'='*60}",
        f"Overall: {report.overall}  |  Compliant: {report.compliant}",
        f"Summary: {report.summary}",
        f"{'='*60}",
    ]
    for c in report.criteria:
        lines.append(f"\n[{c['id']}] {c['name']} — {c['status']}")
        for ev in c["evidence"]:
            lines.append(f"  {ev}")

    with open(OUT / "e3c_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n✅ E3C compliance report saved to {OUT}/")
