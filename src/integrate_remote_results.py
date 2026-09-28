#!/usr/bin/env python3
"""
integrate_remote_results.py -- deterministic local integration of a remote GPU run.

    python src/integrate_remote_results.py --dir results/final_component_ablation

Run this after copying a remote results directory back into the local repository.
It does five things, in order, and stops at the first one that fails:

  1. VERIFY ARTIFACTS   every file the run should have produced is present
  2. VERIFY SEEDS       every requested seed completed, none is missing, no NaNs,
                        parameter counts match the config, checkpoints exist
  3. VERIFY PROVENANCE  the config snapshot matches the config in the repo, the
                        data integrity checks passed remotely, and the git
                        revision is recorded
  4. RECOMPUTE STATS    every paired statistic is recomputed here, locally, from
                        the per-seed files -- then diffed against the remote
                        results.json. A transport error, a package-version skew
                        or a tampered summary shows up as a mismatch instead of
                        being trusted.
  5. REPORT             prints the exact numbers to put in the manuscript, and
                        the exact claims each one bears on

It deliberately does NOT edit the manuscript. Numbers reach the paper only after
the diff in step 4 is clean, and then via src/verify_manuscript_numbers.py, which
re-derives them from these same per-seed files. That keeps the manuscript
downstream of machine-readable results rather than of hand transcription.

Exit codes: 0 clean, 2 artifacts/seeds bad, 3 statistics disagree.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.paired_stats import describe, holm_family, paired_compare  # noqa: E402

TOL = 1e-9          # statistics must agree to numerical noise, not "roughly"
EXIT_ARTIFACTS, EXIT_STATS = 2, 3

problems: list[str] = []


def say(m=""):
    print(m, flush=True)


def head(t):
    say()
    say("=" * 78)
    say(t)
    say("=" * 78)


def bad(m):
    problems.append(m)
    say("  FAIL  " + m)


def ok(m):
    say("  ok    " + m)


# ─────────────────────────────────────────────────────── 1. artifacts
def verify_artifacts(d: Path) -> dict:
    head("1/5  ARTIFACTS")
    if not d.is_dir():
        say(f"  directory not found: {d}")
        sys.exit(EXIT_ARTIFACTS)

    required = ["results.json", "provenance.json", "config_used.yaml",
                "results.csv", "per_seed.csv", "RESULTS_README.md"]
    for f in required:
        (ok if (d / f).exists() else bad)(f"{f} present" if (d / f).exists()
                                         else f"{f} MISSING")
    if not (d / "results.json").exists():
        say("\n  results.json is missing; cannot continue.")
        sys.exit(EXIT_ARTIFACTS)
    return json.load(open(d / "results.json"))


# ─────────────────────────────────────────────────────── 2. seeds
def verify_seeds(d: Path, cfg: dict) -> dict:
    head("2/5  SEED COMPLETENESS AND SANITY")
    metrics = cfg["statistics"]["metrics"]
    seeds = [str(s) for s in cfg["seeds"]]
    arms = {}
    for arm_dir in sorted(p for p in d.iterdir() if p.is_dir()):
        name = arm_dir.name
        present, missing, nonfinite, nockpt, params = [], [], [], [], set()
        for s in seeds:
            f = arm_dir / f"seed_{s}" / "test_metrics.json"
            if not f.exists():
                missing.append(s)
                continue
            j = json.load(open(f))
            present.append(s)
            for m in metrics:
                v = j.get(m)
                if v is None or not isinstance(v, (int, float)) or \
                        not math.isfinite(float(v)):
                    nonfinite.append(f"{s}:{m}={v}")
            if not (arm_dir / f"seed_{s}" / "best_model.pth").exists():
                nockpt.append(s)
            if j.get("model_params"):
                params.add(j["model_params"])
        if not present:
            continue
        arms[name] = {"present": present, "params": sorted(params)}
        say(f"  arm '{name}'")
        (ok if not missing else bad)(
            f"    {len(present)}/{len(seeds)} seeds complete"
            + ("" if not missing else f"  MISSING {missing}"))
        (ok if not nonfinite else bad)(
            "    all metrics finite" if not nonfinite
            else f"    non-finite metrics: {nonfinite}")
        (ok if not nockpt else bad)(
            "    checkpoint present for every seed" if not nockpt
            else f"    missing best_model.pth for seeds {nockpt}")
        exp = next((a.get("expect_params") for a in cfg["arms"]
                    if a["name"] == name), None)
        if exp is not None:
            (ok if params == {exp} else bad)(
                f"    parameter count {params} matches config"
                if params == {exp} else
                f"    parameter count {params} != config's {exp}")
    if not arms:
        bad("no completed arms found in the directory")
    return arms


# ─────────────────────────────────────────────────────── 3. provenance
def verify_provenance(d: Path, cfg: dict) -> None:
    head("3/5  PROVENANCE")
    prov = json.load(open(d / "provenance.json"))

    snap = prov.get("config_snapshot", {})
    for key in ("seeds", "training", "arms"):
        (ok if snap.get(key) == cfg.get(key) else bad)(
            f"config '{key}' in the run matches the repo's config"
            if snap.get(key) == cfg.get(key) else
            f"config '{key}' DIFFERS between the run and the repo's config -- "
            f"the run did not use the recipe now on disk")

    dv = prov.get("data_verified") or {}
    if dv:
        for split, exp in cfg["data"]["expect_beats"].items():
            got = (dv.get(split) or {}).get("n")
            (ok if got == exp else bad)(
                f"remote data check: {split} had {exp} beats"
                if got == exp else
                f"remote data check: {split} had {got}, config expects {exp}")
    else:
        bad("provenance carries no data_verified block")

    env = prov.get("environment", {})
    g = env.get("git", {})
    say(f"  host {env.get('hostname')}  GPU {env.get('gpu_name')}  "
        f"torch {env.get('torch')}  CUDA {env.get('cuda')}")
    if g.get("commit"):
        ok(f"    git {g['commit'][:12]} on {g.get('branch')}"
           f"{'  (DIRTY working tree)' if g.get('dirty') else ''}")
    else:
        bad("    no git revision recorded")
    fails = prov.get("failures") or []
    (ok if not fails else bad)(
        "    no failed runs reported" if not fails
        else f"    {len(fails)} failed run(s): {fails}")


# ─────────────────────────────────────────────────────── 4. recompute
def verify_comparability(d: Path, cfg: dict) -> None:
    """Refuse a run that is not known to share its data with the published arms.

    AUDIT_FINDINGS.md H49: the 2026-09-26 run passed every other check in this
    script -- artifacts, seed completeness, provenance, and local-vs-remote
    statistics all agreed -- and this script printed CLEAN followed by an
    instruction to rewrite the paper's central claim. Its arms had been trained
    and selected on different records from the reference they were compared
    against. Internal consistency is not comparability, so both are required.
    """
    head("3b/5  COMPARABILITY WITH THE PUBLISHED ARMS")
    prov_path = d / "provenance.json"
    if not prov_path.exists():
        bad("provenance.json missing; comparability cannot be established")
        return
    prov = json.load(open(prov_path, encoding="utf-8"))

    gt_path = REPO / "configs" / "mitbih_split_counts.json"
    gt = json.load(open(gt_path, encoding="utf-8"))
    dv = prov.get("data_verified") or {}
    for split in ("train", "val", "test"):
        got = {str(k): int(v) for k, v in (dv.get(split, {}).get("class_counts") or {}).items()}
        want = {str(k): int(v) for k, v in gt["counts"][split].items() if int(v) or str(k) in got}
        got_nz = {k: v for k, v in got.items() if v}
        want_nz = {k: v for k, v in want.items() if v}
        if got_nz != want_nz:
            bad(f"{split}: the run trained on class counts {got_nz}, but the published "
                f"arms used {want_nz} ({gt_path.relative_to(REPO)}). These are different "
                f"records; no comparison with a published number is valid.")
        else:
            ok(f"{split}: class counts equal the PhysioNet-derived published split")

    eq = prov.get("data_equivalence") or (dv.get("data_equivalence") if isinstance(dv, dict) else None)
    if not eq:
        bad("the run has no data-equivalence record: it predates preflight 2b, so the "
            "arrays it trained on are not known to be the published ones")
    elif not eq.get("passed"):
        bad("the run's data-equivalence gate did not pass: " + "; ".join(eq.get("failures", [])[:5]))
    else:
        ok(f"data equivalence passed: {eq['n_seeds']} published checkpoints reproduced "
           f"(max |diff| test {eq['max_abs_test_diff']:.5f}, val {eq['max_abs_val_diff']:.5f})")


def load_metric(d: Path, metric: str) -> dict:
    out = {}
    for f in d.glob("seed_*/test_metrics.json"):
        v = json.load(open(f)).get(metric)
        if v is not None:
            out[f.parent.name.replace("seed_", "")] = float(v)
    return out


def recompute_and_diff(d: Path, cfg: dict, remote: dict, arms: dict) -> dict:
    head("4/5  RECOMPUTE STATISTICS LOCALLY AND DIFF AGAINST THE REMOTE SUMMARY")
    ref_dir = REPO / cfg["experiment"]["reference_arm"]["dir"]
    if not ref_dir.is_dir():
        bad(f"reference arm {ref_dir} not present locally; cannot recompute")
        return {}
    metrics = cfg["statistics"]["metrics"]
    local = {"reference_arm": {}, "arms": {}}
    for m in metrics:
        local["reference_arm"][m] = describe(load_metric(ref_dir, m).values())

    n_checked = 0
    for name in arms:
        arm_dir = d / name
        fam = {}
        for m in metrics:
            a, b = load_metric(ref_dir, m), load_metric(arm_dir, m)
            if b:
                fam[m] = paired_compare(a, b, metric_name=m,
                                        ci=cfg["statistics"]["ci"])
        if not fam:
            continue
        fam = holm_family(fam)
        local["arms"][name] = {"vs_reference": fam,
                               "descriptives": {m: describe(load_metric(arm_dir, m).values())
                                                for m in metrics}}
        wb = cfg["experiment"].get("within_batch_reference")
        if wb and name != wb and (d / wb).is_dir():
            wfam = {}
            for m in metrics:
                a, b = load_metric(d / wb, m), load_metric(arm_dir, m)
                if a and b:
                    wfam[m] = paired_compare(a, b, metric_name=m, ci=cfg["statistics"]["ci"])
            if wfam:
                local["arms"][name]["vs_within_batch_reference"] = holm_family(wfam)
        for key in ("vs_reference", "vs_within_batch_reference"):
            if key not in local["arms"][name]:
                continue
            n_checked += _diff_family(name, key, local["arms"][name][key],
                                      (remote.get("arms") or {}).get(name, {}).get(key, {}))
    say()
    say(f"  {n_checked} statistic fields compared")
    return local


def _diff_family(name, key, fam, rem) -> int:
    n_checked = 0
    if not rem:
        bad(f"remote results.json has no '{key}' statistics for arm '{name}'")
        return 0
    say(f"  arm '{name}'  [{key}]")
    if True:
        for m, r in fam.items():
            rr = rem.get(m)
            if rr is None:
                bad(f"    remote summary lacks metric '{m}'")
                continue
            for field in ("mean_a", "mean_b", "mean_diff", "cohens_d",
                          "p_value", "holm_p", "ci_low", "ci_high"):
                lv, rv = r[field], rr.get(field)
                n_checked += 1
                if rv is None:
                    bad(f"    {m}.{field} absent remotely")
                elif not (abs(float(lv) - float(rv)) <= TOL):
                    bad(f"    {m}.{field}: local {lv!r} != remote {rv!r}")
            if r["n_paired"] != rr.get("n_paired"):
                bad(f"    {m}.n_paired: local {r['n_paired']} != "
                    f"remote {rr.get('n_paired')}")
            if sorted(r["paired_seeds"]) != sorted(rr.get("paired_seeds", [])):
                bad(f"    {m}: paired seed sets differ between local and remote")
        ok(f"    {len(fam)} metrics recomputed and matched to within {TOL:g}")
    return n_checked


# ─────────────────────────────────────────────────────── 5. report
def report(cfg: dict, local: dict) -> None:
    head("5/5  NUMBERS FOR THE MANUSCRIPT")
    if not local.get("arms"):
        say("  nothing to report")
        return
    say("Sign convention: positive Cohen's d favours the REFERENCE arm, i.e.")
    say("favours keeping the ablated component. Holm is within each arm's")
    say(f"{len(cfg['statistics']['metrics'])}-metric family.")
    wb = cfg["experiment"].get("within_batch_reference")
    if wb and wb in local["arms"]:
        drift = local["arms"][wb]["vs_reference"]
        sig = [m for m, r in drift.items() if r["significant_after_holm"]]
        say()
        say("-" * 78)
        say(f"DRIFT CHECK -- '{wb}' is the published configuration re-trained in this job")
        say("-" * 78)
        for m, r in drift.items():
            say(f"  {m:<10} published {r['mean_a']:.4f}  replicate {r['mean_b']:.4f}  "
                f"d {r['cohens_d']:+.2f}  Holm p {r['holm_p']:.2e}"
                + ("  SIGNIFICANT" if r["significant_after_holm"] else ""))
        say("  " + ("no significant drift: the published numbers and this job's numbers "
                    "can be read together." if not sig else
                    "SIGNIFICANT drift on " + ", ".join(sig) + ": read ONLY the within-batch "
                    "comparisons below; pairing new arms with published numbers would "
                    "mix environment with architecture."))
    for name, a in local["arms"].items():
        if name == wb:
            continue
        meta = next((x for x in cfg["arms"] if x["name"] == name), {})
        say()
        say("-" * 78)
        say(f"arm '{name}'   flags: {' '.join(meta.get('flags', []))}")
        if meta.get("protects_claim"):
            say("bears on: " + " ".join(meta["protects_claim"].split()))
        say("-" * 78)
        say(f"  {'metric':<10} {'reference':>17} {'variant':>17} {'diff':>9} "
            f"{'95% CI':>21} {'d':>7} {'Holm p':>10}  sig")
        for m, r in a["vs_reference"].items():
            say(f"  {m:<10} {r['mean_a']:>9.4f}+-{r['std_a']:<6.4f} "
                f"{r['mean_b']:>9.4f}+-{r['std_b']:<6.4f} "
                f"{r['mean_diff']:>+9.4f} "
                f"[{r['ci_low']:>+8.4f},{r['ci_high']:>+8.4f}] "
                f"{r['cohens_d']:>+7.2f} {r['holm_p']:>10.2e}"
                f"  {'YES' if r['significant_after_holm'] else 'no'}")

        primary = a.get("vs_within_batch_reference")
        if primary:
            say(f"  PRIMARY (vs '{wb}', same job):")
            for m, r in primary.items():
                say(f"  {m:<10} {r['mean_a']:>9.4f}+-{r['std_a']:<6.4f} "
                    f"{r['mean_b']:>9.4f}+-{r['std_b']:<6.4f} "
                    f"{r['mean_diff']:>+9.4f} "
                    f"[{r['ci_low']:>+8.4f},{r['ci_high']:>+8.4f}] "
                    f"{r['cohens_d']:>+7.2f} {r['holm_p']:>10.2e}"
                    f"  {'YES' if r['significant_after_holm'] else 'no'}")
        fam_for_reading = primary or a["vs_reference"]
        pcwi = fam_for_reading.get("macro_f1")
        if name == "prior_swap_pt" and pcwi:
            say()
            say("  Reading for the manuscript (the word 'physiology'):")
            if pcwi["significant_after_holm"] and pcwi["cohens_d"] > 0:
                say("    The physiological assignment beats the permuted one: the benefit is")
                say("    morphological, and the name and title are supported.")
            elif pcwi["cohens_d"] > 0:
                say("    Directionally favours the physiological assignment but unresolved;")
                say("    keep 'structured initialisation' wording and report the CI.")
            else:
                say("    The permuted assignment is as good or better: the benefit is")
                say("    structural, not morphological. The method name and title must")
                say("    stop attributing it to physiology.")
        if name == "no_pcwi" and pcwi:
            say()
            say("  Reading for the manuscript:")
            if pcwi["significant_after_holm"] and pcwi["cohens_d"] > 0:
                say("    PCWI's contribution REPLICATES on the adopted architecture.")
                say("    Limitations item (9) can be retired and the Introduction's")
                say("    contribution 1 scope caveat removed; report this comparison")
                say("    alongside the validation ablation.")
            elif pcwi["cohens_d"] > 0:
                say("    PCWI's point estimate favours keeping it but does not reach")
                say("    significance after correction. Report the CI, keep the claim")
                say("    scoped to the base configuration, and say the adopted-model")
                say("    comparison is directionally consistent but unresolved.")
            else:
                say("    PCWI does NOT replicate on the adopted architecture. This")
                say("    contradicts the paper's central architectural claim and must")
                say("    change it: report both results, and reframe the contribution")
                say("    around what remains supported. Do not omit this.")
    say()
    say("Next: update the manuscript from these numbers, then run")
    say("    python src/verify_manuscript_numbers.py")
    say("adding a check for each new value so it is re-derived from the per-seed")
    say("files rather than trusted as typed.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="results/final_component_ablation")
    ap.add_argument("--config", default="configs/final_component_ablation.yaml")
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(open(REPO / args.config, encoding="utf-8"))
    d = REPO / args.dir

    say("=" * 78)
    say(f"integrating {args.dir}")
    say("=" * 78)

    remote = verify_artifacts(d)
    arms = verify_seeds(d, cfg)
    verify_provenance(d, cfg)
    verify_comparability(d, cfg)
    if problems:
        head("RESULT")
        say(f"{len(problems)} problem(s) found before statistics were recomputed:")
        for p in problems:
            say("  - " + p)
        say("\nFix these before using any of these numbers.")
        return EXIT_ARTIFACTS

    local = recompute_and_diff(d, cfg, remote, arms)
    if problems:
        head("RESULT")
        say(f"{len(problems)} problem(s):")
        for p in problems:
            say("  - " + p)
        say("\nLocal and remote statistics disagree. Do NOT use these numbers.")
        return EXIT_STATS

    report(cfg, local)
    head("RESULT")
    say("CLEAN -- artifacts, seeds, provenance and statistics all verified.")
    say("Local recomputation reproduces the remote summary exactly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
