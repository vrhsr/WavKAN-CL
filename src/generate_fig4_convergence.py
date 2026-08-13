"""
generate_fig4_convergence.py -- Real convergence figure (Baseline vs. Curriculum)

Written 2026-08-13, after the real 20-seed x 2-arm Phase 5 run completed
(AUDIT_FINDINGS.md C1/C5). Reads the ACTUAL training_history.json that train_pca.py
wrote for a given seed in both results/wavkan_v2_baseline/ and
results/wavkan_v2_curriculum/ -- no synthetic data, no hardcoded numbers, nothing
typed in by hand. This replaces the old fig4_convergence() in
src/generate_all_figures.py, which C1 found only ever plotted one model's data
despite the manuscript's caption describing a two-curve comparison.

Seed choice: defaults to seed=42 because that is the same default used everywhere
else in this codebase for a single-seed illustrative example (train_pca.py's own
--seed default, the WAS-score checkpoint, etc.) -- it is NOT chosen by looking at
which seed's curves look best. The two arms ran for different numbers of epochs
(independent early stopping, patience=15) -- this script does not truncate or pad
either curve to match the other; the plotted lines are exactly as long as each arm's
real training_history.json.

Usage:
    python src/generate_fig4_convergence.py --seed 42 \\
        --baseline-dir results/wavkan_v2_baseline \\
        --curriculum-dir results/wavkan_v2_curriculum \\
        --output results/figures/fig4_convergence_real.pdf
"""
import argparse
import json
from pathlib import Path


def load_history(run_dir: str, seed: int) -> list:
    path = Path(run_dir) / f"seed_{seed}" / "training_history.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- has seed {seed} of {run_dir} finished training? "
            "This script only ever reads real, already-produced history files."
        )
    with open(path) as f:
        return json.load(f)


def find_phase_transition_epoch(history: list):
    """Returns the epoch of the WARMUP->ANNEAL transition, or None if the arm has no
    such transition (e.g. the --no-curriculum baseline, whose phase is always
    'BASELINE' for every epoch)."""
    phases = [h["phase"] for h in history]
    for i in range(1, len(phases)):
        if phases[i - 1] == "WARMUP" and phases[i] != "WARMUP":
            return history[i]["epoch"]
    return None


def plot_convergence(baseline_history: list, curriculum_history: list, seed: int, save_path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.suptitle(
        f"Training Convergence: Baseline vs. Curriculum (seed={seed}, real run, 2026-08-13)\n"
        f"Baseline ran {len(baseline_history)} epochs, Curriculum ran {len(curriculum_history)} "
        f"epochs (independent early stopping, patience=15)",
        fontsize=10,
    )

    panels = [
        ("loss", "Training Loss", axes[0, 0]),
        ("val_macro_f1", "Validation Macro-F1", axes[0, 1]),
        ("val_v_recall", "Validation V-Recall", axes[1, 0]),
        ("val_s_recall", "Validation S-Recall", axes[1, 1]),
    ]

    transition_epoch = find_phase_transition_epoch(curriculum_history)

    for key, title, ax in panels:
        b_epochs = [h["epoch"] for h in baseline_history]
        b_vals = [h[key] for h in baseline_history]
        c_epochs = [h["epoch"] for h in curriculum_history]
        c_vals = [h[key] for h in curriculum_history]

        ax.plot(b_epochs, b_vals, color="#4e79a7", lw=2, label="Baseline (no curriculum)")
        ax.plot(c_epochs, c_vals, color="#e15759", lw=2, label="Curriculum (PCA)")
        if transition_epoch is not None:
            ax.axvline(transition_epoch, color="grey", lw=1, ls="--",
                       label="Curriculum warmup→anneal transition" if key == "loss" else None)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)

    axes[0, 0].legend(fontsize=8, loc="best")
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--baseline-dir", type=str, default="results/wavkan_v2_baseline")
    parser.add_argument("--curriculum-dir", type=str, default="results/wavkan_v2_curriculum")
    parser.add_argument("--output", type=str, default="results/figures/fig4_convergence_real.pdf")
    args = parser.parse_args()

    baseline_history = load_history(args.baseline_dir, args.seed)
    curriculum_history = load_history(args.curriculum_dir, args.seed)
    plot_convergence(baseline_history, curriculum_history, args.seed, args.output)
