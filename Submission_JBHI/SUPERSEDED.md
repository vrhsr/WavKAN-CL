# SUPERSEDED — do not submit from this folder

As of 2026-09-30 the live manuscript is `Submission_Array/manuscript.tex`, prepared for **Array (Elsevier)** after the final pre-submission audit (`AUDIT_FINDINGS.md` H58–H67, `CHANGELOG.md` 2026-09-30).

`ieee_manuscript_v2.tex` in this folder is kept only as history. Its Method section describes the *intended* system rather than the one the code ran, in several places the audit verified against code and checkpoints:

- the beat window is not centred (the R-peak is at sample 90);
- the "gated P-region fusion" branch reads 28 ms before to 194 ms after R, not the P-wave;
- every "RAC" checkpoint was selected in the class-balanced warm-up phase;
- the RR features include non-beat annotations;
- INCART/SVDB were not processed by the training pipeline;
- the "physiology-constrained" reading of PCWI is not supported.

Stale items here that should **not** be reused:

- `graphical_abstract.jpg`: the old 95K-parameter "WavKAN-CL" model, V-recall 0.88 and wearable icons. Apparently AI-generated artwork.
- `graphical_abstract_caption.txt`, `graphical_abstract_text.txt`, `cover_letter_jbhi.txt`: old title, old model, old journal.
- `fig12_augmentation.pdf`, `rr_interval_ablation.pdf`, `baseline_comparison_seed_stability.pdf`, `training_convergence_curves.pdf`: not used by the live manuscript.

Deleting this folder was deliberately left to the project owner. Everything in it is git-tracked and recoverable.
