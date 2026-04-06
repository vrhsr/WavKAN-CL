# IEEE JBHI — Submission Readiness Audit

## Paper: WavKAN-CL
**Target:** IEEE Journal of Biomedical and Health Informatics (J-BHI)  
**Portal:** https://mc.manuscriptcentral.com/jbhi-embs

---

## 1️⃣ Scientific Readiness

| Criterion | Status | Evidence |
|-----------|--------|----------|
| Clear contribution (not incremental) | ✅ | First wavelet-KAN under inter-patient protocol + curriculum learning + 95K params |
| Clinical relevance | ✅ | V-recall 0.88 = primary safety metric for arrhythmia monitoring |
| Strict patient-level split | ✅ | DS1/DS2 inter-patient protocol (De Chazal et al.) |
| External validation dataset | ✅ | Zero-shot PTB-XL cross-dataset evaluation |
| Multiple seeds | ✅ | 20 independent random seeds |
| Statistical significance test | ✅ | Wilcoxon signed-rank test (p < 0.05 for V-recall) |
| Ablation study | ✅ | 8 ablation configurations (wavelet, structural, training) |
| SOTA comparison | ✅ | Table 9: 6 methods compared, protocol distinction noted |
| Limitations section | ✅ | 5 explicit limitations acknowledged |

---

## 2️⃣ Manuscript Structure (IEEE Compliance)

| Section | Status |
|---------|--------|
| Abstract (≤200 words, no citations) | ✅ ~195 words, citations removed |
| Index Terms (IEEEkeywords) | ✅ 6 keywords |
| Introduction with \IEEEPARstart | ✅ |
| Related Work | ✅ |
| Methodology | ✅ |
| Dataset & Evaluation Protocol | ✅ |
| Experimental Setup | ✅ |
| Results | ✅ |
| Discussion | ✅ |
| Conclusion | ✅ |
| Acknowledgment | ✅ Separate section |
| Data Availability | ✅ Separate section |
| Ethical Statement | ✅ Public dataset, no IRB needed |
| Conflict of Interest | ✅ Separate section |
| References (IEEEtran style) | ✅ |

---

## 3️⃣ Figures & Tables

| Figure/Table | File | Quality |
|-------------|------|---------|
| Fig 1: Architecture | final_methodology_workflow_v2.pdf | ✅ Vector PDF |
| Fig 2: WavKAN micro-arch | wavkan_micro_architecture_v2.pdf | ✅ Vector PDF |
| Fig 3: Seed stability | fig_seed_stability.pdf | ✅ Vector PDF |
| Fig 4: Convergence curves | fig_convergence_curves.pdf | ✅ Vector PDF |
| Fig 5: Confusion matrix | final_confusion_matrix_main.pdf | ✅ Vector PDF |
| Fig 6: RR ablation | fig_rr_ablation_v2.pdf | ✅ Vector PDF |
| Fig 7: Learned wavelets | final_learned_wavelets.pdf | ✅ Vector PDF |
| Tables 1–9 | Embedded in LaTeX | ✅ booktabs formatting |

---

## 4️⃣ Submission Files

| File | Status | Notes |
|------|--------|-------|
| ieee_wavkan.tex | ✅ | All fixes applied |
| references.bib | ✅ | 20 entries, all keys matched |
| CoverLetter.txt | ✅ | Retargeted to JBHI |
| 7 figure PDFs | ✅ | All present, vector format |
| Manuscript PDF | ⚠️ **COMPILE THIS** | Use Overleaf or local LaTeX |

---

## 5️⃣ Before You Click Submit

- [ ] **Compile PDF** via Overleaf and check page count (target ≤7 pages)
- [ ] **Register ORCID** for both authors at orcid.org
- [ ] **Run plagiarism check** (IEEE uses iThenticate — aim for <20% similarity)
- [ ] **Proofread** the compiled PDF one final time
- [ ] **Suggest 3–4 reviewers** (optional but recommended by JBHI)

---

## ⚠️ Page Count Warning

JBHI regular papers: **≤7 pages**. Overlength: **$250/page**.

Your paper has 7 figures + 9 tables + 1 algorithm. If it exceeds 7 pages:
- Consider merging Tables 5+6 (Gap + Stats → single table)
- Reduce figure widths (0.85→0.7 for some)
- Move PTB-XL results to supplementary material

---

## 🔴 Files NOT Needed (Elsevier-only)

- ~~Highlights.txt~~
- ~~CRediT_Author_Statement.txt~~
- ~~Declaration_of_Competing_Interests.txt~~
