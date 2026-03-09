# WavKAN-CL: How I Built an Interpretable AI for Heart Arrhythmia Detection That's 95% Smaller Than Existing Models

*By Venkateramanan Manivannan*

---

> **TL;DR:** I designed WavKAN-CL — a deep learning framework that uses learnable wavelets instead of black-box neural networks to detect dangerous heart rhythms from ECG signals. It achieves 88% recall on life-threatening ventricular arrhythmias with only 95K parameters, making it deployable on wearable devices. Here's the full story of how and why I built it.

---

## The Problem That Keeps Cardiologists Up at Night

Every year, cardiovascular diseases kill more people than any other cause worldwide. The electrocardiogram (ECG) remains the gold standard for detecting arrhythmias — irregular heartbeats that can be fatal if missed. But here's the catch: a single Holter monitor recording generates **110,000+ heartbeats over 24 hours**. Expecting a cardiologist to manually analyze every single beat is unrealistic, error-prone, and simply unscalable.

Deep learning has stepped in to automate this process. Convolutional Neural Networks (CNNs), Transformers, and recurrent models have all been thrown at the problem — and they report stunning accuracies, often exceeding 99%.

**So, problem solved?**

Not even close. There are three critical reasons why most existing AI models for ECG arrhythmia detection **cannot be safely deployed** in real clinical settings.

---

## Three Barriers to Clinical AI Deployment

### 1. The Black-Box Trust Problem

When a CNN classifies a heartbeat as "ventricular ectopic," a cardiologist has no way to understand *why*. The model's internal representations are opaque. You can apply post-hoc methods like Grad-CAM or LIME to generate explanations, but these are approximations — they don't reveal the model's actual reasoning. This opacity is a dealbreaker for regulatory approval and clinical trust.

### 2. The Inter-Patient Generalization Illusion

Here's a dirty secret in ECG research: **most published models cheat without knowing it**. The standard practice in many studies is *intra-patient evaluation* — randomly splitting heartbeats from the same patients into training and test sets. This means the model has already seen the patient's baseline morphology during training. Of course it performs well — it has essentially memorized the patient.

When you enforce *inter-patient evaluation* — where test patients are completely unseen during training — accuracy drops dramatically. A recent systematic review by Silva et al. (2025) found that **only 4.1% of 122 studies** met rigorous clinical evaluation criteria.

### 3. The Edge Computing Gap

Vision Transformers and multi-million-parameter architectures achieve impressive results, but they cannot run on a battery-powered wearable device. If your arrhythmia detector requires a GPU to function, it has no place on a patient's wrist.

---

## Enter WavKAN-CL: A Different Approach

I set out to build a model that addresses all three problems simultaneously. The result is **WavKAN-CL** — a framework that combines three key innovations:

### Learnable Wavelets Instead of Black-Box Layers

The core insight behind WavKAN-CL comes from **Kolmogorov-Arnold Networks (KANs)** — a recent breakthrough in neural network design where learnable activation functions are placed on network *edges* rather than nodes. Unlike standard Multi-Layer Perceptrons (MLPs) where activations are fixed functions like ReLU, KAN edges learn their own mathematical transformations.

But not all basis functions are equal. While prior work (like MAK-Net) used B-spline polynomials, I chose the **Mexican Hat wavelet** — a function that acts as a natural high-frequency transient detector. This isn't arbitrary: the Mexican Hat wavelet is mathematically identical to the negative second derivative of a Gaussian, and it naturally aligns with the sharp, spike-like morphology of the QRS complex in ECG signals.

**In plain English:** each edge in the network learns to "tune" its own wavelet to detect specific ECG features — the sharp upstroke of a ventricular beat, the subtle P-wave abnormality. The learned wavelet parameters (translation and dilation) are directly interpretable by clinicians.

### Dual-Branch Architecture

WavKAN-CL doesn't just look at beat morphology. It has two branches:

- **Morphology Branch (WavKAN → BiGRU):** Extracts shape-based features using 360 × 64 learnable wavelet edges, followed by a bidirectional GRU for temporal projection.
- **Rhythm Branch (RR-interval MLP):** Encodes the timing pattern of the last 5 heartbeats using dynamic normalization. This is crucial for detecting supraventricular arrhythmias, where the rhythm pattern matters more than the shape.

Both branches fuse into an 80-dimensional vector that feeds a lightweight classifier.

### Curriculum Learning for Class Imbalance

The MIT-BIH Arrhythmia Database has a massive class imbalance problem: Normal beats outnumber Fusion beats by **112:1**. Standard training with cross-entropy loss causes the model to collapse onto the majority class.

Instead of using synthetic data augmentation (SMOTE), which I found actually *hurts* inter-patient generalization by creating physiologically unrealistic beats, I designed a **two-phase curriculum learning strategy:**

- **Phase 1 — Discovery (first 30% of training):** The model sees *only* minority-class heartbeats. No Normal beats at all. This forces the network to learn discriminative features of rare arrhythmias before being overwhelmed by the majority class.
- **Phase 2 — Consolidation (remaining 70%):** Training shifts to the full dataset with class-weighted cross-entropy and 10× emphasis on the supraventricular (S) class.

This is fundamentally different from traditional curriculum learning (easy-to-hard). I structure training by **class rarity, not difficulty**.

---

## Results: What WavKAN-CL Achieves

All results are evaluated under the **strict inter-patient DS1/DS2 protocol** — the test set contains 22 patients whom the model has never seen.

| Metric | Result |
|--------|--------|
| **Ventricular (V) Recall** | **0.88 ± 0.03** (mean over 20 random seeds) |
| **Peak V-Recall** | **0.93** (best seed) |
| **Total Parameters** | **95,189** (0.38 MB) |
| **Inference Latency** | **0.78 ms/beat** (CPU) |
| **Validation–Test Gap** | **≤ 0.01** (curriculum learning eliminates overfitting) |

For context:
- **MAK-Net** (the closest KAN-based competitor): 6.1M parameters, 24 ms/beat, GPU required, intra-patient evaluation only
- **WavKAN-CL**: 95K parameters, 0.78 ms/beat, CPU only, strict inter-patient evaluation

That's a **>95% reduction in model size** and **30× faster inference** — while using the harder evaluation protocol.

---

## Why This Matters for the Future of Wearable Health AI

The convergence of three properties makes WavKAN-CL significant:

1. **Glass-box interpretability:** Unlike every other inter-patient ECG model in the literature, WavKAN-CL is *intrinsically* interpretable. The learned wavelet parameters can be directly visualized and mapped to physiological features. No post-hoc attribution needed.

2. **Edge deployability:** At 95K parameters and sub-millisecond inference, this model can run on ARM Cortex-M microcontrollers — the kind found in smartwatches and wearable Holter monitors.

3. **Clinical evaluation compliance:** WavKAN-CL satisfies all four criteria of the E3C clinical evaluation framework proposed by Silva et al. (2025) — placing it among the mere 4.1% of recent models that meet these standards.

---

## Honest Limitations

I believe in transparent reporting. Here's what WavKAN-CL doesn't do well:

- **Supraventricular (S) beat detection** remains at 29% recall. These beats look almost identical to Normal sinus rhythm, and with less than 2% representation in the data, they're extremely challenging under inter-patient conditions.
- **Fusion (F) beats** are near-chance (2.8% recall) due to extreme scarcity (388 test samples) and inherently ambiguous morphology.
- The **overall Macro-F1 score** is 0.362 — consistent with unaugmented inter-patient benchmarks in the literature, but not impressive as a single number.

The point isn't to claim overall superiority. It's to demonstrate that **clinically safe ventricular detection** is achievable with a tiny, interpretable model — and that's the metric that saves lives.

---

## What's Next

My future work focuses on:
- **P-wave attention mechanisms** for improved S-class discrimination
- **Noise-based augmentation** that preserves physiological structure
- **INT8 quantization** and deployment validation on low-power edge hardware
- **Multi-dataset training** with domain adaptation for cross-hospital generalization

---

## About the Author

**Venkateramanan Manivannan** is a researcher specializing in deep learning, artificial intelligence, and neural network architectures. He holds a B.E. in Computer Science and Engineering and is currently pursuing his M.Tech in Computer Science (Big Data Analytics) at VIT, Vellore, India. His research interests span machine learning, explainable AI, computer vision, NLP, biomedical signal processing, and computationally efficient architectures for edge deployment.

This work was conducted under the guidance of **Prof. Dr. Ramanathan Lakshmanan**, Professor at the Department of IoT, School of Computer Science and Engineering, VIT, Vellore.

📄 **Paper:** Submitted to IEEE Journal of Biomedical and Health Informatics (JBHI)
💻 **Code:** [github.com/vrhsr/WavKAN-CL](https://github.com/vrhsr/WavKAN-CL)

---

*If you found this article insightful, consider following **Venkateramanan Manivannan** for more research on interpretable AI, deep learning for healthcare, and edge-deployable neural networks.*

---

### Suggested Medium Tags
`Artificial Intelligence` · `Deep Learning` · `Machine Learning` · `Healthcare AI` · `ECG` · `Arrhythmia Detection` · `Explainable AI` · `Neural Networks` · `Wearable Technology` · `Computer Science`

### SEO Keywords (naturally embedded throughout)
- Venkateramanan Manivannan
- WavKAN-CL
- arrhythmia detection deep learning
- interpretable AI ECG
- Kolmogorov-Arnold Networks
- wavelet neural network
- inter-patient ECG classification
- edge AI healthcare
- explainable artificial intelligence
- wearable heart monitor AI
- VIT Vellore research
- green AI
- curriculum learning imbalanced data
