import re

with open("manuscript_complete.tex", "r", encoding="utf-8") as f:
    tex = f.read()

# Fix 1a: Abstract "10 independent seeds" -> "20 independent seeds"
tex = tex.replace("0.87 \\pm 0.02 across 10 independent seeds", "0.88 \\pm 0.03 across 20 independent seeds")
# Abstract V-recall mean was 0.88, std 0.03 based on validation report

# Fix 1b: Table 5 and other text mentions
tex = tex.replace("across 10 random seeds", "across 20 random seeds")
tex = tex.replace("10 random seed averages", "20 random seed averages")
tex = tex.replace("mean across 10 seeds: 0.87", "mean across 20 seeds: 0.88")
tex = tex.replace("Analysis of Seed Stability (n=10 seeds)", "Analysis of Seed Stability (n=20 seeds)")
tex = tex.replace("scores for 10 random seeds", "scores for 20 random seeds")
tex = tex.replace("10-Seed Comparison", "20-Seed Comparison")
tex = tex.replace("across 10 seeds", "across 20 seeds")
tex = tex.replace("$0.87 \\pm 0.02$ across 10 seeds", "$0.88 \\pm 0.03$ across 20 seeds")

# Fix 2: Table 6 Replacements (Aligning with Seed 42 - the published weights)
t6_old = r"""N (Normal) & 0.900 & 0.772 & 0.831 & 44,232 \\
S (Supra.) & 0.286 & 0.280 & 0.283 & 1,837 \\
\textbf{V (Vent.)} & \textbf{0.490} & \textbf{0.898} & \textbf{0.634} & \textbf{3,220} \\
F (Fusion) & 0.048 & 0.051 & 0.050 & 388 \\
Q (Unknown) & 0.000 & 0.000 & 0.000 & 7 \\ \bottomrule"""

t6_new = r"""N (Normal) & 0.958 & 0.824 & 0.886 & 44,232 \\
S (Supra.) & 0.502 & 0.290 & 0.367 & 1,837 \\
\textbf{V (Vent.)} & \textbf{0.550} & \textbf{0.858} & \textbf{0.670} & \textbf{3,220} \\
F (Fusion) & 0.002 & 0.028 & 0.004 & 388 \\
Q (Unknown) & 0.000 & 0.000 & 0.000 & 7 \\ \bottomrule"""
tex = tex.replace(t6_old, t6_new)

# Update textual claims based on Seed 42
tex = tex.replace("achieving a V-class Recall of 0.898", "achieving a V-class Recall of 0.858")
tex = tex.replace("(Recall 0.280)", "(Recall 0.290)")
tex = tex.replace("(Recall 0.051)", "(Recall 0.028)")

# Fix 3: Table 9 DOG F1
tex = tex.replace("DOG (Gaussian) & 0.354 & 0.856 & 0.268 \\\\", "DOG (Gaussian) & 0.328 & 0.854 & 0.168 \\\\")

with open("manuscript_complete.tex", "w", encoding="utf-8") as f:
    f.write(tex)

print("Manuscript updated successfully.")
