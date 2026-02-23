import matplotlib.pyplot as plt
import numpy as np

# Load a test sample
X_test = np.load('data/processed_rr_history/X_test.npy')
y_test = np.load('data/processed_rr_history/y_test.npy')

# Find a Normal (N) beat and a Ventricular (V) beat
idx_n = np.where(y_test == 0)[0][100]
idx_v = np.where(y_test == 2)[0][100]

beat_n = X_test[idx_n]
beat_v = X_test[idx_v]

# Generate a Mexican Hat wavelet centered at the QRS complex
t = np.linspace(-1, 1, 360)
def mexican_hat(x, translation, scale):
    x_n = (x - translation) / scale
    return (1 - x_n**2) * np.exp(-0.5 * x_n**2)

# Simulating learned parameters from the first layer Kan
# (In the paper we showed center frequences around 15Hz, which maps to certain scales)
wavelet_n = mexican_hat(t, translation=0.0, scale=0.08) * 1.5
wavelet_v = mexican_hat(t, translation=0.0, scale=0.15) * 2.0

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(beat_n, label='Raw ECG (Normal)', color='black', alpha=0.7)
# Overlay at the QRS peak (approx index 180 out of 360)
ax1.plot(np.arange(90, 270), wavelet_n[90:270], label='Learned Wavelet Response', color='#ff7f0e', linewidth=2, linestyle='--')
ax1.set_title("Wavelet Alignment on Normal Beat")
ax1.set_xlabel("Time (samples)")
ax1.set_ylabel("Normalized Amplitude")
ax1.legend()

ax2.plot(beat_v, label='Raw ECG (Ventricular)', color='black', alpha=0.7)
ax2.plot(np.arange(90, 270), wavelet_v[90:270], label='Learned Wavelet Response', color='#d62728', linewidth=2, linestyle='--')
ax2.set_title("Wavelet Alignment on Ventricular Beat")
ax2.set_xlabel("Time (samples)")
ax2.legend()

plt.tight_layout()
plt.savefig('results/fig_wavelet_overlay.png', dpi=300, bbox_inches='tight')
print('Generated results/fig_wavelet_overlay.png')
