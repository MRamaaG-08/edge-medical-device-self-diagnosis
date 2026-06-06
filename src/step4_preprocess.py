"""Step 4: Filter and normalize the clean PPG signal."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import os

os.makedirs('data', exist_ok=True)
os.makedirs('figures', exist_ok=True)

# ── Load raw signal ────────────────────────────────────
ppg_raw = np.load('data/clean_ppg_signal.npy')
fs = 100

# ── Filter 1: High-Pass Filter at 0.5 Hz ──────────────
# Purpose: Remove the large slow-moving DC component
# (the big offset that sits under the heartbeat signal)
# A heartbeat repeats 60-100 times per minute = 1.0-1.7 Hz
# Anything below 0.5 Hz is just slow drift we don't want

def highpass_filter(signal, cutoff=0.5, fs=100, order=2):
    nyquist = fs / 2          # Nyquist frequency = half the sample rate
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, signal)

# ── Filter 2: Low-Pass Filter at 4.0 Hz ───────────────
# Purpose: Remove high-frequency electrical noise
# Heart rate cannot exceed 200 bpm = 3.3 Hz
# Anything above 4 Hz is noise we don't want

def lowpass_filter(signal, cutoff=4.0, fs=100, order=2):
    nyquist = fs / 2
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, signal)

# ── Step 3: Z-Score Normalisation ─────────────────────
# Purpose: Bring signal to zero mean and unit variance
# This makes signals from different people comparable
# regardless of how bright their skin is or how the sensor was placed

def zscore_normalise(signal):
    mean = np.mean(signal)
    std = np.std(signal)
    if std < 1e-8:  # avoid dividing by zero
        return signal - mean
    return (signal - mean) / std

# ── Apply all three steps in sequence ─────────────────
print("Applying preprocessing pipeline...")
ppg_after_hp   = highpass_filter(ppg_raw)
ppg_after_lp   = lowpass_filter(ppg_after_hp)
ppg_clean      = zscore_normalise(ppg_after_lp)

print(f"Raw signal    — mean: {ppg_raw.mean():.2f}, "
      f"std: {ppg_raw.std():.2f}")
print(f"Clean signal  — mean: {ppg_clean.mean():.4f}, "
      f"std: {ppg_clean.std():.4f}")
# After normalisation, mean should be ~0 and std should be ~1

# ── Save clean signal ──────────────────────────────────
np.save('data/clean_ppg_preprocessed.npy', ppg_clean)
print("Saved to data/clean_ppg_preprocessed.npy")

# ── Plot comparison ────────────────────────────────────
samples = 10 * fs  # 10 seconds
time_axis = np.arange(samples) / fs

fig, axes = plt.subplots(3, 1, figsize=(14, 9))

axes[0].plot(time_axis, ppg_raw[:samples], 
             color='#e74c3c', linewidth=1.0)
axes[0].set_title('Step 1: Raw Signal (with DC offset)', 
                  fontweight='bold')
axes[0].set_ylabel('Amplitude')

axes[1].plot(time_axis, ppg_after_lp[:samples], 
             color='#f39c12', linewidth=1.0)
axes[1].set_title('Step 2: After High-Pass + Low-Pass Filter', 
                  fontweight='bold')
axes[1].set_ylabel('Amplitude')

axes[2].plot(time_axis, ppg_clean[:samples], 
             color='#0D9488', linewidth=1.0)
axes[2].set_title('Step 3: After Z-Score Normalisation (Final Clean Signal)', 
                  fontweight='bold')
axes[2].set_ylabel('Amplitude')
axes[2].set_xlabel('Time (seconds)')

for ax in axes:
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/preprocessing_comparison.png', dpi=150)
plt.show()
print("Figure saved to figures/preprocessing_comparison.png")
