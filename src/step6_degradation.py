"""Step 6: Generate synthetic PPG degradation classes for supervised learning."""

import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('data', exist_ok=True)
os.makedirs('figures', exist_ok=True)

# ── Load clean windows ─────────────────────────────────
clean_windows = np.load('data/clean_windows.npy')
print(f"Clean windows loaded: {len(clean_windows)}")

# ════════════════════════════════════════════════════════
# DEGRADATION FUNCTIONS
# Each function takes a clean signal window and returns
# a degraded version simulating a specific sensor fault
# ════════════════════════════════════════════════════════

def inject_gaussian_noise(signal, snr_db):
    """
    Simulates: Electrical interference, poor shielding, 
               noisy environment near the sensor
    snr_db: Signal-to-Noise Ratio in decibels
            Lower value = more noise
            20 dB = mild noise, 5 dB = severe noise
    """
    signal_power = np.mean(signal ** 2)
    noise_power  = signal_power / (10 ** (snr_db / 10))
    noise        = np.random.normal(0, np.sqrt(noise_power), len(signal))
    return signal + noise


def inject_baseline_drift(signal, drift_amplitude=0.05, fs=100):
    """
    Simulates: Electrode drying out, poor skin contact,
               sweat changing skin conductivity over time
    drift_amplitude: How much the baseline drifts
                     Larger value = more drift
    """
    t = np.arange(len(signal)) / fs
    # Sinusoidal drift at 0.05 Hz (very slow oscillation)
    drift = drift_amplitude * np.sin(2 * np.pi * 0.05 * t)
    return signal + drift


def inject_attenuation(signal, factor=0.4):
    """
    Simulates: Sensor losing contact, LED power dropping,
               thick skin or dark pigmentation reducing signal
    factor: How much signal is retained (0 to 1)
            0.7 = mild (30% weaker)
            0.15 = severe (85% weaker)
    """
    return signal * factor


def inject_spikes(signal, probability=0.02, amplitude_factor=10):
    """
    Simulates: Patient moving their finger, 
               loose wire connection, static discharge
    probability: Fraction of samples affected (0.01 = 1%)
    amplitude_factor: How large the spikes are relative to signal
    """
    corrupted    = signal.copy()
    signal_std   = np.std(signal)
    
    for i in range(len(corrupted)):
        if np.random.random() < probability:
            # Random spike upward or downward
            direction = np.random.choice([-1, 1])
            spike     = amplitude_factor * signal_std * direction
            corrupted[i] += spike
    
    return corrupted


# ════════════════════════════════════════════════════════
# GENERATE FULL LABELLED DATASET
# ════════════════════════════════════════════════════════

# Class labels
# 0 = CLEAN
# 1 = NOISY
# 2 = DRIFTED
# 3 = ATTENUATED
# 4 = SPIKED

X_data   = []
y_labels = []

np.random.seed(42)  # for reproducibility

print("\nGenerating labelled dataset...")

# ── Class 0: CLEAN ─────────────────────────────────────
print("Generating CLEAN samples...")
for window in clean_windows:
    X_data.append(window)
    y_labels.append(0)

# ── Class 1: NOISY (3 severity levels) ────────────────
print("Generating NOISY samples...")
for window in clean_windows:
    for snr in [20, 10, 5]:  # mild, moderate, severe
        noisy = inject_gaussian_noise(window, snr_db=snr)
        X_data.append(noisy)
        y_labels.append(1)

# ── Class 2: DRIFTED (3 severity levels) ──────────────
print("Generating DRIFTED samples...")
for window in clean_windows:
    for amplitude in [0.02, 0.05, 0.12]:
        drifted = inject_baseline_drift(window, drift_amplitude=amplitude)
        X_data.append(drifted)
        y_labels.append(2)

# ── Class 3: ATTENUATED (3 severity levels) ───────────
print("Generating ATTENUATED samples...")
for window in clean_windows:
    for factor in [0.7, 0.4, 0.15]:
        attenuated = inject_attenuation(window, factor=factor)
        X_data.append(attenuated)
        y_labels.append(3)

# ── Class 4: SPIKED (3 severity levels) ───────────────
print("Generating SPIKED samples...")
for window in clean_windows:
    for prob, amp in [(0.01, 5), (0.02, 10), (0.05, 15)]:
        spiked = inject_spikes(window, 
                               probability=prob, 
                               amplitude_factor=amp)
        X_data.append(spiked)
        y_labels.append(4)

# ── Convert to arrays ──────────────────────────────────
X_data   = np.array(X_data)
y_labels = np.array(y_labels)

print(f"\nFull dataset generated:")
print(f"Total samples: {len(X_data)}")
print(f"Each sample shape: {X_data[0].shape}")
print("\nClass distribution:")
class_names = ['CLEAN', 'NOISY', 'DRIFTED', 'ATTENUATED', 'SPIKED']
for cls, name in enumerate(class_names):
    count = np.sum(y_labels == cls)
    print(f"  Class {cls} ({name}): {count} samples")

# ── Save dataset ───────────────────────────────────────
np.save('data/X_all.npy', X_data)
np.save('data/y_all.npy', y_labels)
print("\nSaved X_all.npy and y_all.npy to data/")
