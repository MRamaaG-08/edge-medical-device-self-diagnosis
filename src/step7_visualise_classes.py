"""Step 7: Visualize representative windows from each signal condition."""

import numpy as np
import matplotlib.pyplot as plt
from step6_degradation import (inject_gaussian_noise, 
                                inject_baseline_drift,
                                inject_attenuation, 
                                inject_spikes)
import os

os.makedirs('figures', exist_ok=True)

clean_windows = np.load('data/clean_windows.npy')
example = clean_windows[50]  # pick one representative window
fs = 100
time_axis = np.arange(len(example)) / fs

signals = [
    ('CLEAN\n(Healthy Signal)',     
     example,                                          '#2ecc71'),
    ('NOISY\n(Gaussian Noise)',     
     inject_gaussian_noise(example, snr_db=10),        '#e74c3c'),
    ('DRIFTED\n(Baseline Drift)',   
     inject_baseline_drift(example, drift_amplitude=0.08), '#3498db'),
    ('ATTENUATED\n(Weak Signal)',   
     inject_attenuation(example, factor=0.25),         '#f39c12'),
    ('SPIKED\n(Motion Artifact)',   
     inject_spikes(example, probability=0.03, amplitude_factor=12), '#9b59b6'),
]

fig, axes = plt.subplots(5, 1, figsize=(14, 14))
fig.suptitle('Five Signal Classes — Healthy vs Four Fault Types',
             fontsize=14, fontweight='bold', y=1.01)

for ax, (title, signal, color) in zip(axes, signals):
    ax.plot(time_axis, signal, color=color, linewidth=0.9)
    ax.set_title(title, fontweight='bold', fontsize=11)
    ax.set_ylabel('Amplitude')
    ax.set_ylim(-5, 5)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linewidth=0.5, linestyle='--')

axes[-1].set_xlabel('Time (seconds)')
plt.tight_layout()
plt.savefig('figures/five_classes.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved to figures/five_classes.png")
