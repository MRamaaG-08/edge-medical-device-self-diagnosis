"""Step 3: Plot the extracted raw PPG signal."""

import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('figures', exist_ok=True)

# Load the signal
ppg = np.load('data/clean_ppg_signal.npy')
fs = 100  # samples per second

print(f"Signal length: {len(ppg)} samples")
print(f"Duration: {len(ppg)/fs/60:.1f} minutes")
print(f"Min value: {ppg.min():.4f}")
print(f"Max value: {ppg.max():.4f}")
print(f"Mean value: {ppg.mean():.4f}")

# Plot the first 10 seconds
samples_10sec = 10 * fs  # 1000 samples
time_axis = np.arange(samples_10sec) / fs

plt.figure(figsize=(14, 4))
plt.plot(time_axis, ppg[:samples_10sec], 
         color='#0D9488', linewidth=1.0)
plt.title('Raw PPG Signal — First 10 Seconds', fontsize=13, fontweight='bold')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/raw_signal.png', dpi=150)
plt.show()
print("Figure saved to figures/raw_signal.png")
