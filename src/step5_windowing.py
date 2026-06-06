"""Step 5: Split the preprocessed PPG signal into fixed-length windows."""

import numpy as np
import os

ppg_clean = np.load('data/clean_ppg_preprocessed.npy')
fs = 100

WINDOW_SIZE = 500   # 5 seconds × 100 samples/sec = 500 samples
STEP_SIZE   = 250   # 50% overlap — new window every 2.5 seconds

print(f"Total signal length: {len(ppg_clean)} samples")
print(f"Window size: {WINDOW_SIZE} samples = 5 seconds")
print(f"Step size: {STEP_SIZE} samples = 2.5 seconds")

windows = []
start = 0
while start + WINDOW_SIZE <= len(ppg_clean):
    window = ppg_clean[start : start + WINDOW_SIZE]
    windows.append(window)
    start += STEP_SIZE

windows = np.array(windows)
print(f"\nTotal windows created: {len(windows)}")
print(f"Each window shape: {windows[0].shape}")

np.save('data/clean_windows.npy', windows)
print("Saved to data/clean_windows.npy")
