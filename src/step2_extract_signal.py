"""Step 2: Download BIDMC records, extract PPG channels, and resample to 100 Hz."""

import wfdb
import numpy as np
import os
from scipy.signal import resample

os.makedirs('data', exist_ok=True)

print("Loading PPG records from PhysioNet BIDMC dataset...")

all_ppg_signals = []

# Download 5 records — each is about 8 minutes long
# This gives us approximately 40 minutes of clean PPG data
record_names = ['bidmc01', 'bidmc02', 'bidmc03', 'bidmc04', 'bidmc05']

for rec_name in record_names:
    print(f"Downloading {rec_name}...")
    try:
        record = wfdb.rdrecord(rec_name, pn_dir='bidmc/1.0.0')
        
        # Find the PPG channel — it is called PLETH in this dataset
        signal_names = record.sig_name
        ppg_index = None
        for i, name in enumerate(signal_names):
            if 'PLETH' in name.upper() or 'PPG' in name.upper():
                ppg_index = i
                break
        
        if ppg_index is None:
            ppg_index = 0  # use first channel if PPG not found by name
        
        ppg_raw = record.p_signal[:, ppg_index]
        original_fs = record.fs  # 125 Hz for BIDMC
        
        # Remove any NaN values
        ppg_raw = ppg_raw[~np.isnan(ppg_raw)]
        
        # Resample from 125 Hz to 100 Hz
        # This is because we want exactly 100 samples per second
        target_fs = 100
        num_target_samples = int(len(ppg_raw) * target_fs / original_fs)
        ppg_100hz = resample(ppg_raw, num_target_samples)
        
        all_ppg_signals.append(ppg_100hz)
        print(f"  {rec_name}: {len(ppg_100hz)} samples "
              f"({len(ppg_100hz)/target_fs/60:.1f} minutes)")
        
    except Exception as e:
        print(f"  Could not download {rec_name}: {e}")

# Concatenate all signals into one long clean PPG signal
combined_ppg = np.concatenate(all_ppg_signals)
print(f"\nTotal clean PPG data: {len(combined_ppg)} samples")
print(f"Total duration: {len(combined_ppg)/100/60:.1f} minutes")

# Save to disk
np.save('data/clean_ppg_signal.npy', combined_ppg)
print("Saved to data/clean_ppg_signal.npy")
