"""Step 1: Verify access to a PhysioNet BIDMC PPG record."""

import wfdb
import numpy as np
import os

# Create data folder if it doesn't exist
os.makedirs('data', exist_ok=True)

print("Downloading PPG record from PhysioNet...")
print("This may take a minute depending on your internet speed...")

# Download one PPG record from the BIDMC dataset
# This dataset has clean PPG signals from 53 subjects
record = wfdb.rdrecord('bidmc01', 
                        pn_dir='bidmc/1.0.0')

# The record has multiple signals — we want the PPG channel
# Print what channels are available
print("Available signal names:", record.sig_name)
print("Sampling frequency:", record.fs, "Hz")
print("Signal shape:", record.p_signal.shape)

# Save the raw signal information
print("\nDownload successful!")
print(f"Total samples: {record.p_signal.shape[0]}")
print(f"Duration: {record.p_signal.shape[0] / record.fs / 60:.1f} minutes")
