"""Step 8: Extract statistical and spectral features from PPG windows."""

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from scipy.fft import fft, fftfreq
import os

os.makedirs('data', exist_ok=True)

def extract_features(window, fs=100):
    """
    Takes one 500-sample window and returns a dictionary
    of 13 numbers that describe that window's characteristics.
    These 13 numbers are what the classifier actually learns from.
    """
    features = {}
    
    # ── TIME DOMAIN — 8 features ──────────────────────
    # These describe the signal's shape and statistics
    
    features['mean'] = float(np.mean(window))
    # Average value — drifted signals have non-zero mean
    
    features['variance'] = float(np.var(window))
    # How spread out values are — noisy signals have high variance
    
    features['rms'] = float(np.sqrt(np.mean(window**2)))
    # Root mean square — overall signal energy
    
    features['kurtosis'] = float(kurtosis(window))
    # Peakedness of distribution — spiked signals have very high kurtosis
    
    features['zcr'] = float(
        np.sum(np.diff(np.sign(window)) != 0) / len(window)
    )
    # Zero crossing rate — how often signal crosses zero
    # Noisy signals cross zero more frequently
    
    features['peak_to_peak'] = float(np.max(window) - np.min(window))
    # Difference between max and min — attenuated signals have small p2p
    
    features['skewness'] = float(skew(window))
    # Asymmetry of distribution
    
    features['mad'] = float(
        np.mean(np.abs(window - np.mean(window)))
    )
    # Mean absolute deviation — another spread measure
    
    # ── FREQUENCY DOMAIN — 5 features ─────────────────
    # These describe the signal's frequency content using FFT
    
    N       = len(window)
    freqs   = fftfreq(N, 1/fs)[:N//2]        # frequency axis
    fft_vals= np.abs(fft(window))[:N//2]     # magnitude spectrum
    power   = fft_vals ** 2                  # power spectrum
    total_power = np.sum(power) + 1e-10      # avoid division by zero
    prob    = power / total_power            # normalised power
    
    # Which frequency has the most energy
    features['dominant_freq'] = float(freqs[np.argmax(fft_vals)])
    # Clean PPG: dominant freq = heart rate (~1-2 Hz)
    # Noisy signal: dominant freq gets scattered
    
    # How spread out energy is across frequencies
    features['spectral_entropy'] = float(
        -np.sum(prob * np.log2(prob + 1e-10))
    )
    # Clean signal: low entropy (energy concentrated at heartbeat freq)
    # Noisy signal: high entropy (energy scattered everywhere)
    
    # Energy in the heartbeat frequency band (0.8 to 3.3 Hz)
    ppg_mask = (freqs >= 0.8) & (freqs <= 3.3)
    ppg_power = np.sum(power[ppg_mask])
    features['ppg_band_power'] = float(ppg_power)
    
    # Fraction of total energy that is in heartbeat band
    features['power_ratio'] = float(ppg_power / total_power)
    # Clean signal: high ratio (~0.7-0.9)
    # Noisy signal: low ratio (energy everywhere)
    
    # Weighted average frequency
    features['spectral_centroid'] = float(
        np.sum(freqs * power) / total_power
    )
    
    return features


def build_feature_dataset(X, y, fs=100):
    """
    Applies extract_features to every window in the dataset
    and returns a pandas DataFrame ready for ML training.
    """
    print(f"Extracting features from {len(X)} windows...")
    print("This takes a few minutes — please wait...")
    
    rows = []
    for i, window in enumerate(X):
        if i % 500 == 0:
            print(f"  Processed {i}/{len(X)} windows...")
        feat = extract_features(window, fs)
        rows.append(feat)
    
    df = pd.DataFrame(rows)
    df['label'] = y
    return df


if __name__ == '__main__':
    X = np.load('data/X_all.npy')
    y = np.load('data/y_all.npy')
    
    df = build_feature_dataset(X, y)
    
    # Check for any missing values
    print(f"\nDataset shape: {df.shape}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    
    # Fill any NaN with 0 (very rare edge case)
    df = df.fillna(0)
    
    # Save
    df.to_csv('data/feature_dataset.csv', index=False)
    print("Saved to data/feature_dataset.csv")
    print("\nFirst 3 rows:")
    print(df.head(3))
    print("\nFeature statistics:")
    print(df.describe())
