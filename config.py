"""
Central configuration for the Phase 1 rebuild.

Everything that a reviewer could ask "what value did you use" about lives here
and gets written into results/manifest.json by step04.
"""
from __future__ import annotations
import os

SEED = 42

# ---------------------------------------------------------------- paths
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
BIDMC_DIR = os.path.join(DATA, "bidmc")
BUTPPG_DIR = os.path.join(DATA, "butppg")
RESULTS = os.path.join(ROOT, "results")
FIGS = os.path.join(RESULTS, "figures")

for _d in (DATA, BIDMC_DIR, BUTPPG_DIR, RESULTS, FIGS):
    os.makedirs(_d, exist_ok=True)

# ---------------------------------------------------------------- window specs
# Two specs are run end to end so the paper can report that conclusions hold
# under both. SPEC_A (100 Hz, 5 s, 2.5 s shift) is the primary spec for the
# paper AND for Phase 2: the MAX30102 samples at 100 Hz natively, so the device
# needs no resampling. SPEC_B is kept only as a sensitivity check.
WINDOW_SPECS = {
    "A_100hz_5s": dict(fs=100, win_s=5.0, shift_s=2.50),   # 500 samples, 50% overlap
    "B_64hz_8s":  dict(fs=64,  win_s=8.0, shift_s=2.00),   # 512 samples, 75% overlap
}
PRIMARY_SPEC = "A_100hz_5s"  # headline numbers (won on real data, matches accepted paper); B is the sensitivity check

# Embedded footprint estimate for a tree ensemble exported as a compact C array:
# feature index (uint8) + threshold (float32) + left/right child (uint16 x 2)
# + leaf class (uint8) + padding = 12 bytes per node. Used instead of pickle size,
# which carries Python object overhead that does not exist on a microcontroller.
BYTES_PER_NODE = 12

# ---------------------------------------------------------------- filtering
HP_CUTOFF_HZ = 0.5
LP_CUTOFF_HZ = 4.0
FILTER_ORDER = 2

# ---------------------------------------------------------------- classes
CLASSES = ["CLEAN", "NOISY", "DRIFTED", "ATTENUATED", "SPIKED"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# ---------------------------------------------------------------- degradation
# Severity levels are indexed 1..3 (and 4 where an extended level is defined).
#
# IMPORTANT: degradation is injected into the ALREADY z-scored window, so every
# parameter below is relative to a unit-variance signal. That is what makes the
# levels comparable across fault types and it is what stops the normalisation
# step cancelling the attenuation (see steps/step03_order_audit.py).
DEGRADATION = {
    "NOISY": {
        "param": "snr_db",
        "levels": {1: 20.0, 2: 10.0, 3: 5.0},
    },
    "DRIFTED": {
        "param": "amplitude",           # multiples of window std
        "levels": {1: 0.02, 2: 0.05, 3: 0.12},
        # Level 4 is NOT part of the headline five-class problem. It exists only
        # for the drift-detectability analysis in step09, which quantifies why
        # Phase 1 saw Clean/Drifted confusion: at amplitude 0.02 on a unit
        # variance window the drift sits around 34 dB SNR, i.e. below the noise
        # floor of the feature set. Reported separately, never mixed in.
        "extended_levels": {4: 0.30},
        "freq_hz": (0.05, 0.30),        # sampled per window
    },
    "ATTENUATED": {
        "param": "gain",
        "levels": {1: 0.70, 2: 0.40, 3: 0.15},
    },
    "SPIKED": {
        "param": "probability",
        "levels": {1: 0.01, 2: 0.02, 3: 0.05},
        "amp_sigma": {1: 5.0, 2: 10.0, 3: 15.0},
    },
}

# ---------------------------------------------------------------- features
FEATURE_NAMES = [
    # time domain (8)
    "mean", "variance", "rms", "kurtosis",
    "zero_crossing_rate", "peak_to_peak", "skewness", "mean_abs_deviation",
    # frequency domain (5)
    "dominant_freq", "spectral_entropy", "ppg_band_power",
    "power_ratio", "spectral_centroid",
]
N_FEATURES = len(FEATURE_NAMES)
PPG_BAND_HZ = (0.5, 4.0)       # physiological band for band-power
BROAD_BAND_HZ = (0.0, 8.0)     # denominator for power_ratio

# ---------------------------------------------------------------- models
RF_PARAMS = dict(n_estimators=50, max_depth=10, class_weight="balanced",
                 random_state=SEED, n_jobs=-1)
SVM_PARAMS = dict(C=1.0, gamma="scale", kernel="rbf",
                  class_weight="balanced", random_state=SEED)
# RBF SVM is O(n^2). With all 53 records the corpus runs to ~10^5 windows, which
# would take hours per fold. Training subsample is capped; the cap is reported.
SVM_MAX_TRAIN = 20000

# ---------------------------------------------------------------- evaluation
N_FOLDS = 10                   # GroupKFold over records, and StratifiedKFold for the leaky comparison
WINDOW_LEVEL_SPLIT = (0.70, 0.15, 0.15)

# ---------------------------------------------------------------- BIDMC
BIDMC_RECORDS = [f"bidmc{i:02d}" for i in range(1, 54)]   # all 53
BIDMC_PPG_CHANNEL_CANDIDATES = ["PLETH", "Pleth", "PPG"]
BIDMC_FS_NATIVE = 125
