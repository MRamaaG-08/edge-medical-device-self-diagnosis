"""Phase 2 contracts: the numbers and names every owner must agree on.

If a value here changes, all four people re-run their part. So it changes only
after a group decision, and only Ramaa edits this file.
"""
import config

# ---- the window, inherited from Phase 1 Spec A (the accepted paper)
SPEC = config.PRIMARY_SPEC                       # "A_100hz_5s"
FS = config.WINDOW_SPECS[SPEC]["fs"]             # 100 samples per second
WIN = int(round(config.WINDOW_SPECS[SPEC]["win_s"] * FS))      # 500 samples = 5 s
SHIFT = int(round(config.WINDOW_SPECS[SPEC]["shift_s"] * FS))  # 250 samples = 2.5 s

# ---- calibration on a real device (see selfdiag.preprocess.Calibration)
WARMUP_S = 5          # filter settling time thrown away at power-up
CALIB_S = 120         # clean, still baseline used to fix the normalisation scale

# ---- heart rate and the outcome-defined trust label
HR_BAND_HZ = (0.7, 3.5)          # 42 to 210 beats per minute


def ec13_tolerance(hr_ref_bpm):
    """ANSI/AAMI EC13 heart-rate accuracy: within 10% or 5 bpm, whichever is larger."""
    import numpy as np
    return np.maximum(5.0, 0.10 * np.asarray(hr_ref_bpm, dtype=float))


# ---- the certified release gate (Learn-then-Test)
GATE_ALPHA = 0.05    # at most 5% of the heart rates we SHOW may be wrong ...
GATE_DELTA = 0.05    # ... and we want 95% confidence that this holds

# ---- sensor telemetry sent by the node with every window
TELEMETRY = ["dc_level", "perfusion_index", "die_temp_c", "led_current"]

# ---- the class order, inherited from Phase 1 (frozen, the C model depends on it)
CLASSES = list(config.CLASSES)       # CLEAN, NOISY, DRIFTED, ATTENUATED, SPIKED

# ---- one hub log line (JSON Lines, one window per line). Dashboard reads exactly this.
LOG_FIELDS = {
    "t": "unix time when the hub received the window",
    "seq": "packet sequence number from the node",
    "t_node_s": "node clock in seconds since the session started (end of this window)",
    "node_class": "fault class decided ON THE ESP32",
    "node_proba": "5 class probabilities from the ESP32, in CLASSES order",
    "hub_class": "fault class recomputed on the Pi from the raw window (parity check)",
    "trust": "trust score 0..1 from the trust model on the Pi",
    "released": "true if the gate let the heart rate through",
    "hr": "heart rate in bpm, or null when withheld",
    "drift_alarms": "list of telemetry streams that raised a drift alarm",
    "telemetry": "dict with the TELEMETRY values for this window",
    "window": "the 500 calibrated samples (only kept when recording)",
}

# ---- where shared model files live (never in git? small JSON yes, joblib no)
MODELS_DIR = "models"
EDGE_MODEL = "models/edge_rf.joblib"         # Bhumika
TRUST_MODEL = "models/trust_rf.joblib"       # Tejaswini
GATE_FILE = "models/gate.json"               # Tejaswini
DRIFT_FILE = "models/drift.json"             # Vandana
