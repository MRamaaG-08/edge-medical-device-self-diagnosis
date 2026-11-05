"""Quick checks for the shared Phase 2 base. Run: python -m pytest tests -q"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase2 import contracts
from phase2.common import hr_estimate
from phase2.frontend import StreamingFrontEnd


def test_spec_a_sizes():
    assert (contracts.FS, contracts.WIN, contracts.SHIFT) == (100, 500, 250)


def test_ec13_tolerance():
    assert contracts.ec13_tolerance(40) == 5.0
    assert abs(contracts.ec13_tolerance(120) - 12.0) < 1e-9


def test_hr_estimate_finds_72_bpm():
    t = np.arange(500) / 100
    assert abs(hr_estimate(np.sin(2 * np.pi * 1.2 * t)) - 72) < 1.5


def test_frontend_calibrates_then_windows_every_2_5_s():
    fe = StreamingFrontEnd()
    t = np.arange(140 * 100) / 100
    x = 50000 * (1 + 0.01 * np.sin(2 * np.pi * 1.2 * t))
    wins = fe.push(x)
    assert fe.calibrated
    expected = 1 + (len(x) - fe.calibration_samples) // contracts.SHIFT
    assert len(wins) == expected
    w, tele = wins[-1]
    assert w.shape == (500,)
    assert 0.5 < tele["signal_amplitude"] < 2.0


def test_frontend_keeps_gain_loss_visible():
    """The whole point of fixed calibration: a dimmer signal stays dimmer."""
    fe = StreamingFrontEnd()
    t = np.arange(150 * 100) / 100
    x = 50000 * (1 + 0.01 * np.sin(2 * np.pi * 1.2 * t))
    x[140 * 100:] *= 0.5
    w, tele = fe.push(x)[-1]
    assert tele["signal_amplitude"] < 0.7
