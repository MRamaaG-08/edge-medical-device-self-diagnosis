"""
Regression tests that pin the facts the rest of the project depends on.

    python -m pytest tests/ -v
    python tests/test_pipeline_order.py        # runs without pytest too
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from selfdiag import preprocess as P
from selfdiag import degrade as D
from selfdiag import features as F
from selfdiag import sqi as Q
from selfdiag import surrogate as S

FS = 64
WIN_S, SHIFT_S = 8.0, 2.0


def _window(seed=1):
    x = S.make_surrogate(40.0, FS, seed=seed)
    return P.prepare_record(x, FS, FS, WIN_S, SHIFT_S)[0]


def test_zscore_cancels_gain_exactly():
    """The identity the whole pipeline order exists to avoid."""
    w = _window()
    for g in (0.70, 0.40, 0.15, 0.01):
        assert np.max(np.abs(P.zscore(w) - P.zscore(g * w))) < 1e-9, (
            f"gain {g} survived z-scoring, which contradicts the known identity")


def test_attenuation_survives_correct_order():
    """Injecting after normalisation keeps the class detectable."""
    rng = np.random.default_rng(0)
    w = _window()
    att, meta = D.apply_degradation(w, "ATTENUATED", 3, FS, rng)
    assert meta["gain"] == 0.15
    assert np.std(att) < 0.5 * np.std(w), "attenuation did not reduce amplitude"
    fa, fb = F.extract(w, FS), F.extract(att, FS)
    assert np.max(np.abs(fa - fb)) > 1e-6, "attenuation invisible to features"


def test_feature_vector_contract():
    """Length, order and dtype are frozen; the C port depends on all three."""
    w = _window()
    v = F.extract(w, FS)
    assert v.shape == (config.N_FEATURES,)
    assert v.dtype == np.float32
    assert len(config.FEATURE_NAMES) == config.N_FEATURES
    assert np.all(np.isfinite(v)), "non-finite feature produced on a clean window"


def test_all_degradations_change_the_window():
    rng = np.random.default_rng(0)
    w = _window()
    for cls in config.CLASSES[1:]:
        for lvl in (1, 2, 3):
            dw, _ = D.apply_degradation(w, cls, lvl, FS, rng)
            assert not np.allclose(dw, w), f"{cls} level {lvl} was a no-op"


def test_clean_is_untouched():
    rng = np.random.default_rng(0)
    w = _window()
    cw, meta = D.apply_degradation(w, "CLEAN", 1, FS, rng)
    assert np.array_equal(cw, w)
    assert meta == {}


def test_windowing_shape_and_overlap():
    x = np.arange(64 * 30, dtype=np.float64)
    w = P.make_windows(x, FS, WIN_S, SHIFT_S)
    assert w.shape[1] == int(WIN_S * FS)
    assert np.array_equal(w[1][: int(WIN_S * FS) - int(SHIFT_S * FS)],
                          w[0][int(SHIFT_S * FS):]), "overlap is wrong"


def test_no_partial_trailing_window():
    """Zero padding would fabricate low-variance windows that mimic ATTENUATED."""
    x = np.ones(FS * 9, dtype=np.float64)
    w = P.make_windows(x, FS, WIN_S, SHIFT_S)
    assert w.shape[0] == 1


def test_calibration_preserves_amplitude_loss():
    """The real-path normaliser must NOT cancel attenuation the way z-score does."""
    w = _window()
    cal = P.Calibration().fit(w)
    a = cal.apply(w)
    b = cal.apply(0.15 * w)
    assert np.std(b) < 0.5 * np.std(a), (
        "fixed-scale calibration behaved like per-window z-scoring")


def test_sqi_outputs_are_finite():
    w = _window()
    vals = Q.extract_sqi_batch(w[None, :], FS)
    assert vals.shape == (1, 3)
    assert np.all(np.isfinite(vals))


def test_sqi_degrades_with_noise():
    """Template-matching SQI should fall when the signal is wrecked."""
    rng = np.random.default_rng(0)
    w = _window()
    bad, _ = D.apply_degradation(w, "NOISY", 3, FS, rng)
    assert Q.t_sqi(bad, FS) <= Q.t_sqi(w, FS) + 1e-6


def test_effective_snr_explains_mild_drift():
    """
    Severity 1 drift must sit far above the level anything could detect. This is
    the quantitative explanation for the Clean/Drifted confusion reported in
    Phase 1, so it is pinned here rather than left as a claim in the paper.
    """
    rng = np.random.default_rng(0)
    w = _window()
    d1, _ = D.apply_degradation(w, "DRIFTED", 1, FS, rng)
    d3, _ = D.apply_degradation(w, "DRIFTED", 3, FS, rng)
    snr1 = D.effective_snr_db(w, d1)
    snr3 = D.effective_snr_db(w, d3)
    assert snr1 > snr3, "severity ordering inverted"
    assert snr1 > 25.0, f"expected mild drift above 25 dB SDR, got {snr1:.1f}"


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {fn.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"  ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
