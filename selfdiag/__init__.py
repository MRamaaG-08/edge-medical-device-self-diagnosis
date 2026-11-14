"""
Phase 1 rebuild: leakage-free re-evaluation of the PPG signal degradation
detection framework.

Package name matches the Phase 2 convention (selfdiag/) so modules move between
phases without renaming. features.py, degrade.py and preprocess.py are intended
to be the single reference implementation shared by both phases.
"""
__all__ = [
    "preprocess", "degrade", "features", "sqi",
    "splits", "metrics", "io_bidmc", "io_butppg", "surrogate",
]
