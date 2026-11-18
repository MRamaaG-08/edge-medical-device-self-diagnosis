# Edge-Based Medical Device Self-Diagnosis System for Sensor Drift and Signal Degradation Detection

Final-year project, Department of ECE, New Horizon College of Engineering, Bengaluru.
Guides: TVS Adinarayana, Shasikiran S. Team: Mantripragada Ramaa Gayatri, U. Bhumika Asritha,
Tejaswini Badami, Vandana H. S.

A pulse (PPG) sensor that diagnoses its own faults and refuses to show a heart rate it cannot trust.

* **Phase 1** (software, accepted at AIHC 2027 as *Edge-Based Self-Diagnosis of PPG Sensors Using
  Simulation-Augmented Signal Quality Classification*): BIDMC, 53 records, 4 sensor faults x 3 severities
  injected after normalisation, 13 features, Random Forest 83.89% +/- 1.35 under record-level testing.
* **Phase 2** (this branch): the same pipeline on real hardware. MAX30102 + ESP32-WROOM-32 node,
  Raspberry Pi 4 hub, a certified heart-rate release gate, a telemetry drift monitor, measured cost on
  ESP32 and STM32F407.

## Layout
```
config.py, selfdiag/, steps/, run_all.py   Phase 1 pipeline (shared, frozen)
tests/                                     16 tests: pipeline order (11) + Phase 2 base (5)
phase2/contracts.py, common.py, frontend.py  Phase 2 shared base (frozen)
phase2/drift/      Vandana    front-end check, drift monitor
phase2/gate/       Tejaswini  outcome label, certified gate, dashboard
phase2/edge/       Bhumika    edge model, C code, parity, benchmarks
phase2/hardware/   Ramaa      firmware, Pi hub, recording protocol, analysis
handover/          one note per person
models/            frozen small models and settings (in git)
data/, secrets/    never in git
```

## Setup (Windows, Python 3.12)
```
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt -r requirements-phase2.txt
python -m pytest tests -q
```
Raspberry Pi: `pip install -r requirements-pi.txt`.

## Order of work
1. Phase 1 core: `steps/step03`, `step02`, `step04 --spec A_100hz_5s`, `step05 --spec A_100hz_5s`.
2. Each owner's Phase 1 steps on Spec A, then their `phase2/<folder>` scripts in numbered order.
3. Handover pull request into `phase2`; Ramaa runs `python -m phase2.check_handover <name>`.

Every script is run from the repository root as a module, for example
`python -m phase2.drift.v1_frontend_check`.

## Rules
Data never goes in git. Shared files change only after a group decision. Open-source tools only.
