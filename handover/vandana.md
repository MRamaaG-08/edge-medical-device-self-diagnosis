# Handover note: Vandana H S — Sensor Drift Monitoring and Streaming Frontend Validation

## What I am handing over

| File | What it is | Who uses it |
|---|---|---|
| `phase2/drift/drift.py` | CUSUM-based sensor drift monitoring implementation using sensor telemetry | Integration team |
| `models/drift.json` | Exported configuration containing monitored streams, threshold, allowance and baseline windows | Device integration team |
| `results/phase2/v1_frontend_check.json` | Comparison of offline, single-pass streaming and double-pass streaming frontend performance | Frontend integration team |
| `results/phase2/v3_drift_eval.json` | Drift detection evaluation, false-alarm rates and detection delays for different thresholds | Evaluation team |
| `results/figures/phase2_drift_tradeoff.png` | Figure showing the relationship between detection threshold, detection delay and false alarms | Reviewers |
| `results/phase2/drift_streams.npz` | Generated simulated sensor streams for the drift scenarios | Drift evaluation pipeline |
| `handover/vandana.md` | Implementation details, commands, results and limitations | Project team |

## How to re-run my part from scratch

Run these commands from the project root after creating the Python environment and installing the project requirements.

### Phase 1 preparation and evaluation

```powershell
python steps/step02_fetch_bidmc.py
python steps/step03_order_audit.py
python steps/step04_build_dataset.py --spec A_100hz_5s
python steps/step05_train_eval.py --spec A_100hz_5s
python steps/step06_sqi_baselines.py
python steps/step09_ablation_importance.py --extended-drift
python steps/step10_figures.py
```

### Phase 2 drift monitoring

```powershell
python -m phase2.drift.v1_frontend_check
python -m phase2.drift.v2_drift_scenarios
python -m phase2.drift.v3_drift_eval
python -m phase2.drift.v4_export_drift
```

The Phase 2 commands generate the frontend validation results, simulated drift streams, drift evaluation results and exported configuration.

## My headline numbers (copy from results/phase2/*.json)

### V1 — Streaming frontend validation

The Phase 1 offline frontend was compared with two causal streaming approaches using the same trained model.

| Processing method | Accuracy |
|---|---:|
| Offline | 81.52% |
| Stream single | 80.56% |
| Stream double | 83.27% |

- Stream single showed a 0.97 percentage-point drop compared with offline processing.
- Stream double achieved 1.75 percentage points higher accuracy than the offline baseline in this evaluation.
- Recommended frontend: `stream_double`.

### V2 — Drift scenario generation

- Processed 12 BIDMC records.
- Generated 141 windows per scenario for each record.
- Generated simulated scenarios for contact loss, LED aging, pressure creep and normal operation.
- Output: `data/phase2/drift_streams.npz`.

### V3 — Drift detection evaluation

All three simulated fault types achieved 100% detection at all five tested thresholds.

| Threshold | False alarms/hour | Contact loss delay | LED aging delay | Pressure creep delay |
|---:|---:|---:|---:|---:|
| 5 | 20.57 | 12.5 s | 30.0 s | 47.5 s |
| 10 | 12.86 | 15.0 s | 36.2 s | 53.8 s |
| 20 | 8.57 | 22.5 s | 50.0 s | 107.5 s |
| 40 | 4.29 | 27.5 s | 67.5 s | 140.0 s |
| 80 | 1.71 | 40.0 s | 90.0 s | 186.2 s |

### Comparison with the Phase 1 per-window model

| Fault scenario | Detection rate | Detection delay |
|---|---:|---:|
| Contact loss | 100% | 76.2 s |
| LED aging | 50% | 152.5 s |
| Pressure creep | 25% | 135.0 s |

The Phase 1 model also produced alarms in 3 normal sessions.

### V4 — Exported drift configuration

The selected threshold was 80, which provided the lowest false-alarm rate among the five tested thresholds.

The exported configuration in `models/drift.json` contains:

| Parameter | Value |
|---|---|
| Monitored streams | `dc_level`, `perfusion_index`, `signal_amplitude` |
| Threshold | 80 |
| Allowance | 0.5 |
| Baseline windows | 24 |
| False alarms/hour | 1.71 |

## What I checked

- [ ] `python -m pytest tests -q` passes (16 tests) — verify this exact command before marking complete.
- [x] Phase 2 scripts completed using real BIDMC recordings for frontend validation and generated drift scenarios.
- [ ] `git status` shows no `data/` or secrets/ files.

## Limits and things I am not sure about

- The drift scenarios are simulated. Further validation with real MAX30102 sensor hardware and real-world fault conditions is required.
- The selected threshold of 80 produces 1.71 false alarms per hour, which is above the target of 1 false alarm per hour.
- Increasing the threshold reduces false alarms but increases detection delay.
- Pressure creep has the longest detection delay at threshold 80, taking 186.2 seconds.
- The streaming frontend results are based on the current evaluation recordings. Further testing across more recordings and on the actual ESP32 hardware is required.
- The optional BUT PPG transfer dataset was unavailable because its data files were not present in the expected folder.
- The Phase 2 evaluation demonstrates detection on simulated sensor faults; real-device performance should not be assumed from these results alone.
