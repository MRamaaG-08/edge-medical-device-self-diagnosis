# Handover note: Tejaswini, Phase 2 Trust Gate



## What I am handing over



| File | What it is | Who uses it |

|---|---|---|

| models/trust_rf.joblib | Exported Random Forest trust model | Integration team |

| models/gate.json | Selected model and calibrated threshold | Integration team |

| phase2/gate/ | Trust corpus, model evaluation, gate certification and export scripts | Project team |

| results/phase2/t2_trust_models.json | Trust model comparison results | Project team |

| results/phase2/t3_gate.json | Risk-coverage gate evaluation | Project team |

| results/phase2/t5_butppg_check.json | External PPG validation results | Project team |

| phase2/gate/dashboard/ | Trust gate demonstration dashboard | Project team |

| results/figures/fig13_risk_coverage.png | Phase 1 classifier risk-coverage figure | Project team |



## How to re-run my part from scratch



Run from the repository root.



1. python -m phase2.gate.t1_build_trust_corpus

2. python -m phase2.gate.t2_trust_models

3. python -m phase2.gate.t3_certify_gate

4. python -m phase2.gate.t4_export_gate

5. python -m phase2.gate.t5_butppg_check

6. python -m phase2.gate.t6_make_demo_log



For the Phase 1 calibration and risk-coverage figure:



1. python steps/step11_calibration_abstention.py --spec A_100hz_5s

2. python steps/step10_figures.py



## My headline numbers



- Mean AUC: M0_TSQI 0.7252, M1_PCLEAN 0.5315, M2_TRUSTRF 0.8641.

- T3 recommended method: M0_TSQI.

- M2_TRUSTRF was exported based on its highest mean AUC.

- M2 did not consistently satisfy the 5% wrong-rate target across rotations.

- T5 external BUT PPG validation: 10.16% coverage and 36.92% wrong among shown.

- T5 promise was not held.

- Phase 1 classifier coverage at 90% accuracy: 85%.

- Phase 1 classifier coverage at 95% accuracy: 70%.



## What I checked



- [ ] python -m pytest tests -q passes (16 tests)

- [ ] All scripts run start to finish on real BIDMC data

- [ ] git status shows no data/ or secrets/ files



Step 11 calibration and Step 10 figure generation completed successfully locally.

The Phase 2 T1-T6 results were generated in Colab.



## Limits and things I am not sure about



- M2_TRUSTRF has the highest mean AUC, but T3 recommends M0_TSQI.

- M2 did not meet the external BUT PPG validation target.

- The exported calibration result is not independent certification.

- The risk-coverage figure is for the Phase 1 classifier, not the Phase 2 trust gate.

- The dashboard uses simulated demo data, not live hardware.

- External sensor-specific validation and recalibration are still needed.


