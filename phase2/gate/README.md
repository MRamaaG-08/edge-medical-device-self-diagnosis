# phase2/gate (Tejaswini)
Run from the repository root, in order:
1. `python steps/step06_sqi_baselines.py --spec A_100hz_5s`, `steps/step07_butppg_transfer.py --inspect` then `--spec A_100hz_5s`, `steps/step11_calibration_abstention.py --spec A_100hz_5s`
2. `python -m phase2.gate.t1_build_trust_corpus`   outcome label vs ECG heart rate
3. `python -m phase2.gate.t2_trust_models`         three trust scores
4. `python -m phase2.gate.t3_certify_gate`         Learn-then-Test gate
5. `python -m phase2.gate.t4_export_gate`          writes models/gate.json, models/trust_rf.joblib
6. `python -m phase2.gate.t5_butppg_check`         the promise on a phone camera
7. `python -m phase2.gate.t6_make_demo_log` then `streamlit run phase2/gate/dashboard/app.py`
Then `handover/tejaswini.md` and a pull request into `phase2`.
