# phase2/drift (Vandana)
Run from the repository root, in order:
1. `python steps/step09_ablation_importance.py --extended-drift`   (your Phase 1 step)
2. `python -m phase2.drift.v1_frontend_check`    live front end vs Phase 1
3. `python -m phase2.drift.v2_drift_scenarios`   slow faults with known start
4. `python -m phase2.drift.v3_drift_eval`        detection vs false alarms
5. `python -m phase2.drift.v4_export_drift`      writes models/drift.json
Then `handover/vandana.md` and a pull request into `phase2`.
