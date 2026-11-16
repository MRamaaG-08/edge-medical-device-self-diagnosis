# phase2/edge (Bhumika)
Run from the repository root, in order (needs gcc from MSYS2 for step 4):
1. `python steps/step08_edge_benchmark.py --spec A_100hz_5s`   (your Phase 1 step)
2. `python -m phase2.edge.b1_train_edge_rf`    smallest good forest -> models/edge_rf.joblib
3. `python -m phase2.edge.b2_export_c`         c/edge_rf.h, c/edge_filter.h, tests/golden_edge/
4. `python -m phase2.edge.b3_parity`           PARITY PASSED
5. `python -m phase2.edge.b4_make_sketches`    ESP32 and STM32 benchmark folders
6. `python -m phase2.edge.b5_edge_report`      after Ramaa sends esp32_bench.txt and stm32_bench.json
Then `handover/bhumika.md` and a pull request into `phase2`.
