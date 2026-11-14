"""Phase 2: the same Phase 1 pipeline, now running on real hardware.

Layout (one folder per owner):
    contracts.py, common.py   shared, owned by Ramaa, nobody else edits them
    drift/                    Vandana   streaming front end + drift monitor
    gate/                     Tejaswini trust label, certified gate, dashboard
    edge/                     Bhumika   C model, C features, parity tests, benchmarks
    hardware/                 Ramaa     ESP32 firmware, Pi hub, recordings, integration
"""
