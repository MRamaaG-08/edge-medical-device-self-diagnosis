# Handover note: <bhumika>, <edge model, C code, parity, benchmarks>


## What I am handing over

models/edge_rf.joblib -- the chip model in Python form	to Ramaa

phase2/edge/c/* (including the generated edge_rf.h, edge_filter.h)-- everything the chip runs to Ramaa

tests/golden_edge/*, results/phase2/b3_parity.txt -- the parity proof to Ramaa 

results/phase2/b1_edge_rf.json-- the size choice and its evidence paper table
phase2/edge/esp32_bench/, stm32_bench/	--- ready-to-flash benchmarks to Ramaa 


## My headline numbers (copy from results/phase2/*.json)

Sketch uses 436480 bytes (33%) of program storage space. Maximum is 1310720 bytes.
Global variables use 44364 bytes (13%) of dynamic memory, leaving 283316 bytes for local variables. Maximum is 327680 bytes.

{"cpu_hz": 84000000, "frontend_per_sample": 15013.5332, "features": 58575616, "model": 3176.19995,
 "total_per_window": 62332176, "flash_bytes": 174736, "ram_bytes": 24252}

 board       front end/sample    features     model   per window   busy %
host                 0.59 us     1633 us    2.1 us      1783 us   0.071
esp32               60.11 us   282765 us   39.2 us    297831 us  11.913
stm32              178.73 us   697329 us   37.8 us    742050 us  29.682
saved results/phase2/b5_edge_report.json

-


