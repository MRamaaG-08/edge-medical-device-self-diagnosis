"""Day-one check: read the max30102_check sketch for 60 s and answer three questions.

    1. Is the sample rate really 100 per second (within 1%)?
    2. Are any samples missing or doubled?
    3. Is there a pulse? (a clear peak between 42 and 210 beats per minute)

    python -m phase2.hardware.hub.check_stream --port COM5            (Windows laptop)
    python -m phase2.hardware.hub.check_stream --port /dev/ttyUSB0    (Raspberry Pi)

Writes data/sessions/check_stream.csv and results/figures/phase2_first_signal.png
"""
import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import serial

import config
from phase2.common import ensure_dir, hr_estimate
from phase2.contracts import FS
from selfdiag import preprocess as P

ap = argparse.ArgumentParser(); ap.add_argument("--port", required=True); ap.add_argument("--seconds", type=int, default=60)
a = ap.parse_args()
rows = []
with serial.Serial(a.port, 921600, timeout=2) as s:
    s.reset_input_buffer()
    t_end = time.time() + a.seconds
    while time.time() < t_end:
        parts = s.readline().decode(errors="ignore").strip().split(",")
        if len(parts) == 3 and all(p.isdigit() for p in parts):
            rows.append([int(p) for p in parts])
d = np.array(rows, dtype=float)
ensure_dir("data/sessions"); np.savetxt("data/sessions/check_stream.csv", d, delimiter=",", fmt="%d",
                                        header="millis,ir,red", comments="")
ms, ir = d[:, 0], d[:, 1]
rate = (len(ms) - 1) / ((ms[-1] - ms[0]) / 1000)
gaps = np.diff(ms)
x = P.bandpass(ir, FS)
hr = hr_estimate(x[-500:])
print(f"samples received : {len(ms)}")
print(f"sample rate      : {rate:.2f} per second   {'OK' if abs(rate - 100) <= 1 else 'CHECK (expected 100)'}")
print(f"gaps over 30 ms  : {int((gaps > 30).sum())}   (FIFO bursts of ~10 ms are normal)")
print(f"IR light level   : {ir.mean():.0f} counts   {'OK' if ir.mean() > 20000 else 'LOW: finger on the sensor? clip firm?'}")
print(f"pulse            : {hr:.0f} bpm from the last 5 s (check against your own wrist)")
fig, ax = plt.subplots(2, 1, figsize=(7, 4), dpi=140)
ax[0].plot((ms - ms[0]) / 1000, ir, lw=0.6); ax[0].set_ylabel("raw IR")
ax[1].plot((ms[-1000:] - ms[0]) / 1000, x[-1000:], lw=0.8); ax[1].set_ylabel("band-passed"); ax[1].set_xlabel("time (s)")
fig.suptitle("First signal from the MAX30102"); fig.tight_layout()
fig.savefig(os.path.join(ensure_dir(config.FIGS), "phase2_first_signal.png"))
print("saved data/sessions/check_stream.csv and results/figures/phase2_first_signal.png")
