"""The hub: runs on the Raspberry Pi (or a laptop). Signed lines in, decisions out.

For every 5 s window the node sends, the hub:
    1. checks the signature and sequence number (rejects tampered or replayed lines)
    2. recomputes the 13 features from the window and runs the same forest as the chip
       (the two answers must agree: that is the live parity check)
    3. scores trust and applies the certified gate: release the heart rate or withhold it
    4. feeds the telemetry to the drift monitor
    5. writes one JSON line (contracts.LOG_FIELDS) that the dashboard reads

Live from the ESP32:
    python -m phase2.hardware.hub.live --port COM5 --session S01                  (laptop)
    python -m phase2.hardware.hub.live --port /dev/ttyUSB0 --session S01          (Pi)
Run the recording protocol at the same time (prompts + LED changes):
    python -m phase2.hardware.hub.live --port /dev/ttyUSB0 --session S01 --protocol
Replay a file (no hardware):
    python -m phase2.hardware.hub.live --replay data/sessions/sim_packets.txt --session SIM
"""
import argparse
import json
import os
import threading
import time

import numpy as np

import config
from phase2.common import ensure_dir, hr_estimate
from phase2.contracts import DRIFT_FILE, EDGE_MODEL, FS
from phase2.hardware.hub.packets import Verifier, load_key
from selfdiag import features as F

ap = argparse.ArgumentParser()
ap.add_argument("--port"); ap.add_argument("--replay"); ap.add_argument("--session", default="S00")
ap.add_argument("--protocol", action="store_true", help="run the recording protocol (needs --port)")
ap.add_argument("--no-window", action="store_true", help="do not store the raw windows in the log")
a = ap.parse_args()

# ---- the pieces handed over by the three software owners (each one optional, so the
# ---- hub still runs while a handover is pending)
edge = scorer = monitor = None
try:
    import joblib
    edge = joblib.load(EDGE_MODEL)
except Exception as e:                                     # noqa: BLE001
    print(f"[warn] no hub copy of the chip model ({e}); parity check off")
try:
    from phase2.gate.score import TrustScorer
    scorer = TrustScorer.load()
except Exception as e:                                     # noqa: BLE001
    print(f"[warn] no trust gate yet ({e}); every heart rate will be shown as UNCHECKED")
try:
    from phase2.drift.drift import DriftMonitor
    cfg = json.load(open(DRIFT_FILE))
    monitor = DriftMonitor(cfg["streams"], cfg["threshold"], cfg["allowance"], cfg["baseline_windows"])
except Exception as e:                                     # noqa: BLE001
    print(f"[warn] no drift monitor yet ({e})")

ver = Verifier(load_key())
log_path = os.path.join(ensure_dir("data/sessions"), f"{a.session}.jsonl")
log = open(log_path, "a")
ser = None
lock = threading.Lock()


def run_protocol(t0):
    """Prompts the operator block by block and sends the LED commands."""
    from phase2.hardware.hub.protocol import BLOCKS, SESSION_S, led_at
    marks = open(os.path.join("data/sessions", f"{a.session}_blocks.csv"), "a")
    marks.write("wall_time,session_s,block,class,level\n")
    shown, last_led = None, None
    while (now := time.time() - t0) < SESSION_S:
        b = next(x for x in BLOCKS if x[0] <= now < x[1])
        if b[2] != shown:
            shown = b[2]
            marks.write(f"{time.time():.2f},{now:.1f},{b[2]},{b[3]},{b[4]}\n"); marks.flush()
            print(f"\n>>> [{now:5.0f} s] {b[2].upper()}: {b[5]}  (until {b[1]} s)\n")
        led = led_at(now)
        if led != last_led and ser is not None:
            with lock:
                ser.write(f"L{led}\n".encode())
            last_led = led
        time.sleep(0.5)
    marks.write(f"{time.time():.2f},{SESSION_S},end,,\n"); marks.close()
    print("\n>>> PROTOCOL FINISHED. Press Ctrl+C to stop the hub.\n")


def keyboard_commands():
    """Type L<0-255> + Enter to set the LED (demo: L5 = weak sensor, L31 = normal), R to recalibrate."""
    while True:
        s = input().strip().upper()
        if s[:1] in ("L", "R") and ser is not None:
            with lock:
                ser.write((s + "\n").encode())
            print(f">>> sent {s} to the node")


def lines():
    global ser
    if a.replay:
        yield from open(a.replay)
        return
    import serial
    ser = serial.Serial(a.port, 921600, timeout=5)
    ser.write(b"R\n")                                    # restart calibration = session start
    if a.protocol:
        threading.Thread(target=run_protocol, args=(time.time(),), daemon=True).start()
    else:
        threading.Thread(target=keyboard_commands, daemon=True).start()
    while True:
        with lock:
            raw = ser.readline()
        yield raw.decode(errors="ignore")


n_ok = n_agree = 0
try:
    for line in lines():
        if not line.strip().startswith("{"):
            if line.strip():
                print("node says:", line.strip())
            continue
        pkt = ver.check(line)
        if pkt is None:
            print(f"REJECTED (tampered or replayed). Total rejected: {ver.rejected}")
            continue
        if pkt.get("state", 2) < 2:
            print(f"  calibrating ... {pkt['t_ms'] / 1000:5.0f} s (keep still)", end="\r")
            continue
        t_start = time.perf_counter()
        w = np.asarray(pkt["w"], dtype=float)
        node_class = config.CLASSES[int(pkt["cls"])]
        hub_class = None
        if edge is not None:
            p = edge.predict_proba(F.extract(w, FS)[None, :])[0]
            hub_class = config.CLASSES[int(edge.classes_[int(np.argmax(p))])]
            n_agree += hub_class == node_class
        trust = scorer.score(w) if scorer else None
        released = scorer.released(trust) if scorer else True
        hr = hr_estimate(w)
        tele = pkt["tele"]
        alarms = monitor.update(tele) if monitor else []
        if alarms:
            monitor.reset()                              # alarm raised: relearn normal after re-seating
        n_ok += 1
        rec = {"t": round(time.time(), 2), "seq": pkt["seq"], "t_node_s": pkt["t_ms"] / 1000, "node_class": node_class,
               "node_proba": pkt["p"], "hub_class": hub_class,
               "trust": round(trust, 4) if trust is not None else None, "released": bool(released),
               "hr": round(hr, 1) if released else None, "drift_alarms": alarms, "telemetry": tele,
               "node_ms": pkt.get("proc_us", 0) / 1000,
               "hub_ms": round(1000 * (time.perf_counter() - t_start), 2)}
        if not a.no_window:
            rec["window"] = pkt["w"]
        log.write(json.dumps(rec) + "\n"); log.flush()
        print(f"seq {pkt['seq']:5d}  chip {node_class:<10} pi {str(hub_class):<10} "
              f"trust {('%5.2f' % trust) if trust is not None else '  -  '}  "
              f"{('HR %5.1f' % hr) if released else 'HR withheld'}  {'DRIFT ' + ','.join(alarms) if alarms else ''}")
except KeyboardInterrupt:
    pass
finally:
    log.close()
    agree = f"{n_agree / n_ok:.1%}" if (n_ok and edge is not None) else "n/a"
    print(f"\n{n_ok} windows logged to {log_path}; chip and Pi agree on {agree}; {ver.rejected} lines rejected")
