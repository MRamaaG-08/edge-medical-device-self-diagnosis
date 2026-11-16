"""A pretend ESP32: makes the exact signed lines the real node sends, from a
BIDMC record, following the recording protocol (LED dimming, tapping, arm
movement, cable knocks, slow fade). Lets the whole hub + dashboard be tested on
a laptop before any wire is connected.

    python -m phase2.hardware.hub.node_sim                    # writes data/sessions/sim_packets.txt
    python -m phase2.hardware.hub.node_sim --record bidmc07

Uses phase2/frontend.py (the C front end's twin) and models/edge_rf.joblib
(Bhumika's chip model). Adds one tampered and one replayed packet on purpose.
"""
import argparse
import json
import os

import joblib
import numpy as np

import config
from phase2.common import ensure_dir, load_corpus, load_raw_record
from phase2.contracts import EDGE_MODEL, FS
from phase2.frontend import StreamingFrontEnd, device_like
from phase2.hardware.hub.packets import load_key, sign
from phase2.hardware.hub.protocol import LED_NORMAL, SESSION_S, block_at, led_at
from selfdiag import features as F
from selfdiag import preprocess as P

ap = argparse.ArgumentParser()
ap.add_argument("--record", default=None)
ap.add_argument("--out", default="data/sessions/sim_packets.txt")
a = ap.parse_args()

names = [str(n) for n in load_corpus()["group_names"]]
rid = a.record or names[0]
raw, fs_in = load_raw_record(rid, n_surrogate=len(names))
x = device_like(P.resample_to(raw, fs_in, FS))
while x.size < SESSION_S * FS:                       # BIDMC is 8 min, the session is ~15
    x = np.concatenate([x, x])
x = x[: SESSION_S * FS]
t = np.arange(x.size) / FS
rng = np.random.default_rng(config.SEED)
dc = x.mean()
ac = (x - dc)
for i, ti in enumerate(t):                           # physical effects of each block
    b = block_at(ti)
    name = b[2] if b else ""
    gain = led_at(ti) / LED_NORMAL
    x[i] = dc * gain + ac[i] * gain
    if name.startswith("noisy"):
        x[i] += rng.normal(0, (0.6 if name.endswith("strong") else 0.25) * ac.std())
    if name == "arm_wander":
        x[i] += 0.02 * dc * np.sin(2 * np.pi * ti / 20)
if True:
    knocks = [k for k in range(x.size) if block_at(t[k]) and block_at(t[k])[2] == "cable_knocks" and k % (3 * FS) == 0]
    for k in knocks:
        x[k:k + 3] += 8 * ac.std() * rng.choice([-1, 1])

if os.path.exists(EDGE_MODEL):
    model = joblib.load(EDGE_MODEL)
else:                                                 # before Bhumika's handover: a stand-in
    from sklearn.ensemble import RandomForestClassifier
    print(f"[note] {EDGE_MODEL} not here yet; using a stand-in 20-tree model trained now")
    c = load_corpus()
    model = RandomForestClassifier(**dict(config.RF_PARAMS, n_estimators=20)).fit(c["X"], c["y"])
fe = StreamingFrontEnd(double=True)
key = load_key()
lines, seq = [], 0
for start in range(0, x.size, FS):                   # one second at a time, like the node's loop
    for w, tele in fe.push(x[start:start + FS]):
        f = F.extract(w, FS)
        p = model.predict_proba(f[None, :])[0]
        full = np.zeros(len(config.CLASSES)); full[model.classes_.astype(int)] = p
        tm = int((start + FS) * 1000 / FS)
        tele = {**{k: round(float(v), 4) for k, v in tele.items()},
                "die_temp_c": round(31.0 + 0.0005 * tm / 1000, 2), "led_current": led_at(tm / 1000)}
        payload = {"seq": seq, "t_ms": tm, "state": 2, "cls": int(np.argmax(full)),
                   "p": [round(float(v), 4) for v in full], "tele": tele, "proc_us": 0,
                   "w": [round(float(v), 3) for v in w]}
        lines.append(sign(json.dumps(payload, separators=(",", ":")), key)); seq += 1
    if not fe.calibrated:
        lines.append(sign(json.dumps({"seq": seq, "t_ms": int((start + FS) * 1000 / FS), "state": 1,
                                      "led": LED_NORMAL}, separators=(",", ":")), key)); seq += 1
bad = len(lines) // 2
lines.insert(bad, lines[bad].replace('"state":2', '"state":2,"x":1', 1))   # tampered copy
lines.insert(bad + 10, lines[bad - 5])                                      # replayed old packet
ensure_dir(os.path.dirname(a.out))
open(a.out, "w").write("\n".join(lines) + "\n")
print(f"wrote {len(lines)} signed lines ({SESSION_S} s session from {rid}) to {a.out}; 2 are bad on purpose")
