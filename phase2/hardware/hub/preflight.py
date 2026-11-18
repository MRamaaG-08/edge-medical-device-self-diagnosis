"""Pre-flight check: run this before every live session, on the laptop or on the Pi.

    python -m phase2.hardware.hub.preflight                      (software only, no board)
    python -m phase2.hardware.hub.preflight --port COM6          (laptop, board plugged in)
    python -m phase2.hardware.hub.preflight --port /dev/ttyUSB0  (Pi, board plugged in)

It checks, in order: the folder, the virtual environment, the Python packages, the
model files and the scikit-learn version they were saved with, the signing key, the
serial port, and finally what the ESP32 is actually sending (the right sketch, the
right key). Every FAIL line says how to fix it. When everything says OK, the live
command will work.
"""
import argparse
import glob
import importlib
import json
import os
import re
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--port", help="COM6 on Windows, /dev/ttyUSB0 on the Pi; leave out to skip the board checks")
ap.add_argument("--seconds", type=int, default=10, help="how long to listen to the ESP32")
a = ap.parse_args()

fails = []


def ok(msg):
    print(f"  OK    {msg}")


def warn(msg, fix=""):
    print(f"  WARN  {msg}" + (f"\n        fix: {fix}" if fix else ""))


def fail(msg, fix):
    print(f"  FAIL  {msg}\n        fix: {fix}")
    fails.append(msg)


ON_PI = sys.platform.startswith("linux")
ACTIVATE = "source .venv/bin/activate" if ON_PI else ".venv\\Scripts\\Activate.ps1"
REQS = "requirements-pi.txt" if ON_PI else "requirements.txt"

# ------------------------------------------------------------------ 1. folder
print("\n1. Folder")
if os.path.exists("config.py") and os.path.isdir("phase2"):
    ok(f"project folder: {os.getcwd()}")
else:
    fail(f"not in the project folder (you are in {os.getcwd()})",
         "cd into edge-medical-device-self-diagnosis first")
    print("\nStopping: every other check needs the project folder.")
    sys.exit(1)

# ------------------------------------------------------------------ 2. venv
print("\n2. Virtual environment")
if sys.prefix != getattr(sys, "base_prefix", sys.prefix):
    ok(f"(.venv) active, Python {sys.version.split()[0]}")
else:
    fail("the virtual environment is not active", ACTIVATE)

# ------------------------------------------------------------------ 3. packages
print("\n3. Python packages")
need = [("numpy", "numpy"), ("scipy", "scipy"), ("sklearn", "scikit-learn"), ("joblib", "joblib"),
        ("serial", "pyserial"), ("cryptography", "cryptography"), ("matplotlib", "matplotlib")]
missing = []
for mod, pipname in need:
    try:
        m = importlib.import_module(mod)
        ok(f"{pipname} {getattr(m, '__version__', '')}")
    except Exception:                                       # noqa: BLE001
        missing.append(pipname)
if missing:
    fail("missing: " + ", ".join(missing), f"pip install -r {REQS}")
for mod, pipname in [("streamlit", "streamlit")]:
    try:
        importlib.import_module(mod)
        ok(f"{pipname} (dashboard)")
    except Exception:                                       # noqa: BLE001
        warn(f"{pipname} missing: only the dashboard needs it", f"pip install -r {REQS}")

# ------------------------------------------------------------------ 4. models
print("\n4. Model files")
try:
    import sklearn
    have = sklearn.__version__
except Exception:                                           # noqa: BLE001
    have = None


def saved_with(path):
    """The scikit-learn version written inside the file, read without loading it."""
    blob = open(path, "rb").read()
    found = set()
    for mm in re.finditer(rb"_sklearn_version", blob):
        v = re.search(rb"(\d+\.\d+\.\d+)", blob[mm.end(): mm.end() + 40])
        if v:
            found.add(v.group(1).decode())
    return found


needed = ["models/edge_rf.joblib", "models/gate.json", "models/drift.json"]
gate_method = None
if os.path.exists("models/gate.json"):
    gate_method = json.load(open("models/gate.json")).get("method")
    if gate_method != "M0_TSQI":
        needed.append("models/trust_rf.joblib")
for f in needed:
    if os.path.exists(f):
        ok(f"{f} present")
    else:
        fail(f"{f} missing", "git pull on the phase2 branch, or copy it from the laptop with scp")
if gate_method:
    ok(f"gate method: {gate_method}")
for f in ("models/edge_rf.joblib", "models/trust_rf.joblib"):
    if not os.path.exists(f) or (f.endswith("trust_rf.joblib") and gate_method == "M0_TSQI"):
        continue
    vs = saved_with(f)
    if not vs:
        warn(f"{f}: could not read the version it was saved with")
    elif have and vs != {have}:
        fail(f"{f} was saved with scikit-learn {', '.join(sorted(vs))}, this machine has {have}",
             "every model file, the laptop and the Pi must use ONE version; see the guide, 'Before every run'")
    else:
        ok(f"{f} saved with scikit-learn {', '.join(sorted(vs))} = installed {have}")

# ------------------------------------------------------------------ 5. key
print("\n5. Signing key")
key = None
if os.path.exists("secrets/node_key.txt"):
    key = open("secrets/node_key.txt").read().strip()
    if re.fullmatch(r"[0-9a-f]{64}", key):
        ok("secrets/node_key.txt present (64 hex characters)")
    else:
        fail("secrets/node_key.txt is not 64 hex characters", "copy it again from the laptop with scp")
else:
    fail("secrets/node_key.txt missing",
         "laptop: python -m phase2.hardware.hub.make_key ; Pi: scp it from the laptop")
hdr = "phase2/hardware/firmware/sensor_node/node_key.h"
if key and os.path.exists(hdr):
    if key in open(hdr).read():
        ok("node_key.h (the ESP32 copy) holds the same key")
    else:
        fail("node_key.h holds a DIFFERENT key from secrets/node_key.txt",
             "python -m phase2.hardware.hub.make_key, then re-upload sensor_node")

# ------------------------------------------------------------------ 6. port and node
if not a.port:
    print("\n6. Board: skipped (no --port given)")
else:
    print(f"\n6. Serial port {a.port}")
    try:
        import serial
    except Exception:                                       # noqa: BLE001
        serial = None
        fail("pyserial not installed", f"pip install -r {REQS}")
    if ON_PI and not os.path.exists(a.port):
        others = sorted(glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*"))
        fail(f"{a.port} does not exist", "plug the ESP32 into the Pi; ports seen now: " + (", ".join(others) or "none"))
        serial = None
    elif ON_PI and not os.access(a.port, os.R_OK | os.W_OK):
        fail(f"no permission to use {a.port}", "sudo usermod -aG dialout $USER, then log out of ssh and back in")
        serial = None
    if serial is not None:
        try:
            s = serial.Serial(a.port, 921600, timeout=1)
        except Exception as e:                              # noqa: BLE001
            s = None
            fail(f"cannot open {a.port}: {e}",
                 "close the Arduino Serial Monitor and any other hub; check the COM number in Device Manager")
        if s is not None:
            ok(f"{a.port} opened at 921600")
            print(f"\n7. What the ESP32 is sending (listening {a.seconds} s; finger on the sensor)")
            s.reset_input_buffer()
            kinds = {"signed": 0, "bad_sig": 0, "raw": 0, "scanner": 0, "nosensor": 0, "other": 0}
            example = {}
            t_end = time.time() + a.seconds
            while time.time() < t_end:
                ln = s.readline().decode(errors="ignore").strip()
                if not ln:
                    continue
                if ln.startswith("{"):
                    import hashlib, hmac                    # noqa: E401
                    good = False
                    if key and " " in ln:
                        payload, tag = ln.rsplit(" ", 1)
                        good = hmac.compare_digest(
                            hmac.new(key.encode(), payload.encode(), hashlib.sha256).hexdigest(), tag)
                    k = "signed" if good else "bad_sig"
                elif re.fullmatch(r"\d+,\d+,\d+", ln):
                    k = "raw"
                elif ln.startswith("found 0x") or ln == "nothing found":
                    k = "scanner"
                elif "MAX30102 not found" in ln:
                    k = "nosensor"
                else:
                    k = "other"
                kinds[k] += 1
                example.setdefault(k, ln[:90])
            s.close()
            if kinds["signed"]:
                ok(f"sensor_node is running and its key matches ({kinds['signed']} signed lines)")
            if kinds["bad_sig"]:
                fail(f"{kinds['bad_sig']} lines signed with a DIFFERENT key",
                     "the ESP32 was flashed with another key: make sure secrets/node_key.txt here is the "
                     "laptop's, re-run b4_make_sketches on the laptop, re-upload sensor_node")
            if kinds["raw"]:
                fail("the ESP32 is running the max30102_check TEST sketch (raw numbers)",
                     "Arduino IDE: open sensor_node.ino and Upload it")
            if kinds["scanner"]:
                fail("the ESP32 is running the I2C SCANNER sketch", "Arduino IDE: open sensor_node.ino and Upload it")
            if kinds["nosensor"]:
                fail("the ESP32 cannot see the MAX30102", "re-seat the 4 wires (SDA 21, SCL 22, 3V3, GND), press EN")
            if not any(kinds.values()):
                fail("nothing received from the ESP32",
                     "press the EN button; check the cable is a data cable; is this the right port?")
            elif kinds["other"] and not kinds["signed"]:
                warn(f"unrecognised lines, e.g. {example['other']!r}", "wrong sketch or a different baud rate")

# ------------------------------------------------------------------ summary
print()
if fails:
    print(f"NOT READY: {len(fails)} problem(s). Fix them from the top down and run this again.")
    sys.exit(1)
print("READY: run the live command now." + ("" if a.port else "  (Add --port to check the board too.)"))
