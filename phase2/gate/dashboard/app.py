"""The dashboard: what a nurse (or a project guide) would look at.

It only READS the hub log (one JSON line per 5 s window), so it works the same with
the practice log, a saved session, or the live Pi.

    streamlit run phase2/gate/dashboard/app.py
    streamlit run phase2/gate/dashboard/app.py -- --log data/phase2/live_log.jsonl

Opens in your browser at http://localhost:8501
"""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))                       # so "config" and "phase2" are found

import numpy as np
import pandas as pd
import streamlit as st

ap = argparse.ArgumentParser()
ap.add_argument("--log", default=str(ROOT / "data" / "phase2" / "demo_log.jsonl"))
args, _ = ap.parse_known_args()

st.set_page_config(page_title="Sensor self-diagnosis", layout="wide")
st.title("Is this heart rate trustworthy?")
log_path = st.sidebar.text_input("Hub log file", args.log)
live = st.sidebar.toggle("Live (refresh every 2.5 s)", value=False)

p = Path(log_path)
if not p.exists():
    st.warning(f"No log at {p}. Make one with:  python -m phase2.gate.t6_make_demo_log")
    st.stop()
rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
if not rows:
    st.info("Log is empty. Waiting for the first window ..."); st.stop()

i = len(rows) - 1 if live else st.sidebar.slider("Window", 0, len(rows) - 1, len(rows) - 1)
r = rows[i]
gate = {}
gate_file = ROOT / "models" / "gate.json"
if gate_file.exists():
    gate = json.loads(gate_file.read_text())

c1, c2, c3, c4 = st.columns(4)
c1.metric("Heart rate", f"{r['hr']:.0f} bpm" if r["released"] else "withheld")
c2.metric("Trust score", f"{100 * r['trust']:.0f} / 100")
c3.metric("Sensor condition (ESP32)", r["node_class"].title())
c4.metric("Drift alarm", ", ".join(r["drift_alarms"]) if r["drift_alarms"] else "none")

if r["released"]:
    st.success("Reading released: the trust score is above the selected calibration threshold.")
else:
    st.error("Reading withheld: this window cannot be trusted. Check the sensor and the finger.")
if r["drift_alarms"]:
    st.warning("The sensor is drifting slowly (" + ", ".join(r["drift_alarms"]) +
               "). Re-seat the clip and recalibrate.")

if r.get("window"):
    st.subheader("Last 5 seconds of pulse (calibrated)")
    st.line_chart(pd.DataFrame({"pulse": r["window"]}), height=200)

df = pd.DataFrame([{"time (s)": round(x["t"] - rows[0]["t"], 1), "trust": x["trust"],
                    "released": x["released"], "hr": x["hr"], "node": x["node_class"],
                    "hub": x.get("hub_class")} for x in rows[: i + 1]])
st.subheader("Trust over the session")
chart = df[["time (s)", "trust"]].set_index("time (s)")
if gate.get("threshold") is not None:
    chart["certified threshold"] = gate["threshold"]
st.line_chart(chart, height=220)

a, b, c = st.columns(3)
a.metric("Readings released", f"{df['released'].mean():.0%}")
agree = (df["node"] == df["hub"]).mean() if df["hub"].notna().all() else float("nan")
b.metric("ESP32 and Pi agree", f"{agree:.0%}" if np.isfinite(agree) else "n/a")
c.metric("Windows", len(df))
if gate:
    st.caption(f"Selected calibration gate: {gate['method']}, "
               f"threshold {gate['threshold']:.3f}. "
               f"Calibration wrong rate: "
               f"{gate['calibration_wrong_among_shown']:.2%}. "
               f"External BUT PPG validation did not meet the 5% target.")
st.dataframe(df.tail(10), width="stretch")

obs = {"resourceType": "Observation", "status": "final" if r["released"] else "cancelled",
       "code": {"coding": [{"system": "http://loinc.org", "code": "8867-4", "display": "Heart rate"}]},
       "effectiveDateTime": datetime.fromtimestamp(r["t"], timezone.utc).isoformat(),
       "component": [{"code": {"text": "Signal trust score"},
                      "valueQuantity": {"value": round(100 * r["trust"], 1), "unit": "score"}}]}
if r["released"]:
    obs["valueQuantity"] = {"value": r["hr"], "unit": "beats/minute",
                            "system": "http://unitsofmeasure.org", "code": "/min"}
else:
    obs["dataAbsentReason"] = {"text": "Withheld: signal trust below the certified threshold"}
st.download_button("Export this reading (FHIR JSON)", json.dumps(obs, indent=2),
                   file_name=f"reading_{r['seq']}.json", mime="application/json")

if live:
    time.sleep(2.5)
    st.rerun()
