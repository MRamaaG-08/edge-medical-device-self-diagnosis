"""Live demo dashboard: the whole chain from the finger to the decision, on one screen.

It only READS files the hub writes, so it never disturbs a session:
    data/sessions/<SESSION>.jsonl          one line per 5 s window (contracts.LOG_FIELDS)
    data/sessions/<SESSION>_status.json    calibrating / streaming, and the security counters

Three ways to use it (pick in the sidebar):
    Live    follows a session that is being recorded right now
    Replay  plays a saved session back as if it were live (the backup if the hardware fails)
    Review  scrub through a saved session window by window

    streamlit run phase2/hardware/dashboard/app.py -- --log data/sessions/DEMO.jsonl
On the Pi, so a laptop or phone can watch it:
    streamlit run phase2/hardware/dashboard/app.py --server.address 0.0.0.0 -- --log data/sessions/DEMO.jsonl
"""
import argparse
import glob
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

CLASSES = ["CLEAN", "NOISY", "DRIFTED", "ATTENUATED", "SPIKED"]
CLASS_COL = {"CLEAN": "#34d399", "NOISY": "#fbbf24", "DRIFTED": "#a78bfa", "ATTENUATED": "#60a5fa", "SPIKED": "#f87171"}
CLASS_TXT = {"CLEAN": "Good contact, clean pulse",
             "NOISY": "Finger moving: motion noise",
             "DRIFTED": "Baseline wandering (arm or pressure moving)",
             "ATTENUATED": "Weak signal: dim LED or loose clip",
             "SPIKED": "Sudden jumps: cable knock or contact glitch"}
WINDOW_S, BUDGET_MS = 2.5, 2500.0

ap = argparse.ArgumentParser()
ap.add_argument("--log", default=None)
args, _ = ap.parse_known_args()

st.set_page_config(page_title="Sensor self-diagnosis · live", page_icon="❤", layout="wide",
                   initial_sidebar_state="collapsed")

# ------------------------------------------------------------------ look
st.markdown("""
<style>
:root{--bg:#0b1220;--panel:#111a2e;--line:#22304d;--ink:#e6edf7;--mute:#8da2c0;
      --ok:#34d399;--warn:#fbbf24;--bad:#f87171;--idle:#4b5b78;--acc:#38bdf8}
.stApp{background:radial-gradient(1200px 600px at 10% -10%,#132447 0%,var(--bg) 55%) fixed;color:var(--ink)}
header[data-testid="stHeader"]{background:transparent}
.block-container{padding-top:1.2rem;max-width:1400px}
h1,h2,h3,h4,p,span,div,label{color:var(--ink)}
section[data-testid="stSidebar"]{background:#0e1729}
.hero .card{min-height:318px}
@media (max-width:760px){.hero .card{min-height:0}.big{font-size:48px}}
.stAppDeployButton,[data-testid="stAppDeployButton"]{display:none}
.card{background:linear-gradient(180deg,#121d34,#0f182b);border:1px solid var(--line);border-radius:16px;
      padding:16px 18px;height:100%;box-shadow:0 8px 30px rgba(0,0,0,.25)}
.lbl{font:600 11px/1.2 ui-monospace,Menlo,Consolas,monospace;letter-spacing:.12em;text-transform:uppercase;color:var(--mute)}
.big{font:700 64px/1 system-ui,Segoe UI,Roboto,sans-serif;letter-spacing:-.02em;margin:8px 0 4px}
.unit{font:500 18px system-ui;color:var(--mute);margin-left:6px}
.sub{color:var(--mute);font-size:14px;line-height:1.45}
.pill{display:inline-block;padding:3px 10px;border-radius:999px;font:600 12px system-ui;margin-right:6px;border:1px solid var(--line)}
.top{display:flex;align-items:center;justify-content:space-between;gap:16px;flex-wrap:wrap;margin-bottom:10px}
.top h1{font:700 26px system-ui;margin:0}
.top .tag{color:var(--mute);font-size:14px}
.note{border-left:3px solid var(--acc);background:rgba(56,189,248,.07);padding:8px 12px;border-radius:0 10px 10px 0;
      color:#cfe7f7;font-size:13.5px;margin:8px 0 2px}
.bar{height:8px;border-radius:6px;background:#1c2a46;overflow:hidden;margin:3px 0 8px}
.bar>i{display:block;height:100%}
.row{display:flex;justify-content:space-between;font-size:13px;color:var(--mute)}
.ev{font-size:13.5px;padding:6px 0;border-bottom:1px dashed var(--line);display:flex;gap:10px}
.ev b{font-family:ui-monospace,Consolas,monospace;color:var(--mute);font-weight:500;min-width:58px}
@keyframes flow{to{stroke-dashoffset:-24}}
.flow{stroke-dasharray:6 6;animation:flow .8s linear infinite}
@keyframes pulse{0%{transform:scale(1)}15%{transform:scale(1.12)}30%{transform:scale(1)}}
.heart{display:inline-block;animation:pulse .85s ease-in-out infinite;transform-origin:center}
</style>""", unsafe_allow_html=True)


def chart(c, h=None):
    c = c.configure_view(strokeWidth=0).configure(background="transparent").configure_axis(
        labelColor="#8da2c0", titleColor="#8da2c0", gridColor="#1c2a46", domainColor="#22304d", tickColor="#22304d"
    ).configure_legend(labelColor="#c5d3e8", titleColor="#8da2c0")
    try:
        st.altair_chart(c, width="stretch")
    except TypeError:                                    # older Streamlit
        st.altair_chart(c, use_container_width=True)


# ------------------------------------------------------------------ inputs
logs = sorted(glob.glob("data/sessions/*.jsonl"), key=os.path.getmtime, reverse=True)
default = args.log or (logs[0] if logs else "data/sessions/DEMO.jsonl")
with st.sidebar:
    st.markdown("### Dashboard")
    log_path = st.text_input("Session log", default)
    mode = st.radio("Mode", ["Live", "Replay", "Review"], horizontal=True,
                    help="Live: a session running now. Replay: play a saved session as if live. Review: scrub.")
    speed = st.select_slider("Replay speed", [1, 2, 4, 8], value=2, disabled=mode != "Replay")
    notes = st.toggle("Presenter notes", value=True, help="One line under each panel saying what it proves.")
    if mode == "Replay" and st.button("Restart replay"):
        st.session_state.pop("rp", None)
    gate = json.loads(Path("models/gate.json").read_text()) if Path("models/gate.json").exists() else {}
    TH = float(gate.get("threshold", 0.5))
    st.caption(f"Gate: {gate.get('method', 'none')} · threshold {TH:.3f} · α {gate.get('alpha', '-')}")


def note(text):
    if notes:
        st.markdown(f'<div class="note">{text}</div>', unsafe_allow_html=True)


p = Path(log_path)
status_p = Path(str(p)[:-6] + "_status.json") if str(p).endswith(".jsonl") else None
status = {}
if status_p and status_p.exists():
    try:
        status = json.loads(status_p.read_text())
    except Exception:                                    # noqa: BLE001  (file being replaced)
        status = {}
rows = []
if p.exists():
    for line in p.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except Exception:                                # noqa: BLE001  (half-written last line)
            pass
session = p.stem

# ------------------------------------------------------------------ which window
if mode == "Live":
    i = len(rows) - 1
elif mode == "Replay":
    rp = st.session_state.setdefault("rp", {"file": str(p), "i": 0})
    if rp["file"] != str(p):
        rp.update(file=str(p), i=0)
    i = min(rp["i"], len(rows) - 1)
else:
    i = st.sidebar.slider("Window", 0, max(0, len(rows) - 1), max(0, len(rows) - 1)) if rows else -1

age = time.time() - rows[-1]["t"] if rows else 1e9
if mode == "Live":
    if status.get("state") == "calibrating" and time.time() - status.get("t", 0) < 6:
        state, state_col = "CALIBRATING", "var(--warn)"
    elif rows and age < 8:
        state, state_col = "STREAMING", "var(--ok)"
    elif status.get("state") == "stopped" or rows:
        state, state_col = "NO NEW DATA", "var(--bad)"
    else:
        state, state_col = "WAITING FOR HUB", "var(--idle)"
else:
    state, state_col = ("REPLAY" if mode == "Replay" else "REVIEW"), "var(--acc)"

r = rows[i] if rows and i >= 0 else None
seen = rows[: i + 1] if r else []

# ------------------------------------------------------------------ header
st.markdown(f"""
<div class="top"><div><h1>Is this heart rate trustworthy?</h1>
<div class="tag">Edge-based PPG sensor self-diagnosis · ESP32 node · signed link · hub · certified trust gate</div></div>
<div><span class="pill" style="color:{state_col};border-color:{state_col}">● {state}</span>
<span class="pill">session {session}</span><span class="pill">{len(seen)} windows</span></div></div>""",
            unsafe_allow_html=True)

# ------------------------------------------------------------------ waiting / calibrating screens
if not r:
    calib = status.get("node_s", 0) if status.get("state") == "calibrating" else 0
    pct = min(100, 100 * calib / 125)
    st.markdown(f"""<div class="card"><div class="lbl">Starting up</div>
    <div class="big" style="font-size:40px">{'Calibrating the sensor' if calib else 'Waiting for the first window'}</div>
    <div class="bar" style="height:12px"><i style="width:{pct:.0f}%;background:var(--warn)"></i></div>
    <div class="sub">{'%.0f of about 125 s. Keep the finger still: the node is learning this person\'s normal signal level, '
                      'so every later window is measured against it.' % calib if calib else
                      'Start the hub: <code>python -m phase2.hardware.hub.live --port ... --session ' + session + '</code>'}</div>
    </div>""", unsafe_allow_html=True)
    note("Calibration is per person and per placement: skin tone, finger size and clip pressure all change the raw light level, "
         "so the node fixes its own scale before judging anything.")
    if mode == "Live":
        time.sleep(1.0)
        st.rerun()
    st.stop()

cls = r["node_class"]
trust = r.get("trust")
released = bool(r.get("released"))
hub_cls = r.get("hub_class")
agree_now = hub_cls is None or hub_cls == cls
tele = r.get("telemetry", {})
alarms = r.get("drift_alarms") or []
rejected = int(status.get("rejected", 0)) if mode == "Live" else None


# ------------------------------------------------------------------ the chain ("avatars")
def node(x, title, icon, line1, line2, col):
    return f"""
<g transform="translate({x},0)">
  <circle cx="60" cy="58" r="42" fill="#0e1729" stroke="{col}" stroke-width="3"/>
  <circle cx="60" cy="58" r="50" fill="none" stroke="{col}" stroke-opacity=".25" stroke-width="6"/>
  {icon}
  <text x="60" y="130" text-anchor="middle" fill="#e6edf7" font-size="15" font-weight="700">{title}</text>
  <text x="60" y="150" text-anchor="middle" fill="{col}" font-size="13" font-weight="600">{line1}</text>
  <text x="60" y="168" text-anchor="middle" fill="#8da2c0" font-size="12">{line2}</text>
</g>"""


OK, WARN, BAD, IDLE = "#34d399", "#fbbf24", "#f87171", "#4b5b78"
I_SENSOR = '<rect x="38" y="44" width="44" height="28" rx="8" fill="none" stroke="#e6edf7" stroke-width="2.5"/><circle cx="52" cy="58" r="5" fill="#f87171"/><circle cx="68" cy="58" r="5" fill="#a78bfa"/>'
I_CHIP = '<rect x="40" y="38" width="40" height="40" rx="4" fill="none" stroke="#e6edf7" stroke-width="2.5"/>' + "".join(
    f'<line x1="{x}" y1="31" x2="{x}" y2="38" stroke="#e6edf7" stroke-width="2"/><line x1="{x}" y1="78" x2="{x}" y2="85" stroke="#e6edf7" stroke-width="2"/>'
    for x in (48, 56, 64, 72)) + '<text x="60" y="63" text-anchor="middle" fill="#e6edf7" font-size="12" font-weight="700">AI</text>'
I_LOCK = '<rect x="44" y="54" width="32" height="24" rx="4" fill="none" stroke="#e6edf7" stroke-width="2.5"/><path d="M50 54v-8a10 10 0 0 1 20 0v8" fill="none" stroke="#e6edf7" stroke-width="2.5"/><circle cx="60" cy="66" r="3" fill="#e6edf7"/>'
I_HUB = '<rect x="38" y="42" width="44" height="30" rx="3" fill="none" stroke="#e6edf7" stroke-width="2.5"/><line x1="52" y1="80" x2="68" y2="80" stroke="#e6edf7" stroke-width="2.5"/><line x1="60" y1="72" x2="60" y2="80" stroke="#e6edf7" stroke-width="2.5"/><path d="M45 62l6-6 5 5 8-9 7 6" fill="none" stroke="#38bdf8" stroke-width="2"/>'
I_GATE = '<path d="M60 34l20 8v14c0 13-9 21-20 25-11-4-20-12-20-25V42z" fill="none" stroke="#e6edf7" stroke-width="2.5"/>' + (
    '<path d="M51 58l6 6 12-13" fill="none" stroke="#34d399" stroke-width="3"/>' if released else
    '<path d="M52 50l16 16M68 50L52 66" stroke="#f87171" stroke-width="3"/>')
I_SHOW = '<path d="M60 76c-14-9-22-16-22-25a10 10 0 0 1 22-5 10 10 0 0 1 22 5c0 9-8 16-22 25z" fill="none" stroke="#e6edf7" stroke-width="2.5"/>'

ir = tele.get("dc_level")
pi = tele.get("perfusion_index")
c_sensor = BAD if cls == "ATTENUATED" and not released else (WARN if cls != "CLEAN" else OK)
c_chip = CLASS_COL.get(cls, OK)
c_link = (WARN if rejected else OK)
c_hub = OK if agree_now else WARN
c_gate = OK if released else BAD
c_show = OK if released else IDLE
live_flow = state in ("STREAMING", "REPLAY")
links = "".join(
    f'<line x1="{x}" y1="58" x2="{x + 48}" y2="58" stroke="#38bdf8" stroke-opacity="{.9 if live_flow else .3}" stroke-width="3" '
    f'class="{"flow" if live_flow else ""}"/>' for x in (130, 300, 470, 640, 810))
svg = f"""<svg viewBox="0 0 1000 180" width="100%" style="max-height:200px">{links}
{node(10, "Sensor", I_SENSOR, f"IR {ir / 1000:.0f}k" if ir else "MAX30102", f"PI {100 * pi:.2f}%" if pi else "100 samples/s", c_sensor)}
{node(180, "ESP32", I_CHIP, cls.title(), f"{r.get('node_ms', 0):.0f} ms on-chip" if r.get('node_ms') else "decides on-chip", c_chip)}
{node(350, "Signed link", I_LOCK, "HMAC-SHA256 ✓", f"seq {r['seq']}" + (f" · {rejected} rejected" if rejected else ""), c_link)}
{node(520, "Hub", I_HUB, ("agrees ✓" if agree_now else f"says {str(hub_cls).title()}") if hub_cls else "re-checks", f"{r.get('hub_ms', 0):.0f} ms", c_hub)}
{node(690, "Trust gate", I_GATE, "RELEASE" if released else "WITHHOLD", f"trust {trust:.2f} vs {TH:.2f}" if trust is not None else "no trust model", c_gate)}
{node(860, "Clinician", I_SHOW, f"{r['hr']:.0f} bpm" if released and r.get('hr') else "no number", "shown" if released else "withheld, not wrong", c_show)}
</svg>"""
st.markdown(f'<div class="card">{svg}</div>', unsafe_allow_html=True)
note("Read it left to right: light from the finger is judged <b>on the ESP32 itself</b>, sent over a "
     "<b>signed</b> link, re-checked by the hub, and a heart rate reaches the clinician <b>only if the trust gate lets it through</b>.")

# ------------------------------------------------------------------ hero row
c1, c2, c3 = st.columns([1.05, 1, 1.15])
with c1:
    if released and r.get("hr"):
        body = (f'<div class="big"><span class="heart" style="color:var(--bad)">❤</span> {r["hr"]:.0f}'
                f'<span class="unit">bpm</span></div><div class="sub" style="color:var(--ok)">Released: the signal '
                f'is good enough to trust this number.</div>')
    else:
        body = ('<div class="big" style="color:var(--mute);font-size:48px">— withheld</div>'
                '<div class="sub" style="color:var(--bad)">The device refuses to show a number it cannot back up. '
                'Check the finger and the clip.</div>')
    st.markdown(f'<div class="hero"><div class="card"><div class="lbl">Heart rate</div>{body}</div></div>', unsafe_allow_html=True)

with c2:
    t = 0.0 if trust is None else float(trust)
    ang = np.pi * (1 - t)
    ta = np.pi * (1 - TH)
    x, y = 110 + 85 * np.cos(ang), 105 - 85 * np.sin(ang)
    tx1, ty1 = 110 + 72 * np.cos(ta), 105 - 72 * np.sin(ta)
    tx2, ty2 = 110 + 98 * np.cos(ta), 105 - 98 * np.sin(ta)
    col = OK if released else BAD
    gauge = f"""<svg viewBox="0 0 220 125" width="100%" style="max-height:150px">
<path d="M25 105 A85 85 0 0 1 195 105" fill="none" stroke="#1c2a46" stroke-width="16" stroke-linecap="round"/>
<path d="M25 105 A85 85 0 0 1 {x:.1f} {y:.1f}" fill="none" stroke="{col}" stroke-width="16" stroke-linecap="round"/>
<line x1="{tx1:.1f}" y1="{ty1:.1f}" x2="{tx2:.1f}" y2="{ty2:.1f}" stroke="#e6edf7" stroke-width="3"/>
<text x="{tx2:.1f}" y="{ty2 - 4:.1f}" text-anchor="middle" fill="#8da2c0" font-size="10">gate</text>
<text x="110" y="98" text-anchor="middle" fill="#e6edf7" font-size="38" font-weight="700">{'—' if trust is None else f'{100 * t:.0f}'}</text>
<text x="110" y="118" text-anchor="middle" fill="#8da2c0" font-size="11">trust score / 100</text></svg>"""
    st.markdown(f'<div class="hero"><div class="card"><div class="lbl">Trust score</div>{gauge}'
                f'<div class="sub">White tick = certified gate ({100 * TH:.0f}). Above it, released; below it, withheld.</div></div></div>',
                unsafe_allow_html=True)

with c3:
    prob = r.get("node_proba") or [0] * 5
    bars = "".join(
        f'<div class="row"><span>{c.title()}</span><span>{100 * float(v):.0f}%</span></div>'
        f'<div class="bar"><i style="width:{100 * float(v):.0f}%;background:{CLASS_COL[c]}"></i></div>'
        for c, v in zip(CLASSES, prob))
    st.markdown(f'<div class="hero"><div class="card"><div class="lbl">Sensor condition · decided on the ESP32</div>'
                f'<div style="font:700 26px system-ui;margin:6px 0 2px;color:{CLASS_COL.get(cls)}">{cls.title()}</div>'
                f'<div class="sub" style="margin-bottom:8px">{CLASS_TXT.get(cls, "")}</div>{bars}</div></div>',
                unsafe_allow_html=True)
note("Two different questions: the <b>chip</b> names what is wrong with the sensor (5 classes); the <b>trust score</b> "
     "answers whether this particular heart rate would be within the medical tolerance (5 bpm or 10%).")

# ------------------------------------------------------------------ signal + timeline
c1, c2 = st.columns([1, 1.6])
with c1:
    st.markdown('<div class="lbl" style="margin-top:14px">Last 5 seconds of pulse (what the chip saw)</div>',
                unsafe_allow_html=True)
    if r.get("window"):
        w = np.asarray(r["window"], dtype=float)
        dfw = pd.DataFrame({"t (s)": np.arange(w.size) / 100.0, "pulse": w})
        chart(alt.Chart(dfw).mark_line(color=CLASS_COL.get(cls, OK), strokeWidth=2).encode(
            x=alt.X("t (s)", axis=alt.Axis(tickCount=5)), y=alt.Y("pulse", axis=alt.Axis(title=None, labels=False))
        ).properties(height=210))
    else:
        st.info("Raw windows were not stored for this session (--no-window).")
with c2:
    st.markdown('<div class="lbl" style="margin-top:14px">Whole session · trust, gate and sensor condition</div>',
                unsafe_allow_html=True)
    t0 = seen[0].get("t_node_s", 0)
    df = pd.DataFrame([{"t (s)": round(x.get("t_node_s", 0) - t0, 1), "trust": x.get("trust"),
                        "released": "shown" if x.get("released") else "withheld", "condition": x["node_class"],
                        "hr": x.get("hr")} for x in seen])
    base = alt.Chart(df).encode(x=alt.X("t (s):Q", title="seconds since first window"))
    df["t_end"] = df["t (s)"] + 2.5
    strip = alt.Chart(df).mark_rect(opacity=.95).encode(
        x="t (s):Q", x2="t_end:Q", y=alt.datum(-0.13), y2=alt.datum(-0.04),
        color=alt.Color("condition:N", scale=alt.Scale(domain=CLASSES, range=[CLASS_COL[c] for c in CLASSES]),
                        legend=alt.Legend(orient="top", title=None)))
    line = base.mark_line(color="#38bdf8", strokeWidth=2).encode(
        y=alt.Y("trust:Q", scale=alt.Scale(domain=[-0.15, 1.02]), title="trust",
                axis=alt.Axis(values=[0, .25, .5, .75, 1])))
    dots = base.mark_circle(size=28).encode(
        y="trust:Q", fill=alt.Fill("released:N", scale=alt.Scale(domain=["shown", "withheld"], range=[OK, BAD]),
                                    legend=alt.Legend(orient="top", title=None)),
        tooltip=["t (s)", "trust", "released", "condition", "hr"])
    rule = alt.Chart(pd.DataFrame({"y": [TH]})).mark_rule(color="#e6edf7", strokeDash=[5, 4]).encode(y="y:Q")
    chart(alt.layer(strip, line, dots, rule).properties(height=230))
note("Every dot is one 5-second window. Green dots were shown, red were withheld. The dashed line is the threshold "
     "certified on 13 unseen hospital recordings, so that at most about 5% of the numbers shown are wrong.")

# ------------------------------------------------------------------ drift + system health + events
c1, c2, c3 = st.columns([1.3, 1, 1.1])
with c1:
    st.markdown('<div class="lbl" style="margin-top:14px">Slow drift monitor (CUSUM on sensor telemetry)</div>',
                unsafe_allow_html=True)
    tl = []
    for x in seen:
        for k in ("dc_level", "perfusion_index", "signal_amplitude"):
            v = (x.get("telemetry") or {}).get(k)
            if v is not None:
                tl.append({"t (s)": round(x.get("t_node_s", 0) - t0, 1), "stream": k, "value": v,
                           "alarm": k in (x.get("drift_alarms") or [])})
    if tl:
        dt = pd.DataFrame(tl)
        dt["norm"] = dt.groupby("stream")["value"].transform(lambda s: (s - s.min()) / ((s.max() - s.min()) or 1))
        b = alt.Chart(dt).encode(x=alt.X("t (s):Q", title=None))
        ln = b.mark_line(strokeWidth=1.6).encode(y=alt.Y("norm:Q", title=None, axis=alt.Axis(labels=False)),
                                                  color=alt.Color("stream:N", legend=alt.Legend(orient="top", title=None),
                                                                  scale=alt.Scale(range=["#38bdf8", "#a78bfa", "#34d399"])))
        al = b.transform_filter("datum.alarm").mark_point(shape="triangle-up", size=120, filled=True, color=BAD).encode(y="norm:Q")
        chart(alt.layer(ln, al).properties(height=170))
    st.markdown(f'<div class="sub">{"<b style=color:var(--bad)>DRIFT ALARM: " + ", ".join(alarms) + "</b> · re-seat the clip and recalibrate" if alarms else "No drift alarm in this window."}</div>',
                unsafe_allow_html=True)
with c2:
    agree = [x["node_class"] == x.get("hub_class") for x in seen if x.get("hub_class")]
    node_ms = [x.get("node_ms") for x in seen if x.get("node_ms")]
    hub_ms = [x.get("hub_ms") for x in seen if x.get("hub_ms") is not None]
    rel = np.mean([bool(x.get("released")) for x in seen])
    med_node = float(np.median(node_ms)) if node_ms else None
    items = [("Readings shown", f"{rel:.0%}", "of all windows"),
             ("Chip = hub", f"{np.mean(agree):.0%}" if agree else "n/a", "live parity check"),
             ("ESP32 time / window", f"{med_node:.0f} ms" if med_node else "n/a",
              f"{100 * med_node / BUDGET_MS:.0f}% of the 2.5 s budget" if med_node else "benchmark: 298 ms"),
             ("Hub time / window", f"{np.median(hub_ms):.0f} ms" if hub_ms else "n/a", "Pi or laptop"),
             ("Rejected packets", str(rejected) if rejected is not None else "—", "tampered or replayed")]
    html = "".join(f'<div class="row" style="padding:7px 0;border-bottom:1px solid var(--line)"><span>{a}<br>'
                   f'<small style="color:#5f7394">{c}</small></span><span style="font:700 20px system-ui;color:var(--ink)">{b}</span></div>'
                   for a, b, c in items)
    st.markdown(f'<div class="card" style="margin-top:14px"><div class="lbl">System health</div>{html}</div>',
                unsafe_allow_html=True)
with c3:
    ev, prev = [], None
    for x in seen:
        ts = f"{x.get('t_node_s', 0) - t0:6.0f}s"
        if prev is None or x["node_class"] != prev["node_class"]:
            ev.append((ts, f'<span style="color:{CLASS_COL.get(x["node_class"])}">●</span> sensor now <strong>{x["node_class"].title()}</strong>'))
        if prev is not None and bool(x.get("released")) != bool(prev.get("released")):
            ev.append((ts, "✅ heart rate shown again" if x.get("released") else "⛔ heart rate withheld"))
        if x.get("drift_alarms"):
            ev.append((ts, "⚠️ drift alarm: " + ", ".join(x["drift_alarms"])))
        prev = x
    html = "".join(f'<div class="ev"><b>{a}</b><span>{b}</span></div>' for a, b in ev[-9:][::-1])
    st.markdown(f'<div class="card" style="margin-top:14px"><div class="lbl">What just happened</div>{html}</div>',
                unsafe_allow_html=True)
note("The drift monitor watches slow changes no single window can see (an LED ageing, a clip loosening over minutes). "
     "System health shows the edge AI fits comfortably inside the 2.5 s it has per window.")

# ------------------------------------------------------------------ hospital export
with st.expander("Hospital record export (HL7 FHIR Observation)"):
    obs = {"resourceType": "Observation", "status": "final" if released else "cancelled",
           "code": {"coding": [{"system": "http://loinc.org", "code": "8867-4", "display": "Heart rate"}]},
           "effectiveDateTime": datetime.fromtimestamp(r["t"], timezone.utc).isoformat(),
           "component": [{"code": {"text": "Signal trust score"},
                          "valueQuantity": {"value": None if trust is None else round(100 * trust, 1), "unit": "score"}},
                         {"code": {"text": "Sensor condition (edge)"}, "valueString": cls}]}
    if released and r.get("hr"):
        obs["valueQuantity"] = {"value": r["hr"], "unit": "beats/minute", "system": "http://unitsofmeasure.org", "code": "/min"}
    else:
        obs["dataAbsentReason"] = {"text": "Withheld: signal trust below the certified threshold"}
    st.code(json.dumps(obs, indent=2), language="json")
    st.download_button("Download this reading", json.dumps(obs, indent=2), file_name=f"reading_{r['seq']}.json",
                       mime="application/json")
    st.caption("A withheld reading is exported as 'cancelled' with a reason, never as a wrong number.")

# ------------------------------------------------------------------ keep moving
if mode == "Live":
    time.sleep(1.0)
    st.rerun()
elif mode == "Replay" and i < len(rows) - 1:
    time.sleep(WINDOW_S / speed)
    st.session_state["rp"]["i"] = i + 1
    st.rerun()
