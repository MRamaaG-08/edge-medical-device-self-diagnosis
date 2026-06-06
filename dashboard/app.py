"""Streamlit dashboard for interactive PPG degradation simulation and inference.

Run from the repository root with:
    streamlit run dashboard/app.py
"""

from pathlib import Path

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pickle
from scipy.signal import butter, filtfilt
from scipy.fft import fft, fftfreq
from scipy.stats import kurtosis, skew

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

# Load models
with open(MODELS_DIR / 'random_forest.pkl', 'rb') as f:
    rf = pickle.load(f)
with open(MODELS_DIR / 'scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

CLASS_NAMES   = ['CLEAN', 'NOISY', 'DRIFTED', 'ATTENUATED', 'SPIKED']
STATUS_MAP    = {0: 'HEALTHY', 1: 'DEGRADING', 2: 'DEGRADING',
                 3: 'FAULT', 4: 'FAULT'}
STATUS_COLOR  = {'HEALTHY': '#2ecc71', 'DEGRADING': '#f39c12',
                 'FAULT': '#e74c3c'}
CLASS_COLORS  = ['#2ecc71','#e74c3c','#3498db','#f39c12','#9b59b6']

st.set_page_config(page_title='Edge Medical Device Self-Diagnosis',
                   layout='wide')

st.title('Edge-Based Medical Device Self-Diagnosis System')
st.caption('Phase 1 — Simulation and Benchmarking Study')

# Sidebar controls
st.sidebar.header('Signal Controls')
fault_type = st.sidebar.selectbox(
    'Select Signal Condition',
    ['CLEAN', 'NOISY', 'DRIFTED', 'ATTENUATED', 'SPIKED']
)
severity = st.sidebar.slider('Severity Level', 1, 3, 2)

# Generate signal based on selection
clean_windows = np.load(DATA_DIR / 'clean_windows.npy')
base_window   = clean_windows[42]

def generate_signal(base, fault, sev):
    if fault == 'CLEAN':
        return base
    elif fault == 'NOISY':
        snr = [20, 10, 5][sev-1]
        power = np.mean(base**2)
        noise_p = power / (10**(snr/10))
        return base + np.random.normal(0, np.sqrt(noise_p), len(base))
    elif fault == 'DRIFTED':
        amp = [0.02, 0.05, 0.12][sev-1]
        t = np.arange(len(base)) / 100
        return base + amp * np.sin(2*np.pi*0.05*t)
    elif fault == 'ATTENUATED':
        factor = [0.7, 0.4, 0.15][sev-1]
        return base * factor
    elif fault == 'SPIKED':
        prob, amp = [(0.01,5),(0.02,10),(0.05,15)][sev-1]
        sig = base.copy()
        std = np.std(base)
        for i in range(len(sig)):
            if np.random.random() < prob:
                sig[i] += amp * std * np.random.choice([-1,1])
        return sig

signal = generate_signal(base_window, fault_type, severity)

# Extract features
def extract_features(window, fs=100):
    feats = {}
    feats['mean']          = float(np.mean(window))
    feats['variance']      = float(np.var(window))
    feats['rms']           = float(np.sqrt(np.mean(window**2)))
    feats['kurtosis']      = float(kurtosis(window))
    feats['zcr']           = float(np.sum(np.diff(np.sign(window))!=0)/len(window))
    feats['peak_to_peak']  = float(np.max(window)-np.min(window))
    feats['skewness']      = float(skew(window))
    feats['mad']           = float(np.mean(np.abs(window-np.mean(window))))
    N = len(window)
    freqs = fftfreq(N, 1/fs)[:N//2]
    fft_v = np.abs(fft(window))[:N//2]
    power = fft_v**2
    total = np.sum(power)+1e-10
    ppg_m = (freqs>=0.8)&(freqs<=3.3)
    feats['dominant_freq']    = float(freqs[np.argmax(fft_v)])
    feats['spectral_entropy'] = float(-np.sum((power/total)*np.log2(power/total+1e-10)))
    feats['ppg_band_power']   = float(np.sum(power[ppg_m]))
    feats['power_ratio']      = float(np.sum(power[ppg_m])/total)
    feats['spectral_centroid']= float(np.sum(freqs*power)/total)
    return feats

feats      = extract_features(signal)
feat_vec   = np.array(list(feats.values())).reshape(1,-1)
feat_scaled= scaler.transform(feat_vec)
pred_class = int(rf.predict(feat_scaled)[0])
pred_proba = rf.predict_proba(feat_scaled)[0]
status     = STATUS_MAP[pred_class]

# ── Layout ────────────────────────────────────────────
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader('Live Signal Waveform')
    time_ax = np.arange(len(signal)) / 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time_ax, y=signal,
        line=dict(color=CLASS_COLORS[CLASS_NAMES.index(fault_type)], width=1.5),
        name=fault_type
    ))
    fig.update_layout(
        xaxis_title='Time (seconds)',
        yaxis_title='Amplitude',
        height=300,
        margin=dict(l=20,r=20,t=20,b=40),
        plot_bgcolor='#f8f9fa'
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader('Device Status')
    color = STATUS_COLOR[status]
    st.markdown(
        f'<div style="background:{color};padding:20px;border-radius:10px;'
        f'text-align:center;">'
        f'<h1 style="color:white;margin:0">{status}</h1>'
        f'<p style="color:white;margin:5px 0">Predicted: {CLASS_NAMES[pred_class]}</p>'
        f'<p style="color:white;margin:0">Confidence: {pred_proba[pred_class]*100:.1f}%</p>'
        f'</div>',
        unsafe_allow_html=True
    )
    st.markdown('')
    st.subheader('Class Probabilities')
    prob_df = pd.DataFrame({
        'Class': CLASS_NAMES,
        'Probability': pred_proba
    })
    st.bar_chart(prob_df.set_index('Class'))

# ── Feature values ────────────────────────────────────
st.subheader('Extracted Feature Vector (13 values)')
feat_df = pd.DataFrame([feats]).T
feat_df.columns = ['Value']
feat_df['Value'] = feat_df['Value'].round(4)
st.dataframe(feat_df, use_container_width=True)

# ── Results section ───────────────────────────────────
st.divider()
st.subheader('Phase 1 Benchmark Results')

results_col1, results_col2, results_col3 = st.columns(3)
with results_col1:
    st.metric('Best Model', 'Random Forest')
with results_col2:
    st.metric('Test Accuracy', '-- %')
with results_col3:
    st.metric('Macro F1', '--')

st.caption('Fill in your actual numbers from models/results.txt')
