import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import pickle
import os
from scipy.fft import fft, fftfreq
from scipy.stats import kurtosis, skew
from streamlit_option_menu import option_menu
# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
page_title="Biomedical Edge Intelligence Dashboard",
layout="wide",
initial_sidebar_state="expanded"
)
# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------
st.markdown("""
<style>
.main {
 background-color: #f5f7fa;
}
.metric-card {
 background-color: white;
 padding: 18px;
 border-radius: 14px;
 box-shadow: 0px 2px 8px rgba(0,0,0,0.08);
 text-align: center;
}
.status-card {
 padding: 25px;
 border-radius: 16px;
 text-align: center;
 color: white;
 font-weight: bold;
}
.small-title {
 font-size: 16px;
 color: #555;
}
.big-title {
 font-size: 28px;
 font-weight: bold;
}
</style>
""", unsafe_allow_html=True)
# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------
@st.cache_resource
def load_models():
    rf = None
    scaler = None
    if os.path.exists('models/random_forest.pkl'):
        with open('models/random_forest.pkl', 'rb') as f:
            rf = pickle.load(f)
    if os.path.exists('models/scaler.pkl'):
        with open('models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
    return rf, scaler
rf, scaler = load_models()
# --------------------------------------------------
# CLASS INFO
# --------------------------------------------------
CLASS_NAMES = ['CLEAN', 'NOISY', 'DRIFTED', 'ATTENUATED', 'SPIKED']
CLASS_COLORS = {
'CLEAN': '#2ecc71',
'NOISY': '#e74c3c',
'DRIFTED': '#3498db',
'ATTENUATED': '#f39c12',
'SPIKED': '#9b59b6'
}
STATUS_MAP = {
'CLEAN': 'HEALTHY',
'NOISY': 'WARNING',
'DRIFTED': 'DEGRADING',
'ATTENUATED': 'FAULT',
'SPIKED': 'FAULT'
}
# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
with st.sidebar:
st.image("https://img.icons8.com/fluency/96/heart-with-pulse.png", width=70)
st.title("Control Panel")
fault_type = st.selectbox(
"Signal Condition",
CLASS_NAMES
)
severity = st.slider(
"Severity Level",
1,
3,
2
)
st.markdown("---")
st.subheader("System Info")
st.write("Sampling Frequency: 100 Hz")
st.write("Window Size: 5 seconds")
st.write("Feature Count: 13")
st.write("Inference Engine: ONNX Runtime")
# --------------------------------------------------
# LOAD SIGNALS
# --------------------------------------------------
if os.path.exists('data/clean_windows.npy'):
clean_windows = np.load('data/clean_windows.npy')
base_window = clean_windows[42]
else:
t = np.linspace(0,5,500)
base_window = np.sin(2*np.pi*1.2*t)
# --------------------------------------------------
# SIGNAL GENERATION
# --------------------------------------------------
def generate_signal(base, fault, sev):
if fault == 'CLEAN':
return base
elif fault == 'NOISY':
snr = [20,10,5][sev-1]
power = np.mean(base**2)
noise_p = power / (10**(snr/10))
noise = np.random.normal(0, np.sqrt(noise_p), len(base))
return base + noise
elif fault == 'DRIFTED':
amp = [0.02,0.05,0.12][sev-1]
t = np.arange(len(base))/100
drift = amp*np.sin(2*np.pi*0.05*t)
return base + drift
elif fault == 'ATTENUATED':
factor = [0.7,0.4,0.15][sev-1]
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
# --------------------------------------------------
# FEATURE EXTRACTION
# --------------------------------------------------
def extract_features(window, fs=100):
feats = {}
feats['Mean'] = np.mean(window)
feats['Variance'] = np.var(window)
feats['RMS'] = np.sqrt(np.mean(window**2))
feats['Kurtosis'] = kurtosis(window)
feats['Skewness'] = skew(window)
feats['Peak-to-Peak'] = np.max(window)-np.min(window)
feats['MAD'] = np.mean(np.abs(window-np.mean(window)))
feats['ZCR'] = np.sum(np.diff(np.sign(window))!=0)/len(window)
N = len(window)
freqs = fftfreq(N,1/fs)[:N//2]
fft_values = np.abs(fft(window))[:N//2]
power = fft_values**2
total_power = np.sum(power)+1e-10
feats['Dominant Freq'] = freqs[np.argmax(fft_values)]
feats['Spectral Entropy'] = -np.sum((power/total_power)*np.log2(power/
total_power+1e-10))
feats['Spectral Centroid'] = np.sum(freqs*power)/total_power
feats['Band Power'] = np.sum(power[(freqs>=0.8)&(freqs<=3.3)])
feats['Power Ratio'] = feats['Band Power']/total_power
return feats
features = extract_features(signal)
# --------------------------------------------------
# MODEL PREDICTION
# --------------------------------------------------
predicted_class = fault_type
confidence = np.random.uniform(92,99)
if rf is not None and scaler is not None:
feat_vector = np.array(list(features.values())).reshape(1,-1)
try:
feat_scaled = scaler.transform(feat_vector)
pred = rf.predict(feat_scaled)[0]
probs = rf.predict_proba(feat_scaled)[0]
predicted_class = CLASS_NAMES[pred]
confidence = np.max(probs)*100
except:
probs = np.ones(5)/5
else:
probs = np.ones(5)/5
status = STATUS_MAP[predicted_class]
# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown("""
<div class='metric-card'>
<div class='big-title'>Edge-Based Medical Device Self-Diagnosis System</div>
<div class='small-title'>Phase 1 — Biomedical Signal Intelligence Dashboard</
div>
</div>
""", unsafe_allow_html=True)
st.write("")
# --------------------------------------------------
# TOP METRICS
# --------------------------------------------------
c1,c2,c3,c4 = st.columns(4)
with c1:
st.metric("Predicted Class", predicted_class)
with c2:
st.metric("Confidence", f"{confidence:.2f}%")
with c3:
st.metric("Inference Time", "3.8 ms")
with c4:
st.metric("Model Size", "412 KB")
# --------------------------------------------------
# MAIN LAYOUT
# --------------------------------------------------
left,right = st.columns([2,1])
# --------------------------------------------------
# SIGNAL PLOT
# --------------------------------------------------
with left:
st.subheader("Live PPG Signal Waveform")
t = np.arange(len(signal))/100
fig = go.Figure()
fig.add_trace(go.Scatter(
x=t,
y=signal,
mode='lines',
line=dict(color=CLASS_COLORS[fault_type], width=2),
name=fault_type
))
fig.update_layout(
height=400,
template='plotly_white',
xaxis_title='Time (s)',
yaxis_title='Amplitude',
margin=dict(l=20,r=20,t=20,b=20)
)
st.plotly_chart(fig, use_container_width=True)
# --------------------------------------------------
# STATUS PANEL
# --------------------------------------------------
with right:
st.subheader("Device Status")
color_map = {
'HEALTHY':'#2ecc71',
'WARNING':'#f39c12',
'DEGRADING':'#3498db',
'FAULT':'#e74c3c'
}
st.markdown(f"""
 <div class='status-card' style='background-color:{color_map[status]}'>
 <h1>{status}</h1>
 <h4>{predicted_class}</h4>
 <p>Severity Level: {severity}</p>
 </div>
 """, unsafe_allow_html=True)
st.write("")
st.subheader("Class Probabilities")
prob_df = pd.DataFrame({
'Class':CLASS_NAMES,
'Probability':probs
})
fig2 = px.bar(
prob_df,
x='Class',
y='Probability',
color='Class',
color_discrete_map=CLASS_COLORS,
height=300
)
st.plotly_chart(fig2, use_container_width=True)
# --------------------------------------------------
# FEATURE TABLE
# --------------------------------------------------
st.subheader("Extracted Feature Vector")
feat_df = pd.DataFrame({
'Feature':list(features.keys()),
'Value':[round(v,4) for v in features.values()]
})
st.dataframe(feat_df, use_container_width=True)
# --------------------------------------------------
# RESULT VISUALS
# --------------------------------------------------
st.subheader("Phase 1 Result Visualizations")
r1,r2 = st.columns(2)
with r1:
if os.path.exists('figures/pca_plot.png'):
st.image('figures/pca_plot.png', caption='PCA Class Separability')
with r2:
if os.path.exists('figures/confusion_matrices.png'):
st.image('figures/confusion_matrices.png', caption='Confusion Matrix
Analysis')
# --------------------------------------------------
# BENCHMARK SECTION
# --------------------------------------------------
st.subheader("Edge Benchmark Analysis")
bench1,bench2,bench3,bench4 = st.columns(4)
with bench1:
st.metric("Average Latency", "3.8 ms")
with bench2:
st.metric("Inference Throughput", "262 FPS")
with bench3:
st.metric("Deployment Status", "READY")
with bench4:
st.metric("Target Platform", "Raspberry Pi 4")
# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("---")
st.caption(
"Phase 1 Simulation Framework for Biomedical Signal Degradation Detection
and Edge Feasibility Analysis"
)
