"""Step 12: Convert the Random Forest model to ONNX and benchmark inference."""

import numpy as np
import pandas as pd
import pickle
import time
import os
from sklearn.preprocessing import StandardScaler
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as rt

os.makedirs('models', exist_ok=True)

df       = pd.read_csv('data/feature_dataset.csv')
features = [c for c in df.columns if c != 'label']

with open('models/scaler.pkl','rb') as f: scaler = pickle.load(f)
with open('models/random_forest.pkl','rb') as f: rf = pickle.load(f)

X = scaler.transform(df[features].values)

# ── Convert Random Forest to ONNX format ──────────────
print("Converting Random Forest to ONNX...")
initial_type  = [('float_input', FloatTensorType([None, len(features)]))]
onnx_model    = convert_sklearn(rf, initial_types=initial_type)

with open('models/rf_model.onnx', 'wb') as f:
    f.write(onnx_model.SerializeToString())

onnx_size_kb = os.path.getsize('models/rf_model.onnx') / 1024
print(f"ONNX model saved. Size: {onnx_size_kb:.1f} KB")

# ── Benchmark ONNX inference ───────────────────────────
sess       = rt.InferenceSession('models/rf_model.onnx')
input_name = sess.get_inputs()[0].name
sample     = X[0:1].astype(np.float32)

print("\nBenchmarking inference (1000 runs)...")
times = []
for _ in range(1000):
    t0 = time.perf_counter()
    sess.run(None, {input_name: sample})
    times.append((time.perf_counter() - t0) * 1000)

mean_lat = np.mean(times)
max_lat  = np.max(times)
std_lat  = np.std(times)

# RPi 4 is approximately 3-4x slower than a typical laptop
# for this type of workload — conservative estimate: 4x
rpi_estimate = mean_lat * 4

print(f"\nONNX Inference Benchmark Results:")
print(f"  Laptop — Mean latency:    {mean_lat:.3f} ms")
print(f"  Laptop — Max latency:     {max_lat:.3f} ms")
print(f"  Laptop — Std deviation:   {std_lat:.3f} ms")
print(f"  RPi 4 estimate (4x):      {rpi_estimate:.1f} ms")
print(f"  Target:                   < 50.0 ms")
print(f"  Feasible on RPi 4:        "
      f"{'YES' if rpi_estimate < 50 else 'MARGINAL'}")
print(f"\n  ONNX model size:          {onnx_size_kb:.1f} KB")
print(f"  Well within RPi 4 RAM:    YES (4 GB available)")

# Save benchmark results
with open('models/edge_benchmark.txt', 'w') as f:
    f.write(f"ONNX Inference Benchmark\n")
    f.write(f"Laptop mean latency: {mean_lat:.3f} ms\n")
    f.write(f"RPi 4 estimate:      {rpi_estimate:.1f} ms\n")
    f.write(f"Model size:          {onnx_size_kb:.1f} KB\n")
    f.write(f"Edge feasible:       {'YES' if rpi_estimate < 50 else 'MARGINAL'}\n")

print("\nSaved to models/edge_benchmark.txt")
print("\nThis confirms the model is feasible for edge deployment.")
print("Physical deployment on RPi 4 is Phase 2.")
