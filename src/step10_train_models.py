"""Step 10: Train baseline, Random Forest, and SVM classifiers."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import time
import os
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, f1_score)

os.makedirs('models', exist_ok=True)
os.makedirs('figures', exist_ok=True)

# ── Load feature dataset ───────────────────────────────
df  = pd.read_csv('data/feature_dataset.csv')
features    = [c for c in df.columns if c != 'label']
X           = df[features].values
y           = df['label'].values
class_names = ['CLEAN','NOISY','DRIFTED','ATTENUATED','SPIKED']

print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")

# ── Train / Validation / Test Split ───────────────────
# 70% train, 15% validation, 15% test
# Stratified = each split has same proportion of each class

sss1 = StratifiedShuffleSplit(n_splits=1, 
                               test_size=0.30, 
                               random_state=42)
train_idx, temp_idx = next(sss1.split(X, y))

sss2 = StratifiedShuffleSplit(n_splits=1, 
                               test_size=0.50, 
                               random_state=42)
val_idx, test_idx = next(sss2.split(X[temp_idx], y[temp_idx]))

X_train = X[train_idx]
y_train = y[train_idx]
X_val   = X[temp_idx][val_idx]
y_val   = y[temp_idx][val_idx]
X_test  = X[temp_idx][test_idx]
y_test  = y[temp_idx][test_idx]

print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# ── Scale features ─────────────────────────────────────
# StandardScaler brings all features to same scale
# so no single feature dominates just because its values are large

scaler      = StandardScaler()
X_train_s   = scaler.fit_transform(X_train)
X_val_s     = scaler.transform(X_val)
X_test_s    = scaler.transform(X_test)

# Save scaler — needed later during inference
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("Scaler saved.")

# ════════════════════════════════════════════════════════
# CLASSIFIER 1 — THRESHOLD RULES (Baseline)
# ════════════════════════════════════════════════════════

print("\n" + "="*50)
print("CLASSIFIER 1: Threshold-Based Rules")
print("="*50)

# Look at feature statistics to set thresholds
print("\nFeature means per class (to help set thresholds):")
for feat in ['variance', 'kurtosis', 'power_ratio', 
             'peak_to_peak', 'mean']:
    print(f"\n  {feat}:")
    df_train = pd.DataFrame(X_train, columns=features)
    df_train['label'] = y_train
    for cls, name in enumerate(class_names):
        vals = df_train[df_train['label']==cls][feat]
        print(f"    {name}: mean={vals.mean():.3f}")

def threshold_predict(X_feat, feature_names):
    predictions = []
    for row in X_feat:
        feat = dict(zip(feature_names, row))
        
        # Rules based on which features are most discriminative
        if feat['kurtosis'] > 4.5:
            pred = 4  # SPIKED — spikes create high kurtosis
        elif feat['variance'] > 1.8:
            pred = 1  # NOISY — noise increases variance
        elif abs(feat['mean']) > 0.4:
            pred = 2  # DRIFTED — drift shifts the mean
        elif feat['peak_to_peak'] < 0.6:
            pred = 3  # ATTENUATED — attenuation reduces amplitude
        else:
            pred = 0  # CLEAN
        predictions.append(pred)
    return np.array(predictions)

# Calibrate thresholds on validation set
# (adjust the numbers above based on the printed means)
y_thresh_pred = threshold_predict(X_test_s, features)
thresh_acc    = accuracy_score(y_test, y_thresh_pred)
thresh_f1     = f1_score(y_test, y_thresh_pred, average='macro')

print(f"\nThreshold Rules — Test Accuracy: {thresh_acc:.4f}")
print(f"Threshold Rules — Macro F1:      {thresh_f1:.4f}")
print(classification_report(y_test, y_thresh_pred,
                             target_names=class_names))

# ════════════════════════════════════════════════════════
# CLASSIFIER 2 — RANDOM FOREST
# ════════════════════════════════════════════════════════

print("\n" + "="*50)
print("CLASSIFIER 2: Random Forest")
print("="*50)

rf = RandomForestClassifier(
    n_estimators=50,
    max_depth=10,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

print("Training Random Forest...")
rf.fit(X_train_s, y_train)

# Validation check
y_val_rf = rf.predict(X_val_s)
print(f"Validation Accuracy: {accuracy_score(y_val, y_val_rf):.4f}")

# Test evaluation
y_test_rf = rf.predict(X_test_s)
rf_acc    = accuracy_score(y_test, y_test_rf)
rf_f1     = f1_score(y_test, y_test_rf, average='macro')

# Inference time — test on single sample 1000 times
times = []
for _ in range(1000):
    t0 = time.perf_counter()
    rf.predict(X_test_s[0:1])
    times.append((time.perf_counter() - t0) * 1000)
rf_latency = np.mean(times)

print(f"Test Accuracy: {rf_acc:.4f}")
print(f"Macro F1:      {rf_f1:.4f}")
print(f"Latency:       {rf_latency:.3f} ms per window")
print(classification_report(y_test, y_test_rf,
                             target_names=class_names))

# Feature importance
print("\nTop 5 most important features:")
importances = rf.feature_importances_
for name, imp in sorted(zip(features, importances),
                        key=lambda x: x[1], reverse=True)[:5]:
    print(f"  {name}: {imp:.4f}")

with open('models/random_forest.pkl', 'wb') as f:
    pickle.dump(rf, f)
print("Random Forest saved.")

# ════════════════════════════════════════════════════════
# CLASSIFIER 3 — SVM
# ════════════════════════════════════════════════════════

print("\n" + "="*50)
print("CLASSIFIER 3: SVM (RBF Kernel)")
print("="*50)

svm = SVC(kernel='rbf', C=1.0, gamma='scale',
          probability=True, class_weight='balanced',
          random_state=42)

print("Training SVM... (this takes longer than RF)")
svm.fit(X_train_s, y_train)

y_test_svm = svm.predict(X_test_s)
svm_acc    = accuracy_score(y_test, y_test_svm)
svm_f1     = f1_score(y_test, y_test_svm, average='macro')

times = []
for _ in range(1000):
    t0 = time.perf_counter()
    svm.predict(X_test_s[0:1])
    times.append((time.perf_counter() - t0) * 1000)
svm_latency = np.mean(times)

print(f"Test Accuracy: {svm_acc:.4f}")
print(f"Macro F1:      {svm_f1:.4f}")
print(f"Latency:       {svm_latency:.3f} ms per window")
print(classification_report(y_test, y_test_svm,
                             target_names=class_names))

with open('models/svm.pkl', 'wb') as f:
    pickle.dump(svm, f)
print("SVM saved.")

# ════════════════════════════════════════════════════════
# RESULTS TABLE
# ════════════════════════════════════════════════════════

import os
rf_size  = os.path.getsize('models/random_forest.pkl') / 1024
svm_size = os.path.getsize('models/svm.pkl') / 1024

print("\n" + "="*70)
print("FINAL RESULTS TABLE")
print("="*70)
print(f"{'Model':<22} {'Accuracy':>10} {'Macro F1':>10} "
      f"{'Latency(ms)':>13} {'Size(KB)':>10}")
print("-"*70)
print(f"{'Threshold Rules':<22} {thresh_acc:>10.4f} {thresh_f1:>10.4f} "
      f"{'<1':>13} {'0':>10}")
print(f"{'Random Forest':<22} {rf_acc:>10.4f} {rf_f1:>10.4f} "
      f"{rf_latency:>13.3f} {rf_size:>10.1f}")
print(f"{'SVM (RBF)':<22} {svm_acc:>10.4f} {svm_f1:>10.4f} "
      f"{svm_latency:>13.3f} {svm_size:>10.1f}")
print("="*70)

# Save results to file
results_text = (
    f"Threshold Rules: Acc={thresh_acc:.4f} F1={thresh_f1:.4f}\n"
    f"Random Forest:   Acc={rf_acc:.4f}   F1={rf_f1:.4f}   "
    f"Lat={rf_latency:.3f}ms  Size={rf_size:.1f}KB\n"
    f"SVM RBF:         Acc={svm_acc:.4f}   F1={svm_f1:.4f}   "
    f"Lat={svm_latency:.3f}ms  Size={svm_size:.1f}KB\n"
)
with open('models/results.txt', 'w') as f:
    f.write(results_text)
print("\nResults saved to models/results.txt")
