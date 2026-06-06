# Edge-Based Medical Device Self-Diagnosis System

## Overview

This Phase 1 research project detects PPG sensor degradation conditions using signal processing, feature extraction, machine learning, and edge-inference benchmarking. The system builds a labeled dataset from clean PPG windows and simulated degradation classes, trains lightweight classifiers, and evaluates whether the best model is suitable for deployment on edge hardware.

The five signal classes are `CLEAN`, `NOISY`, `DRIFTED`, `ATTENUATED`, and `SPIKED`.

## Problem Statement

Wearable and remote medical devices often rely on PPG signals for heart-rate and oxygen-saturation monitoring. PPG quality can degrade because of noise, baseline drift, weak sensor contact, attenuation, ambient interference, or sudden spikes. If downstream systems process degraded signals without quality checks, physiological estimates can become unreliable. This project addresses that gap by classifying PPG signal quality before clinical interpretation.

## Motivation

The project is motivated by the need for low-cost, real-time, edge-compatible signal quality assessment in wearable healthcare devices, telemedicine systems, and portable biomedical monitors.

## System Architecture

1. Acquire PPG records from the PhysioNet BIDMC dataset.
2. Extract and resample clean PPG signals to 100 Hz.
3. Preprocess signals using filtering and normalization.
4. Segment the signal into 5-second windows.
5. Generate synthetic degradation classes.
6. Extract statistical and spectral features.
7. Train and compare threshold rules, Random Forest, and SVM classifiers.
8. Export the Random Forest model to ONNX.
9. Benchmark ONNX inference latency and model size for edge feasibility.

## Workflow

Run the pipeline from the repository root:

```bash
python src/step1_download_data.py
python src/step2_extract_signal.py
python src/step3_visualise_raw.py
python src/step4_preprocess.py
python src/step5_windowing.py
python src/step6_degradation.py
python src/step7_visualise_classes.py
python src/step8_features.py
python src/step9_feature_analysis.py
python src/step10_train_models.py
python src/step11_confusion_matrices.py
python src/step12_edge_benchmark.py
```

## Dataset

The project uses PPG records from the PhysioNet BIDMC PPG and Respiration Dataset through `wfdb`. The scripts download selected BIDMC records, identify the PPG/PLETH channel, remove missing samples, resample the signal to 100 Hz, and save generated arrays under `data/`.

## Feature Extraction

The feature dataset contains 13 features:

`mean`, `variance`, `rms`, `kurtosis`, `zcr`, `peak_to_peak`, `skewness`, `mad`, `dominant_freq`, `spectral_entropy`, `ppg_band_power`, `power_ratio`, and `spectral_centroid`.

## Machine Learning Models

The training workflow compares:

- Threshold-based rules as a simple baseline.
- Random Forest for robust tabular multi-class classification.
- SVM with RBF kernel for nonlinear class boundaries.

Saved models are stored in `models/`.

## Results

From `models/results.txt`:

| Model | Accuracy | Macro F1 | Latency | Size |
| --- | ---: | ---: | ---: | ---: |
| Threshold Rules | 0.3340 | 0.1935 | <1 ms | 0 KB |
| Random Forest | 0.8642 | 0.8189 | 26346.435 ms | 1811.5 KB |
| SVM RBF | 0.8006 | 0.7606 | 0.429 ms | 509.3 KB |

Note: the Random Forest latency in the saved result appears unusually high for a single-window prediction and should be re-benchmarked on the target development machine before publication.

## Edge Deployment Feasibility

From `models/edge_benchmark.txt`, the ONNX Random Forest benchmark reports:

- Laptop mean latency: `0.108 ms`
- Raspberry Pi 4 estimate: `0.4 ms`
- ONNX model size: `1061.9 KB`
- Edge feasible: `YES`

Physical Raspberry Pi deployment remains a Phase 2 task.

## Folder Structure

```text
.
|-- README.md
|-- LICENSE
|-- requirements.txt
|-- .gitignore
|-- src/                  # Numbered research pipeline scripts
|-- data/                 # Processed arrays and feature dataset
|-- models/               # Trained models, ONNX export, result summaries
|-- figures/              # Generated plots and evaluation figures
|-- docs/                 # Technical documentation and audit report
|-- dashboard/            # Streamlit dashboard
|-- notebooks/            # Reserved for future notebooks
|-- assets/               # Reserved for presentation or media assets
|-- archive/              # Legacy or redundant project artifacts
```

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

For Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Environment Setup

Use Python 3.10 or later. Internet access is required for the PhysioNet download steps. Run commands from the repository root so relative paths resolve correctly.

## Dependencies

Dependencies are listed in `requirements.txt` and are based on actual imports in the source and dashboard files.

## Execution Steps

The numbered scripts are intended to be run sequentially. Later steps depend on artifacts generated by earlier steps, especially `data/clean_ppg_signal.npy`, `data/clean_ppg_preprocessed.npy`, `data/clean_windows.npy`, `data/X_all.npy`, `data/y_all.npy`, `data/feature_dataset.csv`, and saved models in `models/`.

## Expected Outputs

- `data/clean_ppg_signal.npy`
- `data/clean_ppg_preprocessed.npy`
- `data/clean_windows.npy`
- `data/X_all.npy`
- `data/y_all.npy`
- `data/feature_dataset.csv`
- `figures/raw_signal.png`
- `figures/preprocessing_comparison.png`
- `figures/five_classes.png`
- `figures/feature_distributions.png`
- `figures/correlation_matrix.png`
- `figures/pca_plot.png`
- `figures/confusion_matrices.png`
- `models/scaler.pkl`
- `models/random_forest.pkl`
- `models/svm.pkl`
- `models/rf_model.onnx`
- `models/results.txt`
- `models/edge_benchmark.txt`

## Screenshots

Key generated figures:

- Raw PPG signal: `figures/raw_signal.png`
- Preprocessing comparison: `figures/preprocessing_comparison.png`
- Five signal classes: `figures/five_classes.png`
- Feature distributions: `figures/feature_distributions.png`
- Correlation matrix: `figures/correlation_matrix.png`
- PCA class separability: `figures/pca_plot.png`
- Confusion matrices: `figures/confusion_matrices.png`

## Dashboard

Launch the Streamlit dashboard:

```bash
streamlit run dashboard/app.py
```

The dashboard lets users select a simulated signal condition and severity, visualize the PPG waveform, inspect extracted features, and view model prediction results.

`archive/dashboard_v2_legacy_broken.py` was archived because it currently contains a syntax error and is not the recommended launch target.

## Future Scope

- Validate models on additional datasets and real sensor fault samples.
- Deploy the ONNX model on Raspberry Pi or a microcontroller-class edge platform.
- Integrate live PPG acquisition hardware such as MAX30102.
- Add automated tests for feature extraction, degradation simulation, and model inference.
- Add model cards and experiment tracking for future Phase 2 work.

## Contributors

Project contributors: M Ramaa Gayatri, Tejaswini Badami, U Bhumika Asritha, and Vandana H S.

## References

- PhysioNet BIDMC PPG and Respiration Dataset.
- Breiman, L. "Random forests." Machine Learning, 2001.
- Project technical report, Phase 1.

