# Repository Audit

## Summary

The original archive contained a functional Phase 1 research pipeline, generated data/model artifacts, generated figures, two Streamlit dashboard files, a Python bytecode cache, and a full virtual environment. The cleaned repository preserves the pipeline and generated outputs while separating launchable code, documentation, and legacy artifacts.

## Folder Purposes

| Folder | Purpose |
| --- | --- |
| `src/` | Numbered scripts for dataset acquisition, preprocessing, degradation simulation, feature extraction, model training, evaluation, and edge benchmarking. |
| `data/` | Generated NumPy arrays and the extracted feature dataset. |
| `models/` | Trained model pickles, scaler, ONNX model, and textual result summaries. |
| `figures/` | Generated visual outputs used in reporting. |
| `dashboard/` | Working Streamlit demo application. |
| `docs/` | Technical project documentation and repository audit. |
| `archive/` | Legacy or non-launchable project files retained for traceability. |
| `assets/` | Reserved for future media, diagrams, or portfolio assets. |
| `notebooks/` | Reserved for future exploratory notebooks. |

## File Purposes

| File | Purpose |
| --- | --- |
| `src/step1_download_data.py` | Checks PhysioNet BIDMC record access and prints signal metadata. |
| `src/step2_extract_signal.py` | Downloads selected BIDMC records, extracts PPG/PLETH channels, resamples to 100 Hz, and saves `data/clean_ppg_signal.npy`. |
| `src/step3_visualise_raw.py` | Generates `figures/raw_signal.png`. |
| `src/step4_preprocess.py` | Filters/normalizes the clean signal and generates `figures/preprocessing_comparison.png`. |
| `src/step5_windowing.py` | Creates 5-second clean signal windows. |
| `src/step6_degradation.py` | Simulates clean, noisy, drifted, attenuated, and spiked signal classes. |
| `src/step7_visualise_classes.py` | Generates `figures/five_classes.png`. |
| `src/step8_features.py` | Extracts 13 statistical/spectral features and writes `data/feature_dataset.csv`. |
| `src/step9_feature_analysis.py` | Generates feature distribution, correlation, and PCA plots. |
| `src/step10_train_models.py` | Trains threshold, Random Forest, and SVM classifiers; saves models and `models/results.txt`. |
| `src/step11_confusion_matrices.py` | Generates `figures/confusion_matrices.png`. |
| `src/step12_edge_benchmark.py` | Converts the Random Forest model to ONNX and benchmarks inference. |
| `dashboard/app.py` | Streamlit interactive dashboard for signal simulation and model inference. |
| `archive/dashboard_v2_legacy_broken.py` | Archived dashboard version with a syntax error; retained for reference. |

## Execution Order

The project uses the existing numbered order from `step1` through `step12`. Run scripts from the repository root.

## Required Inputs

- Internet access for PhysioNet downloads in steps 1 and 2.
- Generated clean signal and window files for later processing steps.
- `data/feature_dataset.csv` for model training and evaluation.
- `models/random_forest.pkl` and `models/scaler.pkl` for dashboard inference and ONNX conversion.

## Generated Outputs

- Processed data arrays in `data/`.
- Trained models and benchmark summaries in `models/`.
- Visual outputs in `figures/`.

## Dead, Redundant, or Temporary Files

- `venv/` from the original archive is an environment artifact and was excluded from the cleaned deliverable. Recreate it with `python -m venv .venv` and `pip install -r requirements.txt`.
- `src/__pycache__/` and `*.pyc` files were excluded as Python cache artifacts.
- `dashboard_v2.py` was moved to `archive/dashboard_v2_legacy_broken.py` because it does not compile.

## Issues Found

- `dashboard_v2.py` has an unterminated string literal near its results-image section.
- Some source comments contain mojibake characters from encoding corruption. These comments do not affect execution, but future cleanup should normalize file encoding and comments.
- `step10_train_models.py` reports a very high Random Forest latency in `models/results.txt`; the ONNX benchmark is much lower. The model should be re-benchmarked under controlled conditions before publication.
- No automated test suite is present.
- The repository originally lacked root-level README, dependency, license, and ignore files.

