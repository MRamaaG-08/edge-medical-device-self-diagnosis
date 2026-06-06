# Edge-Based Medical Device Self-Diagnosis System

## Abstract

Photoplethysmography (PPG) is a non-invasive optical sensing method used in wearable and portable healthcare systems. Although PPG sensors are compact and inexpensive, their signals are vulnerable to degradation from noise, baseline drift, weak contact, attenuation, ambient interference, and impulse spikes. This project develops a Phase 1 machine learning pipeline that classifies PPG signal quality into five conditions: clean, noisy, drifted, attenuated, and spiked. The workflow combines PhysioNet data acquisition, preprocessing, synthetic degradation generation, feature extraction, model training, evaluation, and edge deployment benchmarking.

## Introduction

PPG-based systems are widely used for physiological monitoring, including heart-rate and oxygen-saturation estimation. In practical wearable environments, degraded sensor signals can produce unreliable measurements if quality is not assessed before downstream analysis. A lightweight self-diagnosis layer can improve reliability by detecting sensor degradation conditions early.

## Literature Review

Prior work in biomedical signal processing shows that PPG quality is affected by motion artifacts, sensor placement, skin contact, ambient light, and hardware limitations. Machine learning methods such as Random Forest and Support Vector Machine are commonly used for tabular biomedical classification because they can learn class boundaries from engineered features while remaining more interpretable and deployment-friendly than large deep learning models.

## Existing System

Many PPG-based health monitoring systems focus on estimating physiological parameters directly. They often lack an explicit signal quality classification stage, which means corrupted input may be processed as if it were reliable. Cloud-based analysis can also introduce latency, privacy, and connectivity constraints.

## Problem Statement

The project aims to classify PPG sensor signal conditions before physiological interpretation. The core problem is to identify whether an incoming PPG window is clean or degraded by noise, drift, attenuation, or spikes, while keeping the solution lightweight enough for future edge deployment.

## Proposed System

The proposed system builds a reproducible pipeline:

1. Acquire PPG signals from the BIDMC dataset.
2. Extract and resample PPG channels.
3. Preprocess the signal using filtering and normalization.
4. Split signals into fixed 5-second windows.
5. Simulate degradation classes.
6. Extract 13 time-domain and frequency-domain features.
7. Train threshold, Random Forest, and SVM classifiers.
8. Evaluate classification performance using accuracy, macro F1, and confusion matrices.
9. Convert the best model to ONNX and estimate edge inference feasibility.

## Hardware and Software Requirements

Suggested development environment:

- Python 3.10 or later
- NumPy, Pandas, SciPy, Matplotlib, Seaborn
- scikit-learn
- WFDB for PhysioNet access
- skl2onnx and onnxruntime for edge benchmarking
- Streamlit and Plotly for dashboard support
- Development laptop or desktop
- Raspberry Pi 4 or equivalent edge target for Phase 2 validation

## Project Analysis

The project uses simulation-augmented development because real labeled sensor-fault datasets are difficult to obtain. Clean PPG windows are transformed into controlled degradation classes, enabling supervised multi-class classification. This approach provides a repeatable foundation for Phase 1 analysis while leaving real-hardware validation for future work.

## Methodology

The signal processing workflow begins with BIDMC PPG records. PPG/PLETH channels are extracted, cleaned of missing values, resampled to 100 Hz, filtered, and normalized. The processed signal is segmented into 5-second windows. Four degradation conditions are injected in addition to the clean class:

- Gaussian noise
- Baseline drift
- Signal attenuation
- Impulse spikes

Each window is converted into a 13-feature vector using statistical and spectral measurements. Models are then trained and evaluated on stratified splits.

## Working

The numbered scripts in `src/` form the complete execution pipeline. Intermediate outputs are saved so each stage can be inspected independently. Visual outputs in `figures/` document raw signals, preprocessing, class simulation, feature behavior, PCA separation, and confusion matrices.

## Results and Discussion

The saved results show:

| Model | Accuracy | Macro F1 |
| --- | ---: | ---: |
| Threshold Rules | 0.3340 | 0.1935 |
| Random Forest | 0.8642 | 0.8189 |
| SVM RBF | 0.8006 | 0.7606 |

Random Forest achieved the strongest overall classification performance in the recorded experiment. SVM also performed well, while threshold rules provided a low-complexity but less accurate baseline. The generated confusion matrices and PCA plot help evaluate class separability and misclassification patterns.

## Edge Deployment Feasibility

The Random Forest model was converted to ONNX and benchmarked with ONNX Runtime. The recorded benchmark reports a mean laptop inference latency of `0.108 ms`, an estimated Raspberry Pi 4 latency of `0.4 ms`, and an ONNX model size of `1061.9 KB`. These results indicate that the feature-based model is feasible for future edge deployment. Physical edge-device testing remains Phase 2 work.

## Advantages

- Provides signal quality assessment before physiological interpretation.
- Uses reproducible simulation for controlled degradation classes.
- Combines interpretable features with lightweight machine learning.
- Produces visual outputs for technical review.
- Supports future edge deployment through ONNX conversion.

## Conclusion

Phase 1 establishes a complete research pipeline for PPG sensor degradation detection. The project demonstrates that engineered features and classical machine learning models can distinguish clean and degraded PPG windows with promising performance. The ONNX benchmark suggests that the approach is practical for low-latency edge inference.

## Future Scope

- Validate with real degraded sensor data.
- Test deployment on Raspberry Pi 4 or embedded hardware.
- Integrate live acquisition from a PPG sensor module.
- Improve model validation using additional PhysioNet records and external datasets.
- Add automated tests and experiment tracking.
- Extend the dashboard for live streaming and fault history.

## References

- PhysioNet BIDMC PPG and Respiration Dataset.
- L. Breiman, "Random forests," Machine Learning, 2001.
- Project Phase 1 technical report.

