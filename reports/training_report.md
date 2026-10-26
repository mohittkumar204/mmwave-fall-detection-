# IWR6843 Fall Detection Training & Experiment Report

## 1. Project Overview
This report documents the end-to-end model development and comparative benchmark for mmWave radar fall detection using Texas Instruments IWR6843ISK-ODS sensor data collected from six human subjects.

## 2. Experimental Setup
- **Hardware:** Texas Instruments IWR6843ISK-ODS (60-64 GHz mmWave radar)
- **Firmware:** TI 3D People Tracking Demo (SDK 3.5), 55 ms frame periodicity (~18.18 Hz)
- **Input Data:** 21,412 temporal sliding windows from 30 valid trial sessions across 6 subjects
- **Validation Scheme:** Leave-One-Subject-Out (LOSO) Cross-Validation (6 folds)
- **Feature Approaches Compared:**
  - **Approach A (Statistical Features):** 23 engineered features per window (mean height, height drop, min vertical velocity, peak acceleration, point spread, SNR)
  - **Approach B (Temporal Representation):** 176 channel-major features (8-frame temporal history of 22 instantaneous target & point attributes)

## 3. Best Validated Model
- **Selected Architecture:** `RandomForest` under `Approach_A_Statistical`
- **Mean Accuracy:** `88.31% ± 5.12%`
- **Mean F1-Score (FALL):** `62.56% ± 12.47%`
- **Mean Sensitivity / Recall:** `58.08% ± 16.26%`
- **Mean Specificity:** `94.92% ± 2.90%`
- **Single-Window Inference Latency:** `53.984 ms`
- **Model Storage Footprint:** `7765.8 KB`

## 4. Artifact Manifest
- `models/best_fall_model.joblib`: Serialized production model
- `artifacts/scaler.joblib`: Pretrained feature standardizer
- `artifacts/best_model_metadata.json`: Feature names, schema, and threshold metadata
- `reports/model_comparison.csv`: Tabular benchmark across all evaluated algorithms
- `reports/cross_validation_report.md`: Complete LOSO fold breakdowns
