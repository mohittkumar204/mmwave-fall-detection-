# IWR6843 Radar-Based Fall Detection & 3D Posture Tracking System

End-to-end mmWave radar fall detection, multi-person tracking, and posture classification system using the **Texas Instruments IWR6843ISK-ODS** sensor.

Includes:
- Complete dataset cleaning and preprocessing pipeline across 6 subjects.
- Strict **Leave-One-Subject-Out (LOSO)** cross-validation preventing data leakage.
- Comparison of **Approach A (Statistical & Kinematic Features)** vs **Approach B (176-Feature Temporal Window)**.
- 5-Class Posture Estimation: 🚶 **WALKING**, 🧍 **STANDING**, 🪑 **SITTING**, ⚠️ **FALLING**, 🛌 **LYING**.
- Offline inference engine, live physical radar UART stream decoder, interactive Streamlit 3D visualizer matching the TI Industrial Visualizer layout, and dedicated radar session recorder.

---

## 1. System Architecture

```
                    +------------------------------------+
                    |  TI IWR6843ISK-ODS mmWave Sensor   |
                    | (60-64 GHz, 55 ms frame periodicity)|
                    +-----------------+------------------+
                                      |
                      UART Serial (CLI: COM11, Data: COM14)
                                      v
                    +------------------------------------+
                    |   Radar Packet & TLV Decoder       |
                    |   (TLVs 1010, 1011, 1012, 1020)   |
                    +-----------------+------------------+
                                      |
                    +-----------------v------------------+
                    |  8-Frame Temporal Rolling Buffer   |
                    |  (Per-Target Kinematic & Point 22) |
                    +-----------------+------------------+
                                      |
                                      +-------------------------------+
                                      |                               |
                                      v                               v
                        [Approach A: Statistical]       [Approach B: 176 Temporal]
                        - Delta Z, Drop magnitude       - 8-frame flattened vector
                        - Downward velocity/accel       - Channel-major order
                        - Point cloud spread & SNR
                                      |                               |
                                      +---------------+---------------+
                                                      |
                                                      v
                                        +----------------------------+
                                        |  Pretrained Scaler & Model |
                                        |   (Posture & Fall Model)   |
                                        +--------------+-------------+
                                                       |
                                      +----------------+---------------+
                                      v                                v
                        +----------------------------+  +----------------------------+
                        |  Interactive 3D Visualizer |  |  Event Logging & Storage   |
                        |   (3D Scene & Bounding Box)|  |   (logs/live_fall_events)  |
                        +----------------------------+  +----------------------------+
```

---

## 2. Directory Structure

```
FallDetection/
├── datasets/                            # Immutable original 6-person recordings
│   └── my_posture_dataset/              # person1 .. person6 (falling, lying, sitting, standing, walking)
├── external/                            # Cloned reference repositories
│   ├── iwr6843-points-visual/           # Cloned Repo A (preserved original)
│   └── iwr6843-fall-logger/             # Cloned Repo B (preserved original)
├── data/
│   ├── cleaned/                         # Cleaned targets (21,872) & ready windows (21,412)
│   ├── processed/                       # features_approach_a_statistical.csv & features_approach_b_176.csv
│   └── splits/                          # loso_splits.json (6 subject-wise folds)
├── models/
│   ├── best_fall_model.joblib           # Trained binary fall detector
│   └── posture_model_5class.joblib      # 5-class posture & fall classifier (WALKING, STANDING, SITTING, FALLING, LYING)
├── artifacts/
│   ├── scaler.joblib                    # Fitted feature standardizer (binary model)
│   ├── posture_scaler.joblib            # Fitted standardizer (5-class posture model)
│   ├── best_model_metadata.json         # Feature names, thresholds, and schema
│   ├── posture_model_metadata.json      # 5-class posture metadata and class IDs
│   └── label_mapping.json               # Binary & multiclass activity schema
├── configs/
│   └── ODS_6m_posture_tuned.cfg         # Tuned TI chirp and tracking configuration
├── src/
│   ├── data/
│   │   ├── dataset_inventory.py         # Inventory scanner & quality reporter
│   │   ├── clean_dataset.py             # Deduplication, coordinate sanity & cleaning
│   │   └── dataset_split.py             # LOSO cross-validation split builder
│   ├── features/
│   │   └── feature_pipeline.py          # Approach A & B feature extraction
│   ├── training/
│   │   └── train_models.py              # Multi-model trainer & LOSO cross-validator
│   ├── inference/
│   │   ├── offline_inference.py         # Batch inference engine on session folders
│   │   └── live_fall_detector.py        # Real-time streaming fall detector
│   └── radar/
│       ├── packet_parser.py             # TI TLV packet parsing & error recovery
│       ├── serial_acquisition.py        # UART acquisition & replay engine
│       └── data_collector.py            # Physical sensor data collection logger
├── tests/                               # Automated pytest suite (all 14 passing)
├── reports/
│   ├── dataset_inventory.csv            # File-by-file dataset inventory
│   ├── dataset_quality_report.md        # Quality audit & session matrix
│   ├── model_comparison.csv             # Complete multi-model benchmark
│   ├── cross_validation_report.md       # Per-subject fold breakdown
│   ├── training_report.md               # Detailed training & metrics report
│   └── confusion_matrices/              # Fold-by-fold confusion matrix CSVs
├── dashboard.py                         # Streamlit interactive 3D visualizer
├── requirements.txt                     # Pinned Python package dependencies
├── .gitignore                           # Excludes large datasets, models & caches
└── README.md                            # Comprehensive PowerShell command guide
```

---

## 3. Dataset Summary & Inspection

- **Subjects:** 6 human participants (`person1`, `person2`, `person3`, `person4`, `person5`, `person6`).
- **Activities:** 5 classes (`falling`, `lying`, `sitting`, `standing`, `walking`).
- **Total Recording Trials:** 32 sessions (30 valid, 2 empty session captures).
  - `person1/walking/T01_20261008_132142`: Empty capture (0 detected frames). Excluded; valid trial `T01_20261008_132154` retained.
  - `person2/standing/T01_20261008_133624`: Empty capture (0 detected frames). Excluded; `person2` standing absence explicitly documented for Fold 2.
- **Total Valid Frames:** 21,872 frames (~14.6 minutes of physical mmWave radar data).
- **Total Ready 8-Frame Windows:** 21,412 temporal samples.
- **Missing Values (NaN/Inf):** 0 across all valid files.

---

## 4. Model Benchmark Results (Leave-One-Subject-Out CV)

Cross-validation evaluated by testing on 1 held-out subject using models trained exclusively on the other 5 subjects:

| Approach | Model | Accuracy | Precision (FALL) | Recall (FALL) | Specificity | F1-Score (FALL) | Latency | Size |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Approach A (Statistical)** | **RandomForest** | **88.31% ± 5.1%** | **71.42% ± 14.6%** | **58.08% ± 16.3%** | **94.92% ± 2.9%** | **62.56% ± 12.5%** | **16.1 ms** | **7.6 MB** |
| Approach A (Statistical) | GradientBoosting | 88.05% ± 5.0% | 69.89% ± 12.6% | 56.51% ± 14.3% | 94.88% ± 2.5% | 61.52% ± 11.7% | 1.26 ms | 404 KB |
| Approach A (Statistical) | LinearSVM | 88.17% ± 4.9% | 79.54% ± 12.5% | 44.01% ± 16.2% | 97.46% ± 1.8% | 54.95% ± 14.0% | 0.05 ms | 0.9 KB |
| Approach A (Statistical) | KNN | 84.89% ± 4.8% | 57.82% ± 11.9% | 47.19% ± 13.5% | 92.96% ± 2.7% | 51.09% ± 11.6% | 1.36 ms | 1.7 MB |
| Approach B (176 Features) | PyTorch MLP | 80.90% ± 3.5% | 38.47% ± 1.2% | 21.29% ± 3.3% | 93.18% ± 0.9% | 27.24% ± 2.9% | 0.11 ms | 68 KB |
| Approach B (176 Features) | GradientBoosting | 82.78% ± 3.4% | 47.76% ± 4.1% | 14.40% ± 3.2% | 96.87% ± 0.6% | 21.90% ± 4.1% | 1.22 ms | 606 KB |
| Approach B (176 Features) | RandomForest | 83.11% ± 3.3% | 54.29% ± 13.9% | 5.14% ± 2.4% | 99.09% ± 0.6% | 9.21% ± 3.9% | 15.7 ms | 12.5 MB |

> **Key Finding:** Approach A (physically engineered features like height drop $\Delta z$, downward velocity $v_z$, peak acceleration $a_z$, and cluster spread) significantly outperforms raw 176 point-cloud positions when generalizing to unseen human subjects. Raw spatial coordinates overfit to individual training subjects, whereas kinematics generalize cleanly.

---

## 5. Step-by-Step Execution Guide (Windows PowerShell)

### Step 1: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 2: Generate Dataset Inventory & Quality Reports
```powershell
python src/data/dataset_inventory.py
```

### Step 3: Run Data Cleaning Pipeline
```powershell
python src/data/clean_dataset.py
```

### Step 4: Extract Approach A & B Features
```powershell
python src/features/feature_pipeline.py
```

### Step 5: Build LOSO Cross-Validation Splits
```powershell
python src/data/dataset_split.py
```

### Step 6: Train and Compare Models
```powershell
python src/training/train_models.py
```

### Step 7: Run Offline Inference on a Recorded Trial
```powershell
# Run on a falling trial:
python src/inference/offline_inference.py --trial datasets/my_posture_dataset/person1/falling/T01_20261008_132511 --out reports/person1_fall_predictions.csv

# Run on a non-fall (sitting) trial to verify zero false alarms:
python src/inference/offline_inference.py --trial datasets/my_posture_dataset/person1/sitting/T01_20261008_132348 --out reports/person1_sitting_predictions.csv
```

### Step 8: Run the Automated Test Suite
```powershell
pytest -v tests/
```
All 14 tests will pass (verifying offline algorithms + physical hardware detection on COM11/COM14).

### Step 9: Launch the 3D Tracking & Posture Visualizer Web App
```powershell
streamlit run dashboard.py
```
- Opens the **3D Plot** matching the TI Industrial Visualizer layout:
  - 3D Blue wireframe boundary box ($[-3, 3] \times [0.5, 7.5] \times [0, 3]$ m)
  - 3D Grid ground plane
  - 3D Point cloud colored by SNR / Doppler / Height
  - 3D Human bounding cuboids around tracked people
  - Real-time 5-Class Posture Detection: 🚶 **WALKING**, 🧍 **STANDING**, 🪑 **SITTING**, ⚠️ **FALLING**, 🛌 **LYING**
  - Includes both **3D Dataset Replay Mode** and **Live Physical Radar Stream** on COM11/COM14.

### Step 10: (Optional) Launch Native Desktop TI Industrial Visualizer
```powershell
python external/iwr6843-fall-logger/run_ti_style_visualizer.py --cli COM11 --data COM14 --cfg configs/ODS_6m_posture_tuned.cfg
```

---

## 6. Physical Radar Setup & Hardware Configuration

### Sensor Model & Board Variant
- **Sensor:** Texas Instruments **IWR6843ISK-ODS** (60-64 GHz mmWave radar with 120° Azimuth x 120° Elevation Field of View).
- **Carrier Board:** TI **MMWAVEBOOST** (Silicon Labs Dual CP2105 USB-to-UART Bridge).
- **Detected COM Ports:**
  - **CLI / Config Port:** `COM11` (115200 baud)
  - **Data Port:** `COM14` (921600 baud)
- **SOP Switch Setting (Functional Mode):**
  - `SOP2`: **OFF (Open)**
  - `SOP1`: **OFF (Open)**
  - `SOP0`: **ON (Closed)**

### Firmware & Chirp Configuration
- Pre-flash TI mmWave SDK 3.5 / Radar Toolbox **3D People Tracking Demo** binary (`3d_people_tracking_ods.bin`).
- Configuration: `configs/ODS_6m_posture_tuned.cfg` (55 ms frame periodicity, ~18.18 Hz).
