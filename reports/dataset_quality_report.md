# IWR6843 Radar Dataset Quality & Inventory Report

## 1. Executive Summary
- **Radar Sensor:** Texas Instruments IWR6843ISK-ODS (60-64 GHz mmWave)
- **Total Subjects:** 6 (person1, person2, person3, person4, person5, person6)
- **Total Activity Classes:** 5 (falling, lying, sitting, standing, walking)
- **Total Files Tracked:** 192
- **Total Recorded Sessions (Trials):** 32
  - **Valid & Usable Sessions:** 30
  - **Empty / Incomplete Sessions:** 2
- **Nominal Radar Frame Rate:** ~18.18 Hz (55 ms chirp frame periodicity)

## 2. Activity Labels & Class Mappings
### Multiclass Mapping:
| Activity Label | Class ID | Pose Class Name |
| :--- | :--- | :--- |
| `standing` | 0 | STANDING |
| `sitting` | 1 | SITTING |
| `lying` | 2 | LYING |
| `falling` | 3 | FALLING |
| `walking` | 4 | WALKING |

### Binary Fall Detection Mapping:
| Target Label | Class Name | Included Activities |
| :--- | :--- | :--- |
| `0` | NO_FALL | standing, sitting, lying, walking |
| `1` | FALL | falling |

## 3. Recording Quality & Session Breakdown

### Per-Subject Trial Matrix
| subject_id   |   falling |   lying |   sitting |   standing |   walking |
|:-------------|----------:|--------:|----------:|-----------:|----------:|
| person1      |         1 |       1 |         1 |          2 |         1 |
| person2      |         1 |       1 |         1 |          0 |         1 |
| person3      |         1 |       1 |         1 |          1 |         1 |
| person4      |         1 |       1 |         1 |          1 |         1 |
| person5      |         1 |       1 |         1 |          1 |         1 |
| person6      |         1 |       1 |         1 |          1 |         1 |

### Data Integrity & Anomaly Audit
1. **Empty / Incomplete Sessions Identified:**
   - `person1/walking/T01_20261008_132142`: 0 detected target frames (0 bytes payload). *Resolution:* Excluded from training. Note that `person1/walking` has a second trial `T01_20261008_132154` with 922 valid frames.
   - `person2/standing/T01_20261008_133624`: 0 detected target frames (0 bytes payload). *Resolution:* Excluded from training. Subject `person2` lacks `standing` data; explicitly documented for Leave-One-Subject-Out (LOSO) CV fold.

2. **Temporal & Point Quality:**
   - **Missing Values (NaN / Inf):** 0 across all 30 valid trial recordings in all feature columns.
   - **Coordinate Conventions:** Sensor-centric Cartesian coordinate frame ($X$ lateral, $Y$ depth, $Z$ vertical height).
   - **Feature Sets Recovered:**
     - `raw_points.csv`: 3D point cloud with radial Doppler velocity, SNR, and cluster index.
     - `targets.csv`: Tracked human centroid state $(x, y, z, v_x, v_y, v_z, a_x, a_y, a_z)$ and height bounding box $(\text{maxZ}, \text{minZ})$.
     - `features_22.csv`: TI 22-dimensional instantaneous kinematic and point feature representation.
     - `features_176.csv`: 8-frame temporal sliding window ($22 \times 8 = 176$ channel-major features).

3. **Motion Signal Preservation:**
   - Sudden accelerations and height drops during falls are genuine physical phenomena and are preserved without synthetic smoothing or premature outlier clipping.
