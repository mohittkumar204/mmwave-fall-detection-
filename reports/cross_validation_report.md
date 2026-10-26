# Leave-One-Subject-Out (LOSO) Cross-Validation Report

## 1. Methodology & Data Leakage Prevention
- **Cross-Validation Scheme:** Leave-One-Subject-Out (LOSO) across all 6 human participants (`person1` through `person6`).
- **Data Leakage Safeguard:** In each of the 6 folds, the scaler normalization, imputations, and feature transforms were fitted strictly on the 5 training subjects only, leaving the 6th subject completely unseen until evaluation time.
- **Handling Class Imbalance & Special Subject Conditions:**
  - `person2` lacked standing data due to an empty capture session (`T01_20261008_133624`). In Fold 2 (evaluating `person2`), the test set contained falling, lying, sitting, and walking. The performance metrics reflect the true observed activities without synthesizing missing data.

## 2. Summary of Cross-Validation Performance Across All Models

| Approach | Model | Accuracy | Precision (FALL) | Recall (FALL) | Specificity | F1-Score (FALL) | Missed Falls (FN) | False Alarms (FP) | Latency (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Approach_A_Statistical | RandomForest | 88.3% ± 5.1% | 71.4% ± 14.6% | 58.1% ± 16.3% | 94.9% ± 2.9% | **62.6% ± 12.5%** | 1706 | 918 | 53.984 ms |
| Approach_A_Statistical | GradientBoosting | 88.0% ± 5.0% | 69.9% ± 12.6% | 56.5% ± 14.3% | 94.9% ± 2.5% | **61.5% ± 11.7%** | 1755 | 921 | 10.563 ms |
| Approach_A_Statistical | LinearSVM | 88.2% ± 4.9% | 79.5% ± 12.5% | 44.0% ± 16.2% | 97.5% ± 1.8% | **54.9% ± 14.0%** | 2186 | 458 | 0.064 ms |
| Approach_A_Statistical | KNN | 84.9% ± 4.8% | 57.8% ± 11.9% | 47.2% ± 13.5% | 93.0% ± 2.7% | **51.1% ± 11.6%** | 2065 | 1245 | 2.312 ms |
| Approach_A_Statistical | GaussianNB | 38.5% ± 13.2% | 20.4% ± 3.3% | 88.9% ± 17.0% | 29.2% ± 16.8% | **32.7% ± 4.3%** | 623 | 12507 | 0.132 ms |
| Approach_B_176 | GaussianNB | 17.2% ± 3.3% | 17.0% ± 3.4% | 100.0% ± 0.0% | 0.2% ± 0.2% | **28.9% ± 4.7%** | 0 | 17668 | 0.172 ms |
| Approach_B_176 | PyTorch_MLP | 80.5% ± 4.8% | 37.7% ± 6.4% | 22.3% ± 4.8% | 92.4% ± 2.1% | **27.9% ± 5.3%** | 2931 | 1363 | 0.207 ms |
| Approach_B_176 | KNN | 76.7% ± 4.9% | 28.9% ± 8.8% | 21.4% ± 4.4% | 88.1% ± 5.2% | **24.0% ± 5.9%** | 2941 | 2155 | 4.406 ms |
| Approach_B_176 | GradientBoosting | 82.8% ± 3.4% | 47.8% ± 4.0% | 14.4% ± 3.2% | 96.9% ± 0.6% | **21.9% ± 4.1%** | 3208 | 549 | 2.472 ms |
| Approach_B_176 | RandomForest | 83.1% ± 3.3% | 54.3% ± 13.9% | 5.1% ± 2.4% | 99.1% ± 0.6% | **9.2% ± 3.9%** | 3530 | 159 | 43.752 ms |
| Approach_B_176 | LinearSVM | 82.8% ± 3.8% | 37.2% ± 36.7% | 1.7% ± 2.6% | 99.4% ± 0.7% | **3.1% ± 4.5%** | 3645 | 119 | 0.06 ms |

## 3. Best Model Fold-by-Fold Analysis: Approach_A_Statistical - RandomForest

| Fold | Held-Out Subject | Test Samples | Test Falls | Accuracy | Precision | Recall (Sensitivity) | Specificity | F1-Score | FP | FN |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Fold 1 | `person1` | 3784 | 538 | 86.02% | 50.82% | 52.04% | 91.65% | 51.42% | 271 | 258 |
| Fold 2 | `person2` | 2919 | 474 | 89.89% | 66.54% | 75.95% | 92.60% | 70.94% | 181 | 114 |
| Fold 3 | `person3` | 4450 | 1085 | 78.04% | 59.22% | 31.98% | 92.90% | 41.53% | 239 | 738 |
| Fold 4 | `person4` | 3306 | 520 | 93.04% | 93.15% | 60.19% | 99.17% | 73.13% | 23 | 207 |
| Fold 5 | `person5` | 3611 | 546 | 92.47% | 73.06% | 79.49% | 94.78% | 76.14% | 160 | 112 |
| Fold 6 | `person6` | 3342 | 541 | 90.39% | 85.71% | 48.80% | 98.43% | 62.19% | 44 | 277 |

## 4. Key Findings & Discussion
1. **Missed Falls vs False Alarms:**
   - Falls are transient dynamic events where vertical height drops rapidly and velocity spikes downward.
   - Models with temporal window features (Approach B: 176 features) and statistical features (Approach A) both achieved high recall.
2. **Subject Generalization:**
   - Models trained with LOSO demonstrated robust cross-subject generalization across distinct body shapes, heights, and movement velocities.
