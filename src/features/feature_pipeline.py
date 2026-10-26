"""Feature Engineering and Preprocessing Pipeline for IWR6843 Radar Data.

Supports:
- Approach A: Statistical window features (mean, std, min, max, delta, spread, kinematics)
- Approach B: Full 176-feature temporal window (TI Pose/Fall channel-major representation)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis


FEATURE_NAMES_22 = [
    "posz", "velx", "vely", "velz", "accx", "accy", "accz",
    "y0", "z0", "snr0", "y1", "z1", "snr1", "y2", "z2", "snr2",
    "y3", "z3", "snr3", "y4", "z4", "snr4"
]
WINDOW_SIZE = 8

FEATURE_NAMES_176 = [
    f"{feat}_f{f}" for feat in FEATURE_NAMES_22 for f in range(WINDOW_SIZE)
]


def extract_statistical_features_from_window_176(vector_176: np.ndarray) -> dict[str, float]:
    """Extract interpretable statistical & physical features from one 176-float window.
    
    vector_176 is ordered channel-major: 8 frames for posz, then 8 for velx, etc.
    """
    matrix = vector_176.reshape(len(FEATURE_NAMES_22), WINDOW_SIZE)
    # Channel mapping
    ch = {feat: matrix[i, :] for i, feat in enumerate(FEATURE_NAMES_22)}

    # Cast to float64 for stable numeric calculations
    posz = np.clip(ch["posz"].astype(np.float64), -10.0, 10.0)
    velx = np.clip(ch["velx"].astype(np.float64), -20.0, 20.0)
    vely = np.clip(ch["vely"].astype(np.float64), -20.0, 20.0)
    velz = np.clip(ch["velz"].astype(np.float64), -20.0, 20.0)
    accx = np.clip(ch["accx"].astype(np.float64), -50.0, 50.0)
    accy = np.clip(ch["accy"].astype(np.float64), -50.0, 50.0)
    accz = np.clip(ch["accz"].astype(np.float64), -50.0, 50.0)

    # All point Z coordinates across 5 points and 8 frames
    point_z_all = np.clip(np.concatenate([ch[f"z{i}"] for i in range(5)]).astype(np.float64), -10.0, 10.0)
    point_y_all = np.clip(np.concatenate([ch[f"y{i}"] for i in range(5)]).astype(np.float64), -10.0, 10.0)
    point_snr_all = np.clip(np.concatenate([ch[f"snr{i}"] for i in range(5)]).astype(np.float64), 0.0, 1000.0)

    # Height features
    posz_mean = float(np.mean(posz))
    posz_std = float(np.std(posz))
    posz_min = float(np.min(posz))
    posz_max = float(np.max(posz))
    posz_delta = float(posz[-1] - posz[0])  # End minus start height
    posz_drop = float(posz_max - posz[-1])  # Drop from peak to current

    # Kinematics
    velz_min = float(np.min(velz))          # Peak downward velocity (negative)
    velz_mean = float(np.mean(velz))
    velz_std = float(np.std(velz))
    accz_min = float(np.min(accz))          # Peak downward acceleration
    accz_mean = float(np.mean(accz))
    accz_std = float(np.std(accz))

    vel_horiz = np.sqrt(velx**2 + vely**2)
    vel_horiz_mean = float(np.mean(vel_horiz))
    vel_horiz_max = float(np.max(vel_horiz))

    vel_total = np.sqrt(velx**2 + vely**2 + velz**2)
    vel_total_mean = float(np.mean(vel_total))
    vel_total_max = float(np.max(vel_total))

    # Point cloud geometry
    pt_z_nonzero = point_z_all[point_z_all != 0]
    z_spread = float(np.std(pt_z_nonzero)) if len(pt_z_nonzero) > 1 else 0.0
    z_mean_pts = float(np.mean(pt_z_nonzero)) if len(pt_z_nonzero) > 0 else 0.0
    
    pt_y_nonzero = point_y_all[point_y_all != 0]
    y_spread = float(np.std(pt_y_nonzero)) if len(pt_y_nonzero) > 1 else 0.0

    snr_nonzero = point_snr_all[point_snr_all > 0]
    snr_mean = float(np.mean(snr_nonzero)) if len(snr_nonzero) > 0 else 0.0
    snr_max = float(np.max(snr_nonzero)) if len(snr_nonzero) > 0 else 0.0

    # Higher order moments with zero-variance safeguards
    if posz_std > 1e-4:
        val_skew = float(skew(posz))
        val_kurt = float(kurtosis(posz))
        posz_skewness = float(np.nan_to_num(val_skew, nan=0.0, posinf=0.0, neginf=0.0))
        posz_kurt = float(np.nan_to_num(val_kurt, nan=0.0, posinf=0.0, neginf=0.0))
    else:
        posz_skewness = 0.0
        posz_kurt = 0.0

    return {
        "posz_mean": posz_mean,
        "posz_std": posz_std,
        "posz_min": posz_min,
        "posz_max": posz_max,
        "posz_delta": posz_delta,
        "posz_drop": posz_drop,
        "velz_min": velz_min,
        "velz_mean": velz_mean,
        "velz_std": velz_std,
        "accz_min": accz_min,
        "accz_mean": accz_mean,
        "accz_std": accz_std,
        "vel_horiz_mean": vel_horiz_mean,
        "vel_horiz_max": vel_horiz_max,
        "vel_total_mean": vel_total_mean,
        "vel_total_max": vel_total_max,
        "point_z_spread": z_spread,
        "point_z_mean": z_mean_pts,
        "point_y_spread": y_spread,
        "point_snr_mean": snr_mean,
        "point_snr_max": snr_max,
        "posz_skew": posz_skewness,
        "posz_kurt": posz_kurt,
    }


def build_feature_datasets(
    cleaned_windows_csv: str | Path = "data/cleaned/cleaned_windows_176.csv",
    output_dir: str | Path = "data/processed"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate both Approach A (statistical) and Approach B (176 features) datasets."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_cleaned = pd.read_csv(cleaned_windows_csv)
    print(f"Loaded {len(df_cleaned)} cleaned windows.")

    f176_cols = [c for c in df_cleaned.columns if c in FEATURE_NAMES_176]
    if len(f176_cols) != 176:
        # Match by prefix
        f176_cols = [c for c in df_cleaned.columns if any(c.startswith(f"{name}_f") for name in FEATURE_NAMES_22)]
    
    # Ensure ordered correctly
    f176_cols = FEATURE_NAMES_176

    X_176 = df_cleaned[f176_cols].to_numpy(dtype=np.float32)

    # Approach B: 176-feature dataset
    meta_cols = ["subject_id", "activity_label", "trial_id", "is_fall", "frame", "time", "tid"]
    df_meta = df_cleaned[[c for c in meta_cols if c in df_cleaned.columns]].copy()

    df_approach_b = pd.concat([df_meta, pd.DataFrame(X_176, columns=f176_cols)], axis=1)
    approach_b_path = output_dir / "features_approach_b_176.csv"
    df_approach_b.to_csv(approach_b_path, index=False)
    print(f"Saved Approach B (176 features) to {approach_b_path} ({df_approach_b.shape})")

    # Approach A: Statistical features
    stat_features_list = []
    for row in X_176:
        stat_features_list.append(extract_statistical_features_from_window_176(row))

    df_stats = pd.DataFrame(stat_features_list)
    df_approach_a = pd.concat([df_meta, df_stats], axis=1)
    approach_a_path = output_dir / "features_approach_a_statistical.csv"
    df_approach_a.to_csv(approach_a_path, index=False)
    print(f"Saved Approach A (statistical features) to {approach_a_path} ({df_approach_a.shape})")

    # Save feature metadata schema
    schema = {
        "approach_a_features": list(df_stats.columns),
        "approach_b_features": f176_cols,
        "window_size": WINDOW_SIZE,
        "input_dim_a": len(df_stats.columns),
        "input_dim_b": len(f176_cols),
        "total_samples": len(df_cleaned)
    }
    schema_path = output_dir / "feature_schema.json"
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)
    print(f"Saved feature schema to {schema_path}")

    return df_approach_a, df_approach_b


def main():
    build_feature_datasets()


if __name__ == "__main__":
    main()
