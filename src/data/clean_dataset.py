"""Reproducible Data Cleaning and Validation Pipeline for IWR6843 mmWave Radar Dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import pandas as pd


def get_dataset_root() -> Path:
    p1 = Path("datasets/my_posture_dataset")
    if p1.exists():
        return p1
    p2 = Path("dataset/my_posture_dataset")
    if p2.exists():
        return p2
    return Path("dataset")


def clean_dataset(
    dataset_root: Path | None = None,
    cleaned_dir: Path | str = "data/cleaned"
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    if dataset_root is None:
        dataset_root = get_dataset_root()
    cleaned_dir = Path(cleaned_dir)
    cleaned_dir.mkdir(parents=True, exist_ok=True)

    cleaning_logs: list[dict[str, Any]] = []
    trial_dirs = sorted([d for d in dataset_root.glob("*/*/*") if d.is_dir()])

    all_cleaned_targets: list[pd.DataFrame] = []
    all_cleaned_windows: list[pd.DataFrame] = []

    for trial_dir in trial_dirs:
        subject_id = trial_dir.parent.parent.name
        activity_label = trial_dir.parent.name
        trial_id = trial_dir.name
        trial_key = f"{subject_id}/{activity_label}/{trial_id}"

        targets_file = trial_dir / "targets.csv"
        f176_file = trial_dir / "features_176.csv"
        f22_file = trial_dir / "features_22.csv"
        meta_file = trial_dir / "metadata.json"

        # 1. Check for empty recording
        if not targets_file.exists() or targets_file.stat().st_size < 100:
            cleaning_logs.append({
                "trial": trial_key,
                "stage": "trial_validation",
                "action": "EXCLUDE_TRIAL",
                "reason": "Empty recording session (no detected targets/points)",
                "details": f"File size {targets_file.stat().st_size if targets_file.exists() else 0} bytes"
            })
            continue

        df_tar = pd.read_csv(targets_file)
        if len(df_tar) == 0:
            cleaning_logs.append({
                "trial": trial_key,
                "stage": "trial_validation",
                "action": "EXCLUDE_TRIAL",
                "reason": "Empty dataframe in targets.csv",
                "details": "0 rows in targets"
            })
            continue

        df_176 = pd.read_csv(f176_file) if f176_file.exists() else pd.DataFrame()
        
        # 2. Duplicate check within trial
        # In radar data, multiple targets (TIDs) can exist per frame.
        # Check for duplicate (frame, tid)
        dup_mask = df_tar.duplicated(subset=["frame", "tid"], keep="first")
        n_dups = int(dup_mask.sum())
        if n_dups > 0:
            df_tar = df_tar[~dup_mask].copy()
            cleaning_logs.append({
                "trial": trial_key,
                "stage": "deduplication",
                "action": "REMOVE_DUPLICATES",
                "reason": "Exact (frame, tid) duplicates",
                "details": f"Removed {n_dups} duplicate records"
            })

        # 3. Temporal monotonic check
        # Frames should be sorted by time and frame number
        df_tar = df_tar.sort_values(by=["time", "frame", "tid"]).reset_index(drop=True)

        # 4. Check coordinate sanity without destroying fall dynamics
        # Sensor height is mounted around 2.0m tilted 15 deg.
        # Height Z can vary between -1.5m to 3.0m in sensor coordinates.
        invalid_z = (df_tar["z"] < -3.0) | (df_tar["z"] > 4.0)
        n_invalid_z = int(invalid_z.sum())
        if n_invalid_z > 0:
            # We log but do not delete unless completely nonsensical (>100m)
            outlier_z = (df_tar["z"] < -10.0) | (df_tar["z"] > 10.0)
            n_outliers = int(outlier_z.sum())
            if n_outliers > 0:
                df_tar = df_tar[~outlier_z].copy()
                cleaning_logs.append({
                    "trial": trial_key,
                    "stage": "coordinate_validation",
                    "action": "FILTER_EXTREME_OUTLIER",
                    "reason": "Extreme sensor coordinate anomaly |z| > 10m",
                    "details": f"Removed {n_outliers} rows"
                })

        # 5. Clean temporal 176 features if present
        if len(df_176) > 0:
            # Remove exact (frame, tid) duplicates if any
            if "frame" in df_176.columns and "tid" in df_176.columns:
                dup_176 = df_176.duplicated(subset=["frame", "tid"], keep="first")
                if dup_176.sum() > 0:
                    df_176 = df_176[~dup_176].copy()

            # Filter ready windows only (ready == 1 means 8 full history frames available)
            if "ready" in df_176.columns:
                n_unready = int((df_176["ready"] != 1).sum())
                df_176_ready = df_176[df_176["ready"] == 1].copy()
                cleaning_logs.append({
                    "trial": trial_key,
                    "stage": "window_preparation",
                    "action": "RETAIN_READY_WINDOWS",
                    "reason": "Transient initial padding slots (first 7 frames of track history)",
                    "details": f"Kept {len(df_176_ready)} ready windows; excluded {n_unready} initial unready slots"
                })
            else:
                df_176_ready = df_176.copy()

            # Attach metadata columns
            df_176_ready["subject_id"] = subject_id
            df_176_ready["activity_label"] = activity_label
            df_176_ready["trial_id"] = trial_id
            df_176_ready["is_fall"] = 1 if activity_label == "falling" else 0
            all_cleaned_windows.append(df_176_ready)

        df_tar["subject_id"] = subject_id
        df_tar["activity_label"] = activity_label
        df_tar["trial_id"] = trial_id
        df_tar["is_fall"] = 1 if activity_label == "falling" else 0
        all_cleaned_targets.append(df_tar)

        cleaning_logs.append({
            "trial": trial_key,
            "stage": "summary",
            "action": "RETAIN_VALID_TRIAL",
            "reason": "Valid trial processed cleanly",
            "details": f"{len(df_tar)} target frames, {len(df_176_ready) if len(df_176) > 0 else 0} ready windows"
        })

    # Combine cleaned data
    df_targets_all = pd.concat(all_cleaned_targets, ignore_index=True) if all_cleaned_targets else pd.DataFrame()
    df_windows_all = pd.concat(all_cleaned_windows, ignore_index=True) if all_cleaned_windows else pd.DataFrame()

    # Save cleaned files
    cleaned_targets_path = cleaned_dir / "cleaned_targets.csv"
    cleaned_windows_path = cleaned_dir / "cleaned_windows_176.csv"
    cleaning_log_path = cleaned_dir / "cleaning_log.csv"

    df_targets_all.to_csv(cleaned_targets_path, index=False)
    df_windows_all.to_csv(cleaned_windows_path, index=False)

    df_logs = pd.DataFrame(cleaning_logs)
    df_logs.to_csv(cleaning_log_path, index=False)

    print(f"Cleaned targets saved to {cleaned_targets_path} ({len(df_targets_all)} rows)")
    print(f"Cleaned 176 windows saved to {cleaned_windows_path} ({len(df_windows_all)} rows)")
    print(f"Cleaning log saved to {cleaning_log_path} ({len(df_logs)} operations)")

    return df_windows_all, cleaning_logs


def main():
    clean_dataset()


if __name__ == "__main__":
    main()
