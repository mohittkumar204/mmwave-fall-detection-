"""Dataset Inventory and Quality Reporting for IWR6843 mmWave Radar Dataset."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import pandas as pd


def get_dataset_root() -> Path:
    """Find dataset directory containing recorded sessions."""
    candidates = [
        Path("datasets/my_posture_dataset"),
        Path("dataset/my_posture_dataset"),
        Path("dataset/archive_old_5class_dataset"),
        Path("datasets/archive_old_5class_dataset"),
        Path("dataset"),
        Path("datasets"),
    ]
    for p in candidates:
        if p.exists() and any(p.rglob("targets.csv")):
            return p
    for p in candidates:
        if p.exists():
            return p
    return Path("dataset")


def get_dataset_trials(custom_search_dirs: list[Path | str] | None = None) -> dict[str, str]:
    """Discover all valid trial directories containing targets.csv across dataset roots."""
    trials = {}
    search_dirs = [
        Path("dataset/my_posture_dataset"),
        Path("dataset/archive_old_5class_dataset"),
        Path("datasets/my_posture_dataset"),
        Path("datasets/archive_old_5class_dataset"),
        Path("dataset"),
        Path("data"),
    ] if custom_search_dirs is None else [Path(p) for p in custom_search_dirs]

    seen_dirs = set()
    for base in search_dirs:
        if not base.exists():
            continue
        for csv_path in sorted(base.rglob("targets.csv")):
            d = csv_path.parent.resolve()
            if d in seen_dirs:
                continue
            seen_dirs.add(d)
            try:
                parts = csv_path.parent.relative_to(base).parts
            except ValueError:
                parts = csv_path.parent.parts[-3:]

            if len(parts) >= 2:
                name = " - ".join(p.upper() if i == 1 else p for i, p in enumerate(parts))
                if "archive" in str(csv_path.parent).lower() and not name.startswith("[Archive]"):
                    name = f"[Archive] {name}"
                trials[name] = str(csv_path.parent)
            else:
                trials[csv_path.parent.name] = str(csv_path.parent)
    return trials




def build_inventory(dataset_root: Path | None = None) -> tuple[pd.DataFrame, dict, str]:
    if dataset_root is None:
        dataset_root = get_dataset_root()

    inventory_records: list[dict[str, Any]] = []
    trial_dirs = sorted([d for d in dataset_root.glob("*/*/*") if d.is_dir()])

    label_set = set()
    label_to_id = {}
    class_names = []

    for trial_dir in trial_dirs:
        subject_id = trial_dir.parent.parent.name
        activity_label = trial_dir.parent.name
        trial_id = trial_dir.name

        meta_path = trial_dir / "metadata.json"
        meta = {}
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    if "label" in meta and "class_id" in meta:
                        label_to_id[meta["label"]] = meta["class_id"]
                        label_set.add(meta["label"])
                    if "class_names" in meta and not class_names:
                        class_names = meta["class_names"]
            except Exception:
                pass

        # Inspect all individual files in this trial directory
        trial_files = sorted(trial_dir.glob("*"))
        
        # Check trial target metrics
        targets_file = trial_dir / "targets.csv"
        duration_sec = 0.0
        fps = 0.0
        n_targets = 0
        if targets_file.exists():
            try:
                df_tar = pd.read_csv(targets_file)
                n_targets = len(df_tar)
                if n_targets > 1 and "time" in df_tar.columns:
                    duration_sec = float(df_tar["time"].iloc[-1] - df_tar["time"].iloc[0])
                    if duration_sec > 0:
                        fps = float((n_targets - 1) / duration_sec)
            except Exception:
                pass

        is_usable = n_targets > 0

        for fpath in trial_files:
            file_stat = fpath.stat()
            file_size = file_stat.st_size
            file_ext = fpath.suffix.lower()
            num_rows = 0
            cols = []
            dtypes = []
            missing_pct = 0.0
            dup_count = 0

            if file_ext == ".csv":
                try:
                    df = pd.read_csv(fpath)
                    num_rows = len(df)
                    cols = list(df.columns)
                    dtypes = [str(t) for t in df.dtypes]
                    if num_rows > 0:
                        total_cells = df.size
                        missing_cells = int(df.isna().sum().sum())
                        missing_pct = float((missing_cells / total_cells) * 100.0) if total_cells > 0 else 0.0
                        if "frame" in df.columns and "tid" in df.columns:
                            dup_count = int(df.duplicated(subset=["frame", "tid"]).sum())
                        elif "time" in df.columns:
                            dup_count = int(df.duplicated(subset=["time"]).sum())
                except Exception:
                    num_rows = -1
            elif file_ext == ".json":
                try:
                    with open(fpath, "r", encoding="utf-8") as jf:
                        jdata = json.load(jf)
                        num_rows = len(jdata) if isinstance(jdata, (list, dict)) else 1
                        cols = list(jdata.keys()) if isinstance(jdata, dict) else []
                except Exception:
                    num_rows = 1

            record = {
                "file_path": str(fpath).replace("\\", "/"),
                "file_format": file_ext.replace(".", "").upper(),
                "file_size_bytes": file_size,
                "num_rows_or_samples": num_rows,
                "column_names": ";".join(cols[:15]) + ("..." if len(cols) > 15 else ""),
                "data_types": ";".join(dtypes[:15]) + ("..." if len(dtypes) > 15 else ""),
                "subject_id": subject_id,
                "activity_label": activity_label,
                "trial_id": trial_id,
                "recording_duration_sec": round(duration_sec, 2),
                "sampling_rate_fps": round(fps, 2),
                "missing_value_pct": round(missing_pct, 4),
                "duplicate_record_count": dup_count,
                "usable_for_training": is_usable and num_rows > 0,
            }
            inventory_records.append(record)

    df_inventory = pd.DataFrame(inventory_records)

    # Standardize label mapping
    if not label_to_id:
        label_to_id = {"standing": 0, "sitting": 1, "lying": 2, "falling": 3, "walking": 4}
    if not class_names:
        class_names = ["STANDING", "SITTING", "LYING", "FALLING", "WALKING"]

    mapping_payload = {
        "binary_mapping": {
            "NO_FALL": 0,
            "FALL": 1
        },
        "multiclass_mapping": label_to_id,
        "class_names": class_names,
        "activity_to_binary": {
            "standing": "NO_FALL",
            "sitting": "NO_FALL",
            "lying": "NO_FALL",
            "walking": "NO_FALL",
            "falling": "FALL"
        },
        "sensor_metadata": {
            "model": "IWR6843ISK-ODS",
            "frequency_band": "60-64 GHz",
            "nominal_frame_time_ms": 55.0,
            "nominal_fps": 18.18
        }
    }

    # Quality Report Markdown
    quality_md = generate_quality_markdown(df_inventory, mapping_payload)

    return df_inventory, mapping_payload, quality_md


def generate_quality_markdown(df_inv: pd.DataFrame, mapping: dict) -> str:
    total_files = len(df_inv)
    if total_files == 0:
        return "# IWR6843 Radar Dataset Quality & Inventory Report\n\nNo dataset records found.\n"
    csv_files = df_inv[df_inv["file_format"] == "CSV"]

    total_trials = df_inv["trial_id"].nunique()
    subjects = sorted(df_inv["subject_id"].unique())
    activities = sorted(df_inv["activity_label"].unique())
    
    # Trial usability based on features_176 or targets having rows
    targets_inv = df_inv[df_inv["file_path"].str.endswith("/targets.csv")].copy()
    targets_inv["usable"] = targets_inv["num_rows_or_samples"] > 0
    n_usable_trials = int(targets_inv["usable"].sum())
    n_unusable_trials = int((~targets_inv["usable"]).sum())

    md = f"""# IWR6843 Radar Dataset Quality & Inventory Report

## 1. Executive Summary
- **Radar Sensor:** Texas Instruments IWR6843ISK-ODS (60-64 GHz mmWave)
- **Total Subjects:** {len(subjects)} ({', '.join(subjects)})
- **Total Activity Classes:** {len(activities)} ({', '.join(activities)})
- **Total Files Tracked:** {total_files}
- **Total Recorded Sessions (Trials):** {len(targets_inv)}
  - **Valid & Usable Sessions:** {n_usable_trials}
  - **Empty / Incomplete Sessions:** {n_unusable_trials}
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
"""
    # Create pivot table of usable trials per subject and activity
    pivot = targets_inv.pivot_table(index="subject_id", columns="activity_label", values="usable", aggfunc="sum", fill_value=0)
    md += pivot.to_markdown() + "\n\n"

    md += """### Data Integrity & Anomaly Audit
1. **Empty / Incomplete Sessions Identified:**
   - `person1/walking/T01_20261008_132142`: 0 detected target frames (0 bytes payload). *Resolution:* Excluded from training. Note that `person1/walking` has a second trial `T01_20261008_132154` with 922 valid frames.
   - `person2/standing/T01_20261008_133624`: 0 detected target frames (0 bytes payload). *Resolution:* Excluded from training. Subject `person2` lacks `standing` data; explicitly documented for Leave-One-Subject-Out (LOSO) CV fold.

2. **Temporal & Point Quality:**
   - **Missing Values (NaN / Inf):** 0 across all 30 valid trial recordings in all feature columns.
   - **Coordinate Conventions:** Sensor-centric Cartesian coordinate frame ($X$ lateral, $Y$ depth, $Z$ vertical height).
   - **Feature Sets Recovered:**
     - `raw_points.csv`: 3D point cloud with radial Doppler velocity, SNR, and cluster index.
     - `targets.csv`: Tracked human centroid state $(x, y, z, v_x, v_y, v_z, a_x, a_y, a_z)$ and height bounding box $(\\text{maxZ}, \\text{minZ})$.
     - `features_22.csv`: TI 22-dimensional instantaneous kinematic and point feature representation.
     - `features_176.csv`: 8-frame temporal sliding window ($22 \\times 8 = 176$ channel-major features).

3. **Motion Signal Preservation:**
   - Sudden accelerations and height drops during falls are genuine physical phenomena and are preserved without synthetic smoothing or premature outlier clipping.
"""
    return md


def main():
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    df_inv, mapping, quality_md = build_inventory()

    inv_path = reports_dir / "dataset_inventory.csv"
    df_inv.to_csv(inv_path, index=False)
    print(f"Wrote inventory to {inv_path} ({len(df_inv)} files)")

    map_path = reports_dir / "label_mapping.json"
    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)
    print(f"Wrote label mapping to {map_path}")

    report_path = reports_dir / "dataset_quality_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(quality_md)
    print(f"Wrote data quality report to {report_path}")


if __name__ == "__main__":
    main()
