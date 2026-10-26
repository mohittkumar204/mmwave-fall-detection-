"""Leave-One-Subject-Out (LOSO) Split Generator to prevent data leakage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd


def generate_loso_splits(
    data_csv: str | Path = "data/processed/features_approach_b_176.csv",
    splits_dir: str | Path = "data/splits"
) -> dict[str, Any]:
    splits_dir = Path(splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_csv)
    subjects = sorted(df["subject_id"].unique())
    print(f"Generating LOSO splits for subjects: {subjects} across {len(df)} samples")

    folds_info = []

    for fold_idx, test_subject in enumerate(subjects, start=1):
        test_mask = df["subject_id"] == test_subject
        train_mask = ~test_mask

        train_subjects = [s for s in subjects if s != test_subject]

        test_indices = df.index[test_mask].tolist()
        train_indices = df.index[train_mask].tolist()

        test_classes = df.loc[test_mask, "activity_label"].value_counts().to_dict()
        train_classes = df.loc[train_mask, "activity_label"].value_counts().to_dict()

        test_binary = df.loc[test_mask, "is_fall"].value_counts().to_dict()
        train_binary = df.loc[train_mask, "is_fall"].value_counts().to_dict()

        notes = []
        if test_subject == "person2" and "standing" not in test_classes:
            notes.append("Subject person2 has 0 standing samples due to empty session.")

        fold_meta = {
            "fold": fold_idx,
            "test_subject": test_subject,
            "train_subjects": train_subjects,
            "num_train_samples": len(train_indices),
            "num_test_samples": len(test_indices),
            "train_classes": train_classes,
            "test_classes": test_classes,
            "train_binary_fall": train_binary,
            "test_binary_fall": test_binary,
            "notes": " ".join(notes) if notes else "Full 5-class coverage in test set."
        }
        folds_info.append(fold_meta)

    splits_manifest = {
        "strategy": "Leave-One-Subject-Out (LOSO) Cross-Validation",
        "num_folds": len(folds_info),
        "total_subjects": len(subjects),
        "subjects": subjects,
        "folds": folds_info
    }

    out_json = splits_dir / "loso_splits.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(splits_manifest, f, indent=2)

    print(f"Saved LOSO splits manifest to {out_json}")
    return splits_manifest


def main():
    generate_loso_splits()


if __name__ == "__main__":
    main()
