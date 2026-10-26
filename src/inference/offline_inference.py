"""Offline Inference Engine for IWR6843 Fall Detection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd

from src.features.feature_pipeline import (
    FEATURE_NAMES_176,
    FEATURE_NAMES_22,
    WINDOW_SIZE,
    extract_statistical_features_from_window_176,
)


class FallInferenceEngine:
    def __init__(
        self,
        metadata_path: str | Path = "artifacts/best_model_metadata.json",
        scaler_path: str | Path = "artifacts/scaler.joblib",
        model_path: str | Path = "models/best_fall_model.joblib",
    ):
        self.metadata_path = Path(metadata_path)
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.feature_names = self.metadata["feature_names"]
        self.approach = self.metadata["best_approach"]

        self.scaler = joblib.load(scaler_path)
        self.model = joblib.load(model_path)

    def predict_window(self, window_176: np.ndarray) -> dict[str, Any]:
        """Predict fall status for a single 176-float window."""
        t0 = time.perf_counter()
        if self.approach == "Approach_A_Statistical":
            stat_feats = extract_statistical_features_from_window_176(window_176)
            feat_vector = np.array([[stat_feats[k] for k in self.feature_names]], dtype=np.float32)
        else:
            feat_vector = window_176.reshape(1, -1)

        X_scaled = self.scaler.transform(feat_vector)
        pred_label = int(self.model.predict(X_scaled)[0])

        proba = None
        if hasattr(self.model, "predict_proba"):
            proba = float(self.model.predict_proba(X_scaled)[0, 1])

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "prediction": pred_label,
            "class_name": "FALL" if pred_label == 1 else "NO_FALL",
            "confidence": proba,
            "latency_ms": latency_ms,
        }

    def run_on_trial(self, trial_dir: str | Path, output_csv: Path | None = None) -> pd.DataFrame:
        trial_dir = Path(trial_dir)
        f176_path = trial_dir / "features_176.csv"
        if not f176_path.exists():
            raise FileNotFoundError(f"features_176.csv not found in {trial_dir}")

        df_176 = pd.read_csv(f176_path)
        if len(df_176) == 0:
            return pd.DataFrame()

        # Filter ready windows
        ready_df = df_176[df_176["ready"] == 1].copy() if "ready" in df_176.columns else df_176.copy()
        raw_cols = FEATURE_NAMES_176

        X_raw = ready_df[raw_cols].to_numpy(dtype=np.float32)

        results = []
        for i in range(len(ready_df)):
            row_meta = ready_df.iloc[i]
            pred = self.predict_window(X_raw[i])
            results.append({
                "frame": int(row_meta.get("frame", i)),
                "time": float(row_meta.get("time", 0.0)),
                "tid": int(row_meta.get("tid", 0)),
                "prediction": pred["prediction"],
                "class_name": pred["class_name"],
                "confidence": pred["confidence"],
                "latency_ms": pred["latency_ms"],
            })

        df_preds = pd.DataFrame(results)
        if output_csv:
            output_csv.parent.mkdir(parents=True, exist_ok=True)
            df_preds.to_csv(output_csv, index=False)
            print(f"Saved predictions to {output_csv}")

        return df_preds


def main():
    parser = argparse.ArgumentParser(description="Offline IWR6843 Fall Detection Inference")
    parser.add_argument("--trial", default="datasets/my_posture_dataset/person1/falling/T01_20261008_132511", help="Path to trial directory")
    parser.add_argument("--out", default="reports/sample_offline_predictions.csv", help="Output predictions CSV")
    args = parser.parse_args()

    engine = FallInferenceEngine()
    print(f"Running inference on trial: {args.trial}")
    preds = engine.run_on_trial(args.trial, Path(args.out))

    if len(preds) > 0:
        fall_count = (preds["prediction"] == 1).sum()
        total_count = len(preds)
        print(f"\nCompleted offline inference across {total_count} windows.")
        print(f"Detected FALL windows: {fall_count} / {total_count} ({fall_count/total_count*100:.1f}%)")
        print(f"Average latency per window: {preds['latency_ms'].mean():.3f} ms")
        if fall_count > 0:
            fall_frames = preds[preds["prediction"] == 1]
            first_fall = fall_frames.iloc[0]
            print(f"First fall detected at frame {first_fall['frame']} (time: {first_fall['time']:.2f}s, confidence: {first_fall['confidence']:.2%})")


if __name__ == "__main__":
    main()
