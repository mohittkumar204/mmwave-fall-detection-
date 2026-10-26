"""Unit Tests for Trained Model Serialization, Loading, and Consistency."""

import json
from pathlib import Path
import joblib
import numpy as np
import pytest

from src.features.feature_pipeline import extract_statistical_features_from_window_176
from src.inference.offline_inference import FallInferenceEngine


def test_model_and_scaler_loading():
    meta_path = Path("artifacts/best_model_metadata.json")
    scaler_path = Path("artifacts/scaler.joblib")
    model_path = Path("models/best_fall_model.joblib")

    assert meta_path.exists(), "Missing artifacts/best_model_metadata.json"
    assert scaler_path.exists(), "Missing artifacts/scaler.joblib"
    assert model_path.exists(), "Missing models/best_fall_model.joblib"

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["best_approach"] in ["Approach_A_Statistical", "Approach_B_176"]
    assert len(meta["feature_names"]) == meta["input_dim"]

    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)
    assert hasattr(scaler, "transform")
    assert hasattr(model, "predict")


def test_inference_engine_consistency():
    engine = FallInferenceEngine()
    fake_window = np.zeros(176, dtype=np.float32)

    pred1 = engine.predict_window(fake_window)
    pred2 = engine.predict_window(fake_window)

    assert pred1["prediction"] in [0, 1]
    assert pred1["class_name"] in ["FALL", "NO_FALL"]
    assert pred1["prediction"] == pred2["prediction"], "Inference predictions must be deterministic"
    assert pred1["latency_ms"] >= 0.0
