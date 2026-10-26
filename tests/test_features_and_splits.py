"""Unit Tests for Feature Extraction, Window Generation, and LOSO Splitting."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.features.feature_pipeline import (
    FEATURE_NAMES_176,
    FEATURE_NAMES_22,
    WINDOW_SIZE,
    extract_statistical_features_from_window_176,
)


def test_window_feature_extraction_dimensions():
    # Synthetic 176-feature window vector
    fake_vector = np.random.randn(176).astype(np.float32)
    stat_feats = extract_statistical_features_from_window_176(fake_vector)

    assert isinstance(stat_feats, dict)
    assert "posz_mean" in stat_feats
    assert "posz_drop" in stat_feats
    assert "velz_min" in stat_feats
    assert "point_z_spread" in stat_feats

    # Ensure no NaN or Inf
    for k, val in stat_feats.items():
        assert not np.isnan(val), f"Feature {k} returned NaN"
        assert not np.isinf(val), f"Feature {k} returned Inf"


def test_zero_variance_numerical_stability():
    # All zeros (flat line / stationary target)
    zero_vector = np.zeros(176, dtype=np.float32)
    stat_feats = extract_statistical_features_from_window_176(zero_vector)

    assert stat_feats["posz_mean"] == 0.0
    assert stat_feats["posz_std"] == 0.0
    assert stat_feats["posz_skew"] == 0.0
    assert stat_feats["posz_kurt"] == 0.0
    assert not np.isnan(stat_feats["posz_skew"])


def test_loso_splits_integrity():
    splits_file = Path("data/splits/loso_splits.json")
    assert splits_file.exists(), "Missing data/splits/loso_splits.json"
    with open(splits_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["num_folds"] == 6
    assert len(manifest["folds"]) == 6

    for fold in manifest["folds"]:
        test_subj = fold["test_subject"]
        train_subjs = fold["train_subjects"]
        assert test_subj not in train_subjs, f"Data leakage in fold {fold['fold']}: {test_subj} in train"
        assert len(train_subjs) == 5, f"Fold {fold['fold']} must have exactly 5 train subjects"
