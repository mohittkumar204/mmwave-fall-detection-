"""Unit and Integration Tests for Dataset Discovery and Label Mapping."""

from pathlib import Path
import json
import pytest

from src.data.dataset_inventory import build_inventory, get_dataset_root


def test_dataset_root_discovery():
    root = get_dataset_root()
    assert root.exists(), f"Dataset root does not exist: {root}"
    subjects = [d.name for d in root.iterdir() if d.is_dir() and d.name.startswith("person")]
    assert len(subjects) == 6, f"Expected 6 subjects, found {len(subjects)}: {subjects}"
    assert "person1" in subjects
    assert "person6" in subjects


def test_label_mapping():
    label_path = Path("reports/label_mapping.json")
    assert label_path.exists(), "Missing reports/label_mapping.json"
    with open(label_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    assert "binary_mapping" in mapping
    assert mapping["binary_mapping"]["FALL"] == 1
    assert mapping["binary_mapping"]["NO_FALL"] == 0

    assert "multiclass_mapping" in mapping
    assert mapping["multiclass_mapping"]["falling"] == 3
    assert mapping["multiclass_mapping"]["standing"] == 0

    assert "sensor_metadata" in mapping
    assert mapping["sensor_metadata"]["model"] == "IWR6843ISK-ODS"


def test_inventory_generation():
    df_inv, mapping, q_md = build_inventory()
    assert len(df_inv) >= 192, f"Expected at least 192 files across trials, got {len(df_inv)}"
    assert "usable_for_training" in df_inv.columns
    # Check that empty trials are flagged as not usable for training
    unusable = df_inv[~df_inv["usable_for_training"]]
    assert len(unusable) > 0, "Expected empty/unusable trials to be flagged"
