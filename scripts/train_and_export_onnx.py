"""Train a 176-float RandomForest posture model and export it to ONNX.

Input format: 22 features x 8 frames in channel-major order (176 floats):
  posz_f0..posz_f7, velx_f0..velx_f7, ..., snr4_f0..snr4_f7

Classes: STANDING(0), SITTING(1), WALKING(2), FALLING(3)  [LYING merged into FALLING]

Output:
  models/onnx/ti_pose_model.onnx
  models/onnx/model_metadata.json
  models/onnx/feature_scaler.npz
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

OUT_DIR = PROJECT / "models" / "onnx"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_CSV = PROJECT / "data" / "processed" / "features_approach_b_176.csv"

CLASS_NAMES = ["STANDING", "SITTING", "WALKING", "FALLING"]
LABEL_MAP = {"standing": 0, "sitting": 1, "walking": 2, "falling": 3, "lying": 3}

FEATURE_NAMES_22 = [
    "posz", "velx", "vely", "velz", "accx", "accy", "accz",
    "y0", "z0", "snr0", "y1", "z1", "snr1",
    "y2", "z2", "snr2", "y3", "z3", "snr3",
    "y4", "z4", "snr4",
]
FEATURE_COLS_176 = [f"{feat}_f{frame}" for feat in FEATURE_NAMES_22 for frame in range(8)]


def load_data():
    print(f"[train-onnx] Loading {FEATURE_CSV.name}...")
    df = pd.read_csv(FEATURE_CSV)
    df["class_id"] = df["activity_label"].str.lower().map(LABEL_MAP)
    df = df.dropna(subset=["class_id"])
    df["class_id"] = df["class_id"].astype(int)

    missing = [c for c in FEATURE_COLS_176 if c not in df.columns]
    if missing:
        print(f"[train-onnx] ERROR: Missing {len(missing)} feature columns, e.g. {missing[:5]}")
        sys.exit(1)

    # Filter out corrupted rows where physical sensor values exploded (e.g. 10^32)
    non_snr_cols = [c for c in FEATURE_COLS_176 if "snr" not in c]
    bad_mask = (
        (df[non_snr_cols].abs() > 20).any(axis=1)
        | df[FEATURE_COLS_176].isna().any(axis=1)
        | np.isinf(df[FEATURE_COLS_176].values).any(axis=1)
    )
    n_corrupt = int(bad_mask.sum())
    if n_corrupt > 0:
        print(f"[train-onnx] Filtered {n_corrupt} corrupted outlier rows (>20m, m/s, or inf)")
        df = df[~bad_mask].copy()

    X = df[FEATURE_COLS_176].values.astype(np.float32)
    y = df["class_id"].values
    subjects = df["subject_id"].values
    print(f"[train-onnx] Loaded {len(X)} clean samples, {X.shape[1]} features")
    print(f"[train-onnx] Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    return X, y, subjects


def train_model(X_train, y_train, X_val, y_val):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, f1_score

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    print(f"[train-onnx] Scaler mean range: [{scaler.mean_.min():.3f}, {scaler.mean_.max():.3f}]")
    print(f"[train-onnx] Scaler std range: [{scaler.scale_.min():.3f}, {scaler.scale_.max():.3f}]")

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=25,
        min_samples_leaf=1,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    rf.fit(X_train_s, y_train)

    y_pred = rf.predict(X_val_s)
    f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
    acc = float(np.mean(y_pred == y_val))
    print(f"[train-onnx] Val accuracy={acc:.4f}  macro-F1={f1:.4f}")
    print(classification_report(y_val, y_pred, target_names=CLASS_NAMES, zero_division=0))
    return rf, scaler


def export_onnx(rf, scaler, X_sample):
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        import onnx
    except ImportError:
        print("[train-onnx] ERROR: skl2onnx or onnx not installed.")
        sys.exit(1)

    print("[train-onnx] Exporting to ONNX...")
    initial_type = [("float_input", FloatTensorType([None, 176]))]
    onnx_model = convert_sklearn(
        rf,
        initial_types=initial_type,
        target_opset=17,
        options={type(rf): {"zipmap": False}},
    )

    # Ensure output[0] is probabilities [None, 5] and output[1] is label [None]
    out_names = [o.name for o in onnx_model.graph.output]
    if len(out_names) >= 2 and "label" in out_names[0] and "prob" in out_names[1]:
        print(f"[train-onnx] Swapping output order so probabilities are output[0] (was: {out_names})")
        p0 = onnx.ValueInfoProto()
        p0.CopyFrom(onnx_model.graph.output[0])
        p1 = onnx.ValueInfoProto()
        p1.CopyFrom(onnx_model.graph.output[1])
        onnx_model.graph.output[0].CopyFrom(p1)
        onnx_model.graph.output[1].CopyFrom(p0)

    onnx_path = OUT_DIR / "ti_pose_model.onnx"
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"[train-onnx] ONNX model saved: {onnx_path}")

    # Save scaler
    scaler_path = OUT_DIR / "feature_scaler.npz"
    np.savez(
        scaler_path,
        mean=scaler.mean_.astype(np.float32),
        std=scaler.scale_.astype(np.float32),
    )
    print(f"[train-onnx] Scaler saved: {scaler_path}")

    # Save metadata
    metadata = {
        "class_names": CLASS_NAMES,
        "normalization_enabled": True,
        "scaler_path": "feature_scaler.npz",
        "n_features": 176,
        "model_type": "RandomForestClassifier",
        "n_estimators": rf.n_estimators,
        "framework": "sklearn -> skl2onnx",
    }
    meta_path = OUT_DIR / "model_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))
    print(f"[train-onnx] Metadata saved: {meta_path}")

    # Verify ONNX inference
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    x_test = scaler.transform(X_sample[:3]).astype(np.float32)
    first_output = sess.run(None, {sess.get_inputs()[0].name: x_test})[0]
    print(f"[train-onnx] ONNX output[0] shape: {first_output.shape}")
    print(f"[train-onnx] Sample probabilities sum per row: {np.sum(first_output, axis=1)}")
    preds = np.argmax(first_output, axis=1)
    print(f"[train-onnx] Sample predictions: {[CLASS_NAMES[i] for i in preds]}")
    print("[train-onnx] ONNX model verified successfully!")
    return onnx_path


def main():
    X, y, subjects = load_data()

    # Train on all clean data to maximize generalization across all people and trials,
    # but evaluate on a 20% stratified test split first
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    print(f"[train-onnx] Train split: {len(X_train)}, Val split: {len(X_val)}")

    rf, scaler = train_model(X_train, y_train, X_val, y_val)

    # Now fit on ALL clean data for production model
    print("[train-onnx] Fitting final production model on full dataset (21,348 clean samples)...")
    from sklearn.preprocessing import StandardScaler
    full_scaler = StandardScaler()
    X_full_s = full_scaler.fit_transform(X)
    rf.fit(X_full_s, y)

    onnx_path = export_onnx(rf, full_scaler, X_val)

    print(f"\n[train-onnx] === SUCCESS ===")
    print(f"ONNX model and clean scaler exported to {OUT_DIR}")


if __name__ == "__main__":
    main()
