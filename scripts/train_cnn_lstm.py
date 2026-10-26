"""Hybrid 1D CNN-LSTM Training Pipeline on 9 Kinematic Radar Features.

Trained on 3-Class Robust Posture & Fall Detection:
  0: UPRIGHT  (standing, sitting, walking)
  1: LYING    (person flat on floor / bed)
  2: FALLING  (rapid downward collapse & impact transition)

Features (9 Kinematic Channels per Frame):
  [x, y, z, vx, vy, vz, ax, ay, az]
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import shutil
import sys

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CLASS_NAMES_3 = ["UPRIGHT", "LYING", "FALLING"]
LABEL_MAP_3 = {
    "upright": 0,
    "standing": 0,
    "sitting": 0,
    "walking": 0,
    "stand": 0,
    "sit": 0,
    "walk": 0,
    "lying": 1,
    "lay": 1,
    "lie": 1,
    "falling": 2,
    "fall": 2,
    "collapsed": 2,
}

KINEMATIC_FEATURES = ["x", "y", "z", "vx", "vy", "vz", "ax", "ay", "az"]


class KinematicCNNLSTM(nn.Module):
    """Hybrid 1D-CNN + LSTM for temporal radar fall & posture recognition."""

    def __init__(
        self,
        in_channels: int = 9,
        seq_len: int = 8,
        conv_filters: int = 64,
        lstm_hidden: int = 64,
        num_classes: int = 3,
        dropout: float = 0.25,
    ):
        super().__init__()
        # 1D-CNN temporal convolution over the 9 channels
        # Input shape: (Batch, Channels=9, SeqLen=8)
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=conv_filters,
            kernel_size=3,
            padding=1,
        )
        self.bn1 = nn.BatchNorm1d(conv_filters)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)  # seq_len 8 -> 4
        self.drop = nn.Dropout(dropout)

        # LSTM Layer: processes sequential features across time slices
        # Transposed shape: (Batch, SeqLen=4, Features=conv_filters)
        self.lstm = nn.LSTM(
            input_size=conv_filters,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            dropout=dropout if dropout > 0 else 0,
            bidirectional=False,
        )

        # Dense Classifier Head
        self.fc1 = nn.Linear(lstm_hidden, 32)
        self.fc_drop = nn.Dropout(dropout)
        self.out = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is (Batch, SeqLen, Channels) -> transpose to (Batch, Channels, SeqLen) for Conv1D
        x = x.transpose(1, 2)
        c = self.relu(self.bn1(self.conv1(x)))
        p = self.pool(c)  # (Batch, conv_filters, 4)
        p = self.drop(p)

        # Transpose to (Batch, SeqLen=4, Features=conv_filters) for LSTM
        lstm_in = p.transpose(1, 2)
        lstm_out, _ = self.lstm(lstm_in)

        # Last time-step representation
        last_step = lstm_out[:, -1, :]
        h = self.relu(self.fc1(last_step))
        h = self.fc_drop(h)
        logits = self.out(h)
        return logits


def find_dataset_files(data_dir: Path) -> list[tuple[Path, str, str, str]]:
    """Scan data_dir flexibly for targets.csv or alternative recording files.
    
    Returns list of (file_path, subject_name, activity_name, session_id).
    """
    results = []
    candidates = list(data_dir.rglob("targets.csv"))
    if not candidates:
        candidates = list(data_dir.rglob("features_22.csv"))

    for fpath in candidates:
        parts = fpath.relative_to(data_dir).parts
        if len(parts) >= 2:
            subject = parts[0]
            activity = parts[1].lower()
            trial = parts[2] if len(parts) >= 3 else fpath.parent.name
            session_id = f"{subject}_{activity}_{trial}"
            results.append((fpath, subject, activity, session_id))
    return results


def load_sliding_windows(
    data_dir: Path,
    window_size: int = 8,
    class_names: list[str] = CLASS_NAMES_3,
    label_map: dict[str, int] = LABEL_MAP_3,
):
    print(f"[data] Scanning dataset in: {data_dir.resolve()}")
    matched_entries = find_dataset_files(data_dir)
    print(f"[data] Found {len(matched_entries)} recording sessions.")

    if len(matched_entries) == 0:
        print("[data] WARNING: No recording files (targets.csv) found in the dataset directory!")
        print(f"[data] Please ensure files are placed under: {data_dir} / <person> / <activity> / targets.csv")
        return None, None, None, None

    all_windows = []
    all_labels = []
    all_subjects = []
    all_sessions = []

    for fpath, subject_name, activity_name, session_id in matched_entries:
        if activity_name not in label_map:
            # Check if any key is in activity_name
            found_key = None
            for k in label_map:
                if k in activity_name:
                    found_key = k
                    break
            if found_key is not None:
                label_id = label_map[found_key]
            else:
                continue
        else:
            label_id = label_map[activity_name]

        try:
            df = pd.read_csv(fpath)
            if df.empty:
                continue
        except Exception as e:
            print(f"[data] Skipping unreadable file {fpath}: {e}")
            continue

        # Check required columns
        missing_cols = [c for c in KINEMATIC_FEATURES if c not in df.columns]
        if missing_cols:
            # Attempt fallback from pos_x, pos_y, etc. if prefixed
            rename_dict = {f"pos_{c}": c for c in ["x", "y", "z"]}
            rename_dict.update({f"vel_{c}": f"v{c}" for c in ["x", "y", "z"]})
            rename_dict.update({f"acc_{c}": f"a{c}" for c in ["x", "y", "z"]})
            df = df.rename(columns=rename_dict)
            missing_cols = [c for c in KINEMATIC_FEATURES if c not in df.columns]
            if missing_cols:
                continue

        # Group by Target ID (tid) so tracks aren't mixed
        groupby_col = "tid" if "tid" in df.columns else None
        groups = df.groupby(groupby_col) if groupby_col else [(0, df)]

        # Stride: 1 for falling (maximizes fall windows), 2 for upright/lying (prevents static flooding)
        stride = 1 if label_id == label_map.get("falling", 2) else 2

        for _, t_df in groups:
            if "frame" in t_df.columns:
                t_df = t_df.sort_values("frame").reset_index(drop=True)
            if len(t_df) < window_size:
                continue

            # Extract 9 features
            data_mat = t_df[KINEMATIC_FEATURES].values.astype(np.float32)

            for i in range(0, len(data_mat) - window_size + 1, stride):
                win = data_mat[i : i + window_size]  # (window_size, 9)
                # Filter out corrupt exploding values (> 25 m or m/s or NaN/Inf)
                if np.abs(win).max() > 25.0 or np.isnan(win).any() or np.isinf(win).any():
                    continue

                all_windows.append(win)
                all_labels.append(label_id)
                all_subjects.append(subject_name)
                all_sessions.append(session_id)

    if not all_windows:
        print("[data] ERROR: No valid sliding windows could be extracted.")
        return None, None, None, None

    X = np.array(all_windows, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int64)
    subjects = np.array(all_subjects)
    sessions = np.array(all_sessions)

    print(f"[data] Total sliding windows extracted: {len(X)}")
    print(f"[data] Shape: X={X.shape}, y={y.shape}")
    dist = {class_names[i]: int((y == i).sum()) for i in range(len(class_names))}
    print(f"[data] Class distribution: {dist}")
    return X, y, subjects, sessions


def train(
    data_dir: str | Path = PROJECT_ROOT / "dataset" / "my_posture_dataset",
    epochs: int = 35,
    batch_size: int = 64,
    lr: float = 0.001,
    window_size: int = 8,
    out_dir: Path = PROJECT_ROOT / "models",
    split_by_session: bool = True,
):
    data_dir = Path(data_dir)
    class_names = CLASS_NAMES_3
    label_map = LABEL_MAP_3
    num_classes = len(class_names)

    X, y, subjects, sessions = load_sliding_windows(
        data_dir=data_dir,
        window_size=window_size,
        class_names=class_names,
        label_map=label_map,
    )

    if X is None or len(X) < 10:
        print("\n[train] Dataset does not yet have enough collected samples.")
        print("[train] Please collect dataset sessions into:")
        for p in range(1, 7):
            print(f"  {data_dir}/person {p}/[upright|lying|falling]/targets.csv")
        print("\nOnce data is collected, re-run this command to train the model!")
        return False

    # Check class variety
    present_classes = np.unique(y)
    if len(present_classes) < 2:
        print(f"[train] ERROR: Dataset only contains class(es) {present_classes}. At least 2 classes needed.")
        return False

    # Train / Val Split
    if split_by_session and sessions is not None:
        from sklearn.model_selection import GroupShuffleSplit
        print("[train] Performing Leakage-Resistant Grouped Split by Recording Session...")
        gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
        train_idx, val_idx = next(gss.split(X, y, groups=sessions))
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        val_sess = np.unique(sessions[val_idx])
        print(f"[train] Held-out validation sessions ({len(val_sess)}): {list(val_sess)}")
    else:
        # Standard window-level split (historical reproduction)
        try:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.15, random_state=42, stratify=y
            )
        except Exception:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.15, random_state=42, shuffle=True
            )

    print(f"[train] Train samples: {len(X_train)}, Val samples: {len(X_val)}")
    tr_dist = {class_names[i]: int((y_train == i).sum()) for i in range(len(class_names))}
    val_dist = {class_names[i]: int((y_val == i).sum()) for i in range(len(class_names))}
    print(f"[train] Train class distribution: {tr_dist}")
    print(f"[train] Val class distribution:   {val_dist}")

    # Fit 2D StandardScaler on flattened (N*window_size, 9) strictly from training partition
    N_tr, T, F = X_train.shape
    scaler = StandardScaler()
    X_train_flat = X_train.reshape(-1, F)
    scaler.fit(X_train_flat)

    X_train_norm = scaler.transform(X_train_flat).reshape(N_tr, T, F).astype(np.float32)
    X_val_norm = scaler.transform(X_val.reshape(-1, F)).reshape(X_val.shape[0], T, F).astype(np.float32)

    # Balanced class weights to prevent fall under-detection (calculated strictly from training partition)
    class_counts = np.bincount(y_train, minlength=num_classes)
    total_samples = len(y_train)
    class_weights = total_samples / (float(num_classes) * np.maximum(class_counts, 1))
    # Give slight extra priority to FALLING class
    fall_idx = label_map.get("falling", 2)
    if fall_idx < len(class_weights):
        class_weights[fall_idx] *= 1.25

    weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
    print(f"[train] Class weights: {weights_tensor.numpy().round(3)}")

    # DataLoaders
    train_dataset = TensorDataset(torch.from_numpy(X_train_norm), torch.from_numpy(y_train))
    val_dataset = TensorDataset(torch.from_numpy(X_val_norm), torch.from_numpy(y_val))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False)

    # Model definition
    model = KinematicCNNLSTM(
        in_channels=9,
        seq_len=window_size,
        conv_filters=64,
        lstm_hidden=64,
        num_classes=num_classes,
        dropout=0.25,
    )
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=4
    )

    best_val_f1 = -1.0
    best_weights = model.state_dict().copy()

    print(f"\n[train] Training Hybrid 1D-CNN + LSTM ({num_classes}-Class: {class_names})...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_x)

        train_loss = total_loss / len(train_dataset)

        # Validation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                out = model(batch_x)
                preds = torch.argmax(out, dim=1).numpy()
                val_preds.extend(preds)
                val_targets.extend(batch_y.numpy())

        val_preds = np.array(val_preds)
        val_targets = np.array(val_targets)
        val_acc = float(np.mean(val_preds == val_targets))
        val_f1 = float(f1_score(val_targets, val_preds, average="macro", zero_division=0))
        scheduler.step(val_f1)

        if val_f1 >= best_val_f1:
            best_val_f1 = val_f1
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == epochs or epoch == 1:
            print(
                f"Epoch {epoch:2d}/{epochs:2d} | Train Loss: {train_loss:.4f} | "
                f"Val Acc: {val_acc:.4f} | Val Macro-F1: {val_f1:.4f}"
            )

    print(f"\n[train] Best Validation Macro-F1: {best_val_f1:.4f}")
    model.load_state_dict(best_weights)
    model.eval()

    # Classification Report
    with torch.no_grad():
        final_preds = []
        for batch_x, _ in val_loader:
            out = model(batch_x)
            final_preds.extend(torch.argmax(out, dim=1).numpy())
    target_names = [class_names[i] for i in sorted(list(set(y_val) | set(final_preds)))]
    print("\n[eval] Validation Set Detailed Report:")
    print(classification_report(y_val, final_preds, target_names=target_names, digits=4, zero_division=0))

    # Save PyTorch Model
    out_dir.mkdir(parents=True, exist_ok=True)
    pt_path = out_dir / "kinematic_cnn_lstm.pt"
    torch.save(model.state_dict(), pt_path)
    print(f"[save] PyTorch model saved to {pt_path}")

    # Save Scaler
    scaler_npz = out_dir / "kinematic_scaler.npz"
    np.savez(scaler_npz, mean=scaler.mean_.astype(np.float32), std=scaler.scale_.astype(np.float32))
    try:
        import joblib
        joblib.dump(scaler, out_dir / "kinematic_scaler.joblib")
    except Exception:
        pass
    print(f"[save] Scaler saved to {scaler_npz}")

    # Export to ONNX
    onnx_dir = out_dir / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = onnx_dir / "kinematic_cnn_lstm.onnx"

    dummy_input = torch.randn(1, window_size, 9, dtype=torch.float32)
    try:
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            input_names=["kinematic_window"],
            output_names=["logits"],
            dynamic_axes={"kinematic_window": {0: "batch_size"}, "logits": {0: "batch_size"}},
            opset_version=14,
            dynamo=False,
        )
        print(f"[save] ONNX model exported to {onnx_path}")
    except TypeError:
        try:
            torch.onnx.export(
                model,
                dummy_input,
                str(onnx_path),
                input_names=["kinematic_window"],
                output_names=["logits"],
                dynamic_axes={"kinematic_window": {0: "batch_size"}, "logits": {0: "batch_size"}},
                opset_version=14,
            )
            print(f"[save] ONNX model exported to {onnx_path}")
        except Exception as e:
            print(f"[save] Warning exporting ONNX: {e}")
    except Exception as e:
        print(f"[save] Warning exporting ONNX: {e}")

    # Save Metadata
    meta = {
        "model_name": "Kinematic_1D_CNN_LSTM",
        "features": KINEMATIC_FEATURES,
        "input_shape": [1, window_size, 9],
        "classes": class_names,
        "num_classes": num_classes,
        "window_size": window_size,
        "nominal_fps": 18.18,
        "window_duration_seconds": round(window_size / 18.18, 3),
        "best_val_macro_f1": float(best_val_f1),
        "pt_model": str(pt_path.relative_to(PROJECT_ROOT)),
        "onnx_model": str(onnx_path.relative_to(PROJECT_ROOT)),
        "scaler": str(scaler_npz.relative_to(PROJECT_ROOT)),
    }
    meta_path = out_dir / "kinematic_cnn_lstm_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as mf:
        json.dump(meta, mf, indent=2)
    print(f"[save] Metadata written to {meta_path}")

    # Also sync to ti_style visualizer location if present
    ti_onnx = onnx_dir / "ti_pose_model.onnx"
    try:
        shutil.copy2(onnx_path, ti_onnx)
        print(f"[save] Synced model to {ti_onnx}")
    except Exception:
        pass

    print("\n[success] Model training and export complete!")
    return True


def parse_args():
    parser = argparse.ArgumentParser(description="Train 1D CNN-LSTM Fall & Posture Detection Model")
    parser.add_argument(
        "--data-dir",
        default=str(PROJECT_ROOT / "dataset" / "my_posture_dataset"),
        help="Path to dataset root folder containing person 1..6 subfolders",
    )
    parser.add_argument("--epochs", type=int, default=35, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--window-size", type=int, default=8, help="Sliding window size in frames")
    parser.add_argument(
        "--split-by-session",
        action="store_true",
        help="Perform group-aware train/val split by recording session to prevent sliding-window data leakage",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        window_size=args.window_size,
        split_by_session=args.split_by_session,
    )
