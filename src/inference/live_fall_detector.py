"""Live Fall Detection & Posture Classification Engine for IWR6843 mmWave Radar.

Supports both:
1. Hybrid 1D-CNN + LSTM on 9 Kinematic Channels [x, y, z, vx, vy, vz, ax, ay, az]
   (IEEE Access 2025 Architecture: 96.9% accuracy, sub-millisecond ONNX / PyTorch inference)
2. Fallback to Approach A Statistical / Approach B 176-feature models
"""

from __future__ import annotations

from collections import defaultdict, deque
import csv
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Deque, Dict, List, Optional

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np

# Safe onnxruntime availability check — onnxruntime 1.31 can crash with a native
# access violation on Python 3.13 when combined with PyQt5's offscreen/OpenGL
# backend. A subprocess probe prevents that from killing the main process.
_ONNX_AVAILABLE = False
if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
    # Qt offscreen mode conflicts with onnxruntime native code on Python 3.13
    print("[detector] Qt offscreen mode detected; skipping onnxruntime (PyTorch fallback).")
else:
    try:
        import subprocess as _sp
        _probe = _sp.run(
            [sys.executable, "-c",
             "import onnxruntime as ort; assert callable(ort.InferenceSession)"],
            capture_output=True, timeout=10,
        )
        _ONNX_AVAILABLE = (_probe.returncode == 0)
    except Exception:
        pass
if not _ONNX_AVAILABLE and os.environ.get("QT_QPA_PLATFORM") != "offscreen":
    print("[detector] onnxruntime not available on this Python (native crash guard); will use PyTorch fallback.")

from src.features.feature_pipeline import (
    FEATURE_NAMES_176,
    FEATURE_NAMES_22,
    WINDOW_SIZE,
    extract_statistical_features_from_window_176,
)
from src.radar.packet_parser import ParsedFrame, Point, Target
from src.radar.noise_filter import (
    RadarNoiseFilter,
    FALL_VEL_Z_THRESHOLD,
    FALL_ACC_Z_THRESHOLD,
    FALL_HEIGHT_DROP_M,
    FALL_FLOOR_Z_M,
)

KINEMATIC_FEATURES = ["x", "y", "z", "vx", "vy", "vz", "ax", "ay", "az"]
CLASS_NAMES_3 = ["UPRIGHT", "LYING", "FALLING"]
CLASS_NAMES = CLASS_NAMES_3


class LiveFallDetector:
    def __init__(
        self,
        metadata_path: str | Path = "artifacts/best_model_metadata.json",
        scaler_path: str | Path = "artifacts/scaler.joblib",
        model_path: str | Path = "models/best_fall_model.joblib",
        log_file: str | Path = "logs/live_fall_events.csv",
        hold_alert_frames: int = 35,
        esp32_port: Optional[str] = None,
        single_person_mode: bool = True,
        min_snr: float = 8.0,
        min_track_age: int = 1,
    ):
        # 0a. Hardware ESP32 Buzzer Setup
        self.esp32_port = esp32_port
        self.buzzer = None
        try:
            from src.alerts.esp32_buzzer import Esp32BuzzerAlert
            self.buzzer = Esp32BuzzerAlert(port=esp32_port)
        except Exception as e:
            print(f"[detector] Warning initializing ESP32 buzzer: {e}")

        # 0b. Telegram Fall Notification Dispatcher
        self.telegram = None
        try:
            from src.alerts.telegram_notifier import TelegramNotifier
            self.telegram = TelegramNotifier()
        except Exception as e:
            print(f"[detector] Warning initializing Telegram notifier: {e}")

        # 0c. Radar Noise / Ghost Track Filter
        self.noise_filter = RadarNoiseFilter(
            min_snr=min_snr,
            min_age=min_track_age,
            single_person_mode=single_person_mode,
        )
        print(f"[detector] Noise filter: single_person={single_person_mode}, min_snr={min_snr} dB, min_track_age={min_track_age} frames")

        # 1. Check for Hybrid CNN-LSTM Model (Primary 9-Kinematic Engine)
        self.cnn_lstm_session = None
        self.cnn_lstm_pt = None
        self.kinematic_scaler_mean = None
        self.kinematic_scaler_std = None
        self.use_cnn_lstm = False
        self.class_names = list(CLASS_NAMES_3)
        self.cnn_class_names = list(CLASS_NAMES_3)

        onnx_model_path = PROJECT_ROOT / "models" / "onnx" / "kinematic_cnn_lstm.onnx"
        pt_model_path = PROJECT_ROOT / "models" / "kinematic_cnn_lstm.pt"
        scaler_npz_path = PROJECT_ROOT / "models" / "kinematic_scaler.npz"
        meta_json_path = PROJECT_ROOT / "models" / "kinematic_cnn_lstm_metadata.json"

        self.window_size = 10
        if meta_json_path.exists():
            try:
                with open(meta_json_path, "r", encoding="utf-8") as f:
                    meta_cnn = json.load(f)
                if "classes" in meta_cnn:
                    self.cnn_class_names = meta_cnn["classes"]
                if "window_size" in meta_cnn:
                    self.window_size = int(meta_cnn["window_size"])
            except Exception as e:
                print(f"[detector] Warning loading kinematic metadata: {e}")

        if scaler_npz_path.exists():
            try:
                npz = np.load(scaler_npz_path)
                self.kinematic_scaler_mean = npz["mean"].astype(np.float32)
                self.kinematic_scaler_std = npz["std"].astype(np.float32)
                self.kinematic_scaler_std[self.kinematic_scaler_std == 0] = 1.0
            except Exception as e:
                print(f"[detector] Warning loading kinematic scaler: {e}")

        if _ONNX_AVAILABLE and onnx_model_path.exists() and self.kinematic_scaler_mean is not None:
            try:
                import onnxruntime as ort
                # Quick sanity check: ensure InferenceSession is callable
                # (catches partial-import or ABI mismatch issues)
                if not callable(getattr(ort, "InferenceSession", None)):
                    raise ImportError("onnxruntime InferenceSession not callable")
                self.cnn_lstm_session = ort.InferenceSession(
                    str(onnx_model_path),
                    providers=["CPUExecutionProvider"],
                )
                self.cnn_lstm_input_name = self.cnn_lstm_session.get_inputs()[0].name
                self.use_cnn_lstm = True
                print(f"[detector] Loaded ONNX Hybrid CNN-LSTM model from {onnx_model_path.name} (window: {self.window_size}, classes: {self.cnn_class_names})")
            except Exception as e:
                print(f"[detector] ONNX load failed ({e}), checking PyTorch fallback...")

        if not self.use_cnn_lstm and pt_model_path.exists() and self.kinematic_scaler_mean is not None:
            try:
                import torch
                from scripts.train_cnn_lstm import KinematicCNNLSTM
                pt_model = KinematicCNNLSTM(9, self.window_size, 64, 64, len(self.cnn_class_names), 0.25)
                pt_model.load_state_dict(torch.load(str(pt_model_path), map_location="cpu"))
                pt_model.eval()
                self.cnn_lstm_pt = pt_model
                self.use_cnn_lstm = True
                print(f"[detector] Loaded PyTorch Hybrid CNN-LSTM model from {pt_model_path.name} (window: {self.window_size})")
            except Exception as e:
                print(f"[detector] PyTorch load failed: {e}")

        # 2. Legacy / Secondary Models (Fallback)
        self.metadata = {}
        self.scaler = None
        self.model = None
        meta_p = Path(metadata_path)
        if meta_p.exists():
            with open(meta_p, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
            self.feature_names = self.metadata.get("feature_names", [])
            self.approach = self.metadata.get("best_approach", "Approach_A_Statistical")
            if Path(scaler_path).exists():
                self.scaler = joblib.load(scaler_path)
            if Path(model_path).exists():
                self.model = joblib.load(model_path)

        # 3. Rolling Windows per Target ID
        # 9 kinematic channels: shape (window_size, 9)
        self.kinematic_windows: dict[int, deque[list[float]]] = defaultdict(lambda: deque(maxlen=self.window_size))
        # 22 features window for legacy compatibility
        self.history_windows: dict[int, deque[list[float]]] = defaultdict(lambda: deque(maxlen=self.window_size))

        # 4. State & Alert Tracking
        self.alert_hold_counters: dict[int, int] = defaultdict(int)
        self.hold_alert_frames = hold_alert_frames
        self.target_height_baselines: dict[int, float] = {}
        self.recent_postures: dict[int, deque[str]] = defaultdict(lambda: deque(maxlen=5))

        # 5. Event Logging
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.log_file.exists():
            with open(self.log_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "frame_num", "tid", "event", "posture", "confidence", "latency_ms", "pos_z"])

    def process_frame(self, frame: ParsedFrame) -> list[dict[str, Any]]:
        """Process one incoming radar frame and return predictions for each target."""
        results = []

        # Apply noise/ghost filter FIRST before any processing
        frame = self.noise_filter.filter_frame(frame)

        if not frame.targets:
            return []

        # Map associated points to target IDs
        points_by_tid: dict[int, list[Point]] = defaultdict(list)
        for pt in frame.points:
            points_by_tid[pt.track_index].append(pt)

        for target in frame.targets:
            tid = target.tid
            assoc_pts = points_by_tid.get(tid, [])

            # Update baseline upright height
            tz = target.pos_z
            if tid not in self.target_height_baselines:
                self.target_height_baselines[tid] = tz
            else:
                # Slowly adapt baseline only when person is clearly upright and not falling
                if tz > self.target_height_baselines[tid] * 0.8:
                    self.target_height_baselines[tid] = 0.98 * self.target_height_baselines[tid] + 0.02 * tz

            # Build instantaneous 9 kinematic vector
            k9 = [
                target.pos_x,
                target.pos_y,
                target.pos_z,
                target.vel_x,
                target.vel_y,
                target.vel_z,
                target.acc_x,
                target.acc_y,
                target.acc_z,
            ]
            self.kinematic_windows[tid].append(k9)

            # Build 22 features for legacy buffer
            f22 = self._build_22_features(target, assoc_pts)
            self.history_windows[tid].append(f22)

            # Buffering check
            if len(self.kinematic_windows[tid]) < self.window_size:
                results.append({
                    "tid": tid,
                    "frame_num": frame.frame_num,
                    "status": "BUFFERING",
                    "posture": "BUFFERING",
                    "model_class": "BUFFERING",
                    "motion_label": f"Calibrating radar window ({len(self.kinematic_windows[tid])}/{self.window_size})...",
                    "confidence": 0.0,
                    "latency_ms": 0.0,
                    "target_x": target.pos_x,
                    "target_y": target.pos_y,
                    "target_z": target.pos_z,
                    "vel_x": target.vel_x,
                    "vel_y": target.vel_y,
                    "vel_z": target.vel_z,
                    "points": [{"x": p.x, "y": p.y, "z": p.z, "snr": p.snr, "doppler": p.doppler} for p in assoc_pts],
                })
                continue

            t0 = time.perf_counter()
            posture_str = "STANDING"
            confidence = 0.95
            is_fall_instant = False

            # Kinematic indicators
            horiz_speed = float(np.hypot(target.vel_x, target.vel_y))
            total_speed = float(np.sqrt(target.vel_x**2 + target.vel_y**2 + target.vel_z**2))
            vertical_drop_vel = target.vel_z  # negative = downward motion
            vertical_accel = target.acc_z

            # Run 1D CNN-LSTM model
            if self.use_cnn_lstm:
                win_np = np.array(list(self.kinematic_windows[tid]), dtype=np.float32)  # (10, 9)
                win_norm = (win_np - self.kinematic_scaler_mean) / self.kinematic_scaler_std
                win_input = np.expand_dims(win_norm, axis=0).astype(np.float32)  # (1, 10, 9)

                if self.cnn_lstm_session is not None:
                    logits = self.cnn_lstm_session.run(None, {self.cnn_lstm_input_name: win_input})[0][0]
                else:
                    import torch
                    with torch.no_grad():
                        logits = self.cnn_lstm_pt(torch.from_numpy(win_input)).numpy()[0]

                # Softmax probabilities
                exp_logits = np.exp(logits - np.max(logits))
                probs = exp_logits / np.sum(exp_logits)
                pred_class_id = int(np.argmax(probs))
                if pred_class_id < len(self.cnn_class_names):
                    raw_posture = self.cnn_class_names[pred_class_id]
                else:
                    raw_posture = "UPRIGHT"

                posture_str = raw_posture
                confidence = float(probs[pred_class_id])
                fall_class_id = self.cnn_class_names.index("FALLING") if "FALLING" in self.cnn_class_names else 2
                is_fall_instant = (posture_str == "FALLING") or (probs[fall_class_id] > 0.35 and vertical_drop_vel < -0.20)

                # Guard against false triggers only if target is higher than upright baseline
                base_z = self.target_height_baselines.get(tid, tz)
                if is_fall_instant and tz > base_z + 0.20:
                    is_fall_instant = False
                    posture_str = "UPRIGHT"
                    confidence = 0.88
            elif self.model is not None and self.scaler is not None:
                # Fallback to Approach A
                window_matrix = list(self.history_windows[tid])
                vector_176 = []
                for feat_idx in range(len(FEATURE_NAMES_22)):
                    for f_slot in range(min(len(window_matrix), WINDOW_SIZE)):
                        vector_176.append(window_matrix[f_slot][feat_idx])
                # Pad to 176 if needed
                while len(vector_176) < 176:
                    vector_176.append(0.0)
                vector_176_np = np.array(vector_176[:176], dtype=np.float32)
                stat_dict = extract_statistical_features_from_window_176(vector_176_np)
                feat_vec = np.array([[stat_dict[k] for k in self.feature_names]], dtype=np.float32)
                feat_scaled = self.scaler.transform(feat_vec)
                pred = int(self.model.predict(feat_scaled)[0])
                confidence = float(self.model.predict_proba(feat_scaled)[0, 1]) if hasattr(self.model, "predict_proba") else 0.95
                posture_str = "FALLING" if pred == 1 else "UPRIGHT"
                is_fall_instant = (pred == 1)

            # =========================================================================
            # PHYSICAL FALL DETECTION GATE & KINEMATIC MOTION ESTIMATION
            # Evaluates relative vertical drop from person's baseline upright height.
            # =========================================================================
            base_z = self.target_height_baselines.get(tid, tz)
            rel_drop = base_z - tz

            # Rule 1a: Rapid downward collapse trigger (downward velocity + relative drop)
            kinematic_fall = (
                (vertical_drop_vel < -0.32 and rel_drop > 0.18)
                or (vertical_drop_vel < -0.48 and rel_drop > 0.10)
            )
            # Rule 1b: Floor impact / low posture guard (requires downward motion)
            floor_impact = (
                (rel_drop > 0.38 and vertical_drop_vel < -0.22)
                or (tz < 0.40 and rel_drop > 0.22 and vertical_drop_vel < -0.20)
            )

            # Track maturity guard against fleeting single-frame radar ghosts
            track_age = 1
            if hasattr(self, "noise_filter") and hasattr(self.noise_filter, "_states"):
                st = self.noise_filter._states.get(tid)
                if st:
                    track_age = st.age_frames
            if track_age < 2:
                kinematic_fall = False
                floor_impact = False

            if kinematic_fall or floor_impact:
                is_fall_instant = True
                posture_str = "FALLING"
                confidence = max(confidence, 0.92)

            # Check Telegram Remote Command Override (/fall or /upright) and Height-Gate (< 0.68m)
            forced = self.get_forced_state()
            if forced == "FALL":
                posture_str = "FALLING"
                is_fall_instant = True
                confidence = 0.999
            elif forced == "UPRIGHT":
                posture_str = "UPRIGHT"
                is_fall_instant = False
                self.alert_hold_counters[tid] = 0
                confidence = 0.999
            elif tz > 0.68:
                # Target height is above 0.68m: physically upright, suppress fall trigger
                if is_fall_instant or posture_str == "FALLING":
                    posture_str = "UPRIGHT"
                    is_fall_instant = False
                    self.alert_hold_counters[tid] = 0

            # Kinematic motion estimation (separate from model class)
            if horiz_speed >= 0.35:
                rule_motion = f"Moving ({horiz_speed:.2f} m/s)"
            elif 0.65 <= tz < 1.20:
                rule_motion = f"Lowered Centroid ({tz:.2f}m)"
            elif tz < 0.65:
                rule_motion = f"Floor-Level ({tz:.2f}m)"
            else:
                rule_motion = f"Stationary (Height: {tz:.2f}m)"

            # Posture queue smoothing (avoids 1-frame jitter)
            self.recent_postures[tid].append(posture_str)

            # Fall Alert Hold Debounce
            if is_fall_instant or posture_str == "FALLING":
                self.alert_hold_counters[tid] = self.hold_alert_frames
            elif posture_str == "UPRIGHT" and tz >= 0.80 * base_z and len(self.recent_postures[tid]) >= 3 and all(p == "UPRIGHT" for p in list(self.recent_postures[tid])[-3:]):
                self.alert_hold_counters[tid] = 0

            active_alert = self.alert_hold_counters[tid] > 0
            if self.alert_hold_counters[tid] > 0:
                self.alert_hold_counters[tid] -= 1

            # Trigger Hardware ESP32 Buzzer Alarm
            if self.buzzer:
                if is_fall_instant or active_alert:
                    self.buzzer.trigger_fall_alarm()
                else:
                    self.buzzer.clear_alarm()

            latency_ms = (time.perf_counter() - t0) * 1000.0

            if is_fall_instant:
                self._log_event(frame.frame_num, tid, "FALL_DETECTED", posture_str, confidence, latency_ms, target.pos_z)

            # Motion description string & annotations
            if active_alert or is_fall_instant or posture_str == "FALLING":
                motion_desc = "⚠️ FALL DETECTED (High Downward Kinematics)"
            elif posture_str == "UPRIGHT":
                motion_desc = f"UPRIGHT ({rule_motion})"
            elif posture_str == "LYING":
                motion_desc = f"LYING ({rule_motion})"
            else:
                motion_desc = f"{posture_str} ({rule_motion})"

            # Trigger Asynchronous Telegram Fall Notification
            if self.telegram:
                if is_fall_instant:
                    self.telegram.send_fall_alert({
                        "tid": tid,
                        "frame_num": frame.frame_num,
                        "timestamp": time.time(),
                        "model_class": posture_str,
                        "confidence": confidence,
                        "pos_z": target.pos_z,
                        "vel_z": target.vel_z,
                        "motion_label": motion_desc,
                    })
                elif posture_str == "UPRIGHT":
                    self.telegram.update_target_posture(tid, posture_str)

            results.append({
                "tid": tid,
                "frame_num": frame.frame_num,
                "status": "FALL" if (active_alert or is_fall_instant) else posture_str,
                "posture": posture_str,
                "model_class": posture_str,
                "rule_motion": rule_motion,
                "is_fall_instant": is_fall_instant,
                "active_alert": active_alert,
                "confidence": confidence,
                "latency_ms": latency_ms,
                "motion_label": motion_desc,
                "target_x": target.pos_x,
                "target_y": target.pos_y,
                "target_z": target.pos_z,
                "vel_x": target.vel_x,
                "vel_y": target.vel_y,
                "vel_z": target.vel_z,
                "acc_x": target.acc_x,
                "acc_y": target.acc_y,
                "acc_z": target.acc_z,
                "speed_total": total_speed,
                "speed_horiz": horiz_speed,
                "points": [{"x": p.x, "y": p.y, "z": p.z, "snr": p.snr, "doppler": p.doppler} for p in assoc_pts],
            })

        return results

    def _build_22_features(self, target: Target, points: list[Point]) -> list[float]:
        points_sorted = sorted(points, key=lambda p: p.z, reverse=True)[:5]
        f22 = [
            target.pos_z,
            target.vel_x,
            target.vel_y,
            target.vel_z,
            target.acc_x,
            target.acc_y,
            target.acc_z,
        ]
        for pt in points_sorted:
            f22.extend([pt.y - target.pos_y, pt.z, pt.snr])
        while len(f22) < len(FEATURE_NAMES_22):
            f22.extend([0.0, 0.0, 0.0])
        return f22[: len(FEATURE_NAMES_22)]

    def _log_event(self, frame_num: int, tid: int, event: str, posture: str, conf: float, latency: float, z: float):
        try:
            with open(self.log_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([time.time(), frame_num, tid, event, posture, f"{conf:.4f}", f"{latency:.3f}", f"{z:.3f}"])
        except Exception:
            pass

    def acknowledge_alarm(self):
        """Immediately acknowledge fall alarm, silencing buzzer and resetting alert hold."""
        if self.buzzer:
            try:
                self.buzzer.acknowledge_alarm()
            except Exception:
                pass
        self.alert_hold_counters.clear()

    def trigger_fall(self, tid: int = 0, height: Optional[float] = None):
        """Externally trigger fall alert (e.g. from TI native height-derivative threshold or GUI)."""
        self.alert_hold_counters[tid] = self.hold_alert_frames
        if self.buzzer:
            try:
                self.buzzer.trigger_fall_alarm()
            except Exception:
                pass
        if self.telegram:
            try:
                self.telegram.send_fall_alert({
                    "tid": tid,
                    "timestamp": time.time(),
                    "model_class": "FALLING",
                    "confidence": 0.98,
                    "pos_z": height if height is not None else 0.40,
                    "motion_label": "Floor-Level Fall Detected (Height < 0.68m)",
                })
            except Exception:
                pass

    def test_alert_channels(self) -> dict[str, Any]:
        """Send test signals to active hardware buzzer and Telegram bot."""
        results: dict[str, Any] = {}
        if self.buzzer:
            try:
                results["esp32_buzzer"] = self.buzzer.test_beep()
            except Exception as e:
                results["esp32_buzzer"] = f"Error: {e}"
        if self.telegram:
            try:
                results["telegram"] = self.telegram.send_test_notification()
            except Exception as e:
                results["telegram"] = (False, f"Error: {e}")
        return results

    def get_forced_state(self) -> Optional[str]:
        """Return the current Telegram remote command override state, if any."""
        if self.telegram and hasattr(self.telegram, "get_forced_state"):
            return self.telegram.get_forced_state()
        return None

    def set_forced_state(self, state: Optional[str]):
        """Set the forced override state programmatically."""
        if self.telegram and hasattr(self.telegram, "set_forced_state"):
            self.telegram.set_forced_state(state)

    def close(self):
        """Cleanly release serial ports, network sockets, and worker threads."""
        if self.buzzer:
            try:
                self.buzzer.close()
            except Exception:
                pass
        if self.telegram:
            try:
                self.telegram.close()
            except Exception:
                pass

