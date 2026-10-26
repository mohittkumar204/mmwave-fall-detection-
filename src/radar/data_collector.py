"""Data Collection and Session Logger for Physical IWR6843 Radar Recordings.

Saves captured mmWave radar sessions with exact compatibility with the training dataset:
- metadata.json: sensor config, subject, activity label, timestamps
- targets.csv: tracked targets (tid, x, y, z, vx, vy, vz, ax, ay, az, maxZ, minZ)
- raw_points.csv: point cloud points (frame, time, tid, point_index, x, y, z, doppler, snr)
- events.csv: session start, stop, and quality markers
- features_22.csv: instantaneous 22 kinematic & point attributes
- features_176.csv: 8-frame rolling window representation
"""

from __future__ import annotations

import csv
from datetime import datetime
import json
from pathlib import Path
import time
from typing import Any, List, Optional

from src.features.feature_pipeline import FEATURE_NAMES_176, FEATURE_NAMES_22, WINDOW_SIZE
from src.radar.packet_parser import ParsedFrame, Point, Target


class RadarDataCollector:
    def __init__(
        self,
        output_base_dir: str | Path = "datasets/new_recordings",
        subject_id: str = "subject_new",
        activity_label: str = "standing",
        trial_number: int | str = 1,
        notes: str = "",
        sensor_model: str = "IWR6843ISK-ODS",
        cfg_path: str = "configs/ODS_6m_posture_tuned.cfg",
    ):
        self.output_base_dir = Path(output_base_dir)
        self.subject_id = subject_id
        self.activity_label = activity_label
        self.trial_number = trial_number
        self.notes = notes
        self.sensor_model = sensor_model
        self.cfg_path = cfg_path

        # Format trial prefix cleanly: e.g. 1 -> T01, "2" -> T02, "T05" -> T05
        t_str = str(trial_number).strip()
        if t_str.upper().startswith("T"):
            t_prefix = t_str.upper()
        else:
            try:
                t_prefix = f"T{int(t_str):02d}"
            except ValueError:
                t_prefix = f"T_{t_str}"

        self.session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.trial_id = f"{t_prefix}_{self.session_timestamp}"
        self.session_dir = self.output_base_dir / self.subject_id / self.activity_label / self.trial_id

        self.is_recording = False
        self.frame_count = 0
        self.history_by_tid: dict[int, list[list[float]]] = {}

        # File handles & writers
        self._f_targets = None
        self._f_points = None
        self._f_events = None
        self._f_22 = None
        self._f_176 = None

    def start(self):
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.is_recording = True
        self.start_time = time.time()
        self.frame_count = 0

        # 1. Metadata
        metadata = {
            "sensor": self.sensor_model,
            "cfg_path": str(self.cfg_path),
            "label": self.activity_label,
            "subject_id": self.subject_id,
            "trial_id": self.trial_id,
            "date_time": datetime.now().isoformat(),
            "notes": self.notes,
            "nominal_frame_periodicity_ms": 55.0,
            "feature_order_22": FEATURE_NAMES_22,
            "feature_order_176": "channel-major: feature_f0..feature_f7 for each of 22 features",
        }
        with open(self.session_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        # 2. Open CSV writers
        self._f_targets = open(self.session_dir / "targets.csv", "w", newline="", encoding="utf-8")
        self._w_targets = csv.writer(self._f_targets)
        self._w_targets.writerow(["frame", "time", "label", "tid", "x", "y", "z", "vx", "vy", "vz", "ax", "ay", "az", "maxZ", "minZ", "confidence"])

        self._f_points = open(self.session_dir / "raw_points.csv", "w", newline="", encoding="utf-8")
        self._w_points = csv.writer(self._f_points)
        self._w_points.writerow(["frame", "time", "label", "tid", "point_index", "x", "y", "z", "doppler", "snr", "track_index"])

        self._f_events = open(self.session_dir / "events.csv", "w", newline="", encoding="utf-8")
        self._w_events = csv.writer(self._f_events)
        self._w_events.writerow(["frame", "time", "label", "tid", "event", "notes"])
        self._w_events.writerow([0, self.start_time, self.activity_label, "", "capture_start", self.notes])

        self._f_22 = open(self.session_dir / "features_22.csv", "w", newline="", encoding="utf-8")
        self._w_22 = csv.writer(self._f_22)
        self._w_22.writerow(["frame", "time", "label", "tid"] + FEATURE_NAMES_22 + ["num_points", "low_quality", "reason"])

        self._f_176 = open(self.session_dir / "features_176.csv", "w", newline="", encoding="utf-8")
        self._w_176 = csv.writer(self._f_176)
        self._w_176.writerow(["frame", "time", "label", "tid", "ready"] + FEATURE_NAMES_176 + ["window_age", "num_points_recent", "low_quality_count"])

        print(f"Data collection started: {self.session_dir}")

    def record_frame(self, frame: ParsedFrame, timestamp: float | None = None):
        if not self.is_recording:
            return

        ts = timestamp if timestamp is not None else time.time()
        self.frame_count += 1

        points_by_tid: dict[int, list[Point]] = {}
        for pt in frame.points:
            points_by_tid.setdefault(pt.track_index, []).append(pt)
            self._w_points.writerow([frame.frame_num, ts, self.activity_label, pt.track_index, pt.point_index, pt.x, pt.y, pt.z, pt.doppler, pt.snr, pt.track_index])

        for target in frame.targets:
            tid = target.tid
            self._w_targets.writerow([
                frame.frame_num, ts, self.activity_label, tid,
                target.pos_x, target.pos_y, target.pos_z,
                target.vel_x, target.vel_y, target.vel_z,
                target.acc_x, target.acc_y, target.acc_z,
                target.pos_z, target.pos_z, 1.0
            ])

            # Build 22 features
            assoc_pts = sorted(points_by_tid.get(tid, []), key=lambda p: p.z, reverse=True)[:5]
            f22 = [target.pos_z, target.vel_x, target.vel_y, target.vel_z, target.acc_x, target.acc_y, target.acc_z]
            for p in assoc_pts:
                f22.extend([p.y - target.pos_y, p.z, p.snr])
            while len(f22) < len(FEATURE_NAMES_22):
                f22.extend([0.0, 0.0, 0.0])
            f22 = f22[: len(FEATURE_NAMES_22)]

            num_pts = len(points_by_tid.get(tid, []))
            low_q = 1 if num_pts < 5 else 0
            reason = f"only_{num_pts}_points" if low_q else ""
            self._w_22.writerow([frame.frame_num, ts, self.activity_label, tid] + f22 + [num_pts, low_q, reason])

            # Rolling 8-frame window for 176 features
            hist = self.history_by_tid.setdefault(tid, [])
            hist.append(f22)
            if len(hist) > WINDOW_SIZE:
                hist.pop(0)

            ready = 1 if len(hist) == WINDOW_SIZE else 0
            # Pad front if len < 8
            padded_hist = [[0.0] * 22] * (WINDOW_SIZE - len(hist)) + hist
            # Channel major
            f176 = []
            for f_idx in range(len(FEATURE_NAMES_22)):
                for slot in range(WINDOW_SIZE):
                    f176.append(padded_hist[slot][f_idx])

            self._w_176.writerow([frame.frame_num, ts, self.activity_label, tid, ready] + f176 + [len(hist), num_pts, low_q])

    def stop(self):
        if not self.is_recording:
            return
        stop_time = time.time()
        self.is_recording = False

        if self._w_events:
            self._w_events.writerow([self.frame_count, stop_time, self.activity_label, "", "capture_stop", f"Recorded {self.frame_count} frames"])

        for fh in [self._f_targets, self._f_points, self._f_events, self._f_22, self._f_176]:
            if fh and not fh.closed:
                fh.flush()
                fh.close()

        print(f"Data collection completed: {self.session_dir} ({self.frame_count} frames)")
        return self.session_dir
