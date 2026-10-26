"""Serial Data Acquisition and Sensor Replay for IWR6843 mmWave Radar."""

from __future__ import annotations

from pathlib import Path
import struct
import time
from typing import Generator, Iterable, Optional
import pandas as pd

from src.radar.packet_parser import MAGIC_WORD, ParsedFrame, Point, Target, TargetHeight, parse_radar_frame


class RadarSerialAcquisition:
    """Manages physical radar UART connection with auto-sync and recovery."""

    def __init__(
        self,
        cli_port: str = "COM5",
        data_port: str = "COM6",
        cli_baud: int = 115200,
        data_baud: int = 921600,
        timeout: float = 0.6,
    ):
        self.cli_port = cli_port
        self.data_port = data_port
        self.cli_baud = cli_baud
        self.data_baud = data_baud
        self.timeout = timeout
        self.serial_conn = None

    def send_configuration(self, cfg_path: str | Path) -> list[str]:
        """Send radar chirp config lines to the CLI port."""
        import serial

        cfg_path = Path(cfg_path)
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config file not found: {cfg_path}")

        responses = []
        with serial.Serial(self.cli_port, self.cli_baud, timeout=1.0) as cli_ser:
            with open(cfg_path, "r", encoding="utf-8") as f:
                for line in f:
                    clean_line = line.strip()
                    if not clean_line or clean_line.startswith("%"):
                        continue
                    cli_ser.write((clean_line + "\n").encode("ascii"))
                    time.sleep(0.02)
                    resp = cli_ser.read_all().decode("ascii", errors="ignore")
                    responses.append(f"{clean_line} -> {resp.strip()}")
        return responses

    def connect_data_port(self):
        import serial
        self.serial_conn = serial.Serial(self.data_port, self.data_baud, timeout=self.timeout)

    def stop_sensor(self):
        """Send sensorStop command to CLI port to halt radar transmission."""
        import serial
        try:
            with serial.Serial(self.cli_port, self.cli_baud, timeout=0.5) as cli_ser:
                cli_ser.write(b"sensorStop\n")
                time.sleep(0.05)
        except Exception:
            pass

    def close(self):
        self.stop_sensor()
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()

    def read_frame(self) -> ParsedFrame:
        """Read a single frame from the physical data port."""
        if not self.serial_conn or not self.serial_conn.is_open:
            raise ConnectionError("Radar serial port is not open")

        # Sync on 8-byte magic word
        frame_bytes = bytearray()
        match_idx = 0
        while True:
            b = self.serial_conn.read(1)
            if not b:
                raise TimeoutError("UART timeout waiting for magic word")
            val = b[0]
            if val == MAGIC_WORD[match_idx]:
                frame_bytes.append(val)
                match_idx += 1
                if match_idx == len(MAGIC_WORD):
                    break
            else:
                if val == MAGIC_WORD[0]:
                    frame_bytes = bytearray([val])
                    match_idx = 1
                else:
                    frame_bytes = bytearray()
                    match_idx = 0

        # Read version and total packet length
        header_rem = self.serial_conn.read(8)
        if len(header_rem) < 8:
            raise TimeoutError("Incomplete header bytes received")
        frame_bytes += header_rem
        total_len = struct.unpack("<I", header_rem[4:8])[0]

        if total_len < 16 or total_len > 65536:
            raise ValueError(f"Invalid packet length: {total_len}")

        rem_len = total_len - 16
        payload = self.serial_conn.read(rem_len)
        if len(payload) < rem_len:
            raise TimeoutError(f"Expected {rem_len} bytes, received {len(payload)}")
        frame_bytes += payload

        return parse_radar_frame(bytes(frame_bytes))


class RadarReplayAcquisition:
    """Replays pre-recorded radar trial recordings frame-by-frame for offline validation and dry-runs."""

    def __init__(self, trial_dir: str | Path, playback_fps: float = 18.18):
        self.trial_dir = Path(trial_dir)
        self.playback_fps = playback_fps
        self.frame_interval = 1.0 / playback_fps if playback_fps > 0 else 0.055

        self.targets_csv = self.trial_dir / "targets.csv"
        self.points_csv = self.trial_dir / "raw_points.csv"
        self.meta_json = self.trial_dir / "metadata.json"

        if not self.targets_csv.exists():
            raise FileNotFoundError(f"Missing targets.csv in {self.trial_dir}")

        self.df_targets = pd.read_csv(self.targets_csv)
        self.df_points = pd.read_csv(self.points_csv) if self.points_csv.exists() else pd.DataFrame()

    def stream_frames(self, real_time_delay: bool = False) -> Generator[ParsedFrame, None, None]:
        unique_frames = sorted(self.df_targets["frame"].unique())

        for f_num in unique_frames:
            tar_sub = self.df_targets[self.df_targets["frame"] == f_num]
            pts_sub = self.df_points[self.df_points["frame"] == f_num] if len(self.df_points) > 0 else pd.DataFrame()

            parsed = ParsedFrame(frame_num=int(f_num))

            for _, row in tar_sub.iterrows():
                parsed.targets.append(
                    Target(
                        tid=int(row["tid"]),
                        pos_x=float(row.get("x", 0.0)),
                        pos_y=float(row.get("y", 0.0)),
                        pos_z=float(row.get("z", 0.0)),
                        vel_x=float(row.get("vx", 0.0)),
                        vel_y=float(row.get("vy", 0.0)),
                        vel_z=float(row.get("vz", 0.0)),
                        acc_x=float(row.get("ax", 0.0)),
                        acc_y=float(row.get("ay", 0.0)),
                        acc_z=float(row.get("az", 0.0)),
                    )
                )
                parsed.heights.append(
                    TargetHeight(
                        tid=int(row["tid"]),
                        max_z=float(row.get("maxZ", row.get("z", 0.0))),
                        min_z=float(row.get("minZ", row.get("z", 0.0))),
                    )
                )

            for _, prow in pts_sub.iterrows():
                parsed.points.append(
                    Point(
                        point_index=int(prow.get("point_index", 0)),
                        elevation=0.0,
                        azimuth=0.0,
                        doppler=float(prow.get("doppler", 0.0)),
                        range_val=float(prow.get("range", 0.0)),
                        snr=float(prow.get("snr", 0.0)),
                        x=float(prow.get("x", 0.0)),
                        y=float(prow.get("y", 0.0)),
                        z=float(prow.get("z", 0.0)),
                        track_index=int(prow.get("track_index", 0)),
                    )
                )

            if real_time_delay:
                time.sleep(self.frame_interval)

            yield parsed
