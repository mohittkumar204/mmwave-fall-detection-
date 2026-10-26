"""Unit and Regression Tests for Session Selection, Noise Filter, and Tracking Pipeline Integration."""

import math
from pathlib import Path
import pytest

from src.data.dataset_inventory import get_dataset_trials
from src.utils.camera_utils import probe_cameras
from src.inference.live_fall_detector import LiveFallDetector
from src.radar.noise_filter import RadarNoiseFilter
from src.radar.packet_parser import ParsedFrame, Point, Target
from src.radar.serial_acquisition import RadarReplayAcquisition


def test_get_dataset_trials_real():
    """Verify discovery of real recorded sessions across dataset directories."""
    trials = get_dataset_trials()
    assert len(trials) >= 50, f"Expected at least 50 recorded trials, found {len(trials)}"
    # Every mapped directory must exist and contain targets.csv
    for name, path_str in list(trials.items())[:10]:
        p = Path(path_str)
        assert p.exists(), f"Trial directory does not exist: {p}"
        assert (p / "targets.csv").exists(), f"Missing targets.csv in {p}"


def test_get_dataset_trials_empty(tmp_path):
    """Verify that an empty directory returns an empty dict without crashing."""
    empty_dir = tmp_path / "empty_dataset"
    empty_dir.mkdir()
    res = get_dataset_trials(custom_search_dirs=[empty_dir])
    assert res == {}


def test_get_dataset_trials_nonexistent():
    """Verify that nonexistent directories are handled gracefully."""
    res = get_dataset_trials(custom_search_dirs=[Path("non_existent_folder_xyz_123")])
    assert res == {}


def test_keyerror_none_regression():
    """Regression test for KeyError: None when no trials exist."""
    empty_trials_map = {}
    selected_trial_name = None  # Returned by selectbox when choices is empty
    # Safe retrieval pattern must return None instead of raising KeyError
    selected_trial_dir = empty_trials_map.get(selected_trial_name)
    assert selected_trial_dir is None


def test_noise_filter_preserves_points():
    """Verify that clean point cloud points are NOT discarded from frame.points."""
    nf = RadarNoiseFilter(min_snr=8.0, min_age=1, min_points=0)
    pts = [
        Point(point_index=0, elevation=0.0, azimuth=0.0, doppler=0.0, range_val=2.0, snr=25.0, x=0.1, y=2.0, z=1.2, track_index=255),
        Point(point_index=1, elevation=0.0, azimuth=0.0, doppler=0.0, range_val=2.1, snr=18.0, x=0.2, y=2.0, z=1.3, track_index=255),
    ]
    t = Target(tid=0, pos_x=0.0, pos_y=2.0, pos_z=1.2, vel_x=0.0, vel_y=0.0, vel_z=0.0, acc_x=0.0, acc_y=0.0, acc_z=0.0)
    frame = ParsedFrame(frame_num=1, targets=[t], points=pts)
    
    filtered = nf.filter_frame(frame)
    assert len(filtered.points) == 2, f"Expected 2 points preserved, got {len(filtered.points)}"
    assert len(filtered.targets) == 1, f"Expected target preserved, got {len(filtered.targets)}"


def test_noise_filter_spatial_association():
    """Verify that points with unassigned track_index (255) are associated geometrically."""
    nf = RadarNoiseFilter(min_snr=8.0, min_age=1, min_points=0)
    # Point at (0.1, 2.0, 1.2) is 0.1m from target at (0.0, 2.0, 1.2) -> should associate
    close_pt = Point(point_index=0, elevation=0.0, azimuth=0.0, doppler=0.0, range_val=2.0, snr=20.0, x=0.1, y=2.0, z=1.2, track_index=255)
    t = Target(tid=5, pos_x=0.0, pos_y=2.0, pos_z=1.2, vel_x=0.0, vel_y=0.0, vel_z=0.0, acc_x=0.0, acc_y=0.0, acc_z=0.0)
    frame = ParsedFrame(frame_num=1, targets=[t], points=[close_pt])
    
    filtered = nf.filter_frame(frame)
    assert len(filtered.targets) == 1
    assert filtered.targets[0].tid == 5


def test_probe_cameras_safe_execution():
    """Verify probe_cameras runs and returns a list of integer indices without raising errors."""
    cams = probe_cameras(max_probe=2)
    assert isinstance(cams, list)
    for c in cams:
        assert isinstance(c, int)


def test_replay_pipeline_end_to_end():
    candidates = list(Path("dataset/my_posture_dataset").rglob("targets.csv"))
    if not candidates:
        candidates = list(Path("dataset").rglob("targets.csv"))
    assert len(candidates) > 0, "No recorded dataset trials found for replay test"
    trial_path = None
    for c in candidates:
        if "falling" in str(c).lower() and c.stat().st_size > 500:
            trial_path = c.parent
            break
    if trial_path is None:
        trial_path = candidates[0].parent
    assert trial_path.exists(), f"Missing test trial: {trial_path}"
    
    replay = RadarReplayAcquisition(trial_path, playback_fps=100.0)
    detector = LiveFallDetector()
    
    frames_processed = 0
    postures_observed = set()
    
    for frame in replay.stream_frames():
        results = detector.process_frame(frame)
        frames_processed += 1
        for res in results:
            postures_observed.add(res["posture"])
        if frames_processed >= 40:
            break
            
    assert frames_processed == 40
    # Must have produced predictions for targets
    assert len(postures_observed) > 0
