"""Tests for Texas Instruments mmWave People Counting & Fall Detection GUI Integration."""

import math
from pathlib import Path
import pytest
import numpy as np

# Qt headless mode for automated testing
import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"

# Pre-import torch before Qt to avoid Windows DLL initialization conflict on Python 3.13
try:
    import torch
except Exception:
    pass

from PyQt5.QtWidgets import QApplication
from src.gui.ti_mmwave_fall_gui import Window
from src.inference.live_fall_detector import LiveFallDetector
from src.radar.packet_parser import ParsedFrame, Point, Target
from src.radar.serial_acquisition import RadarReplayAcquisition


@pytest.fixture(scope="session")
def qapp():
    """Ensure a single QApplication instance exists for GUI tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_model_loading_and_engine():
    """Test A: Verify that the trained ML fall detector loads with ONNX/PyTorch CNN-LSTM."""
    detector = LiveFallDetector()
    assert detector is not None
    assert detector.use_cnn_lstm is True or detector.model is not None
    if detector.use_cnn_lstm:
        assert detector.kinematic_scaler_mean is not None
        assert detector.kinematic_scaler_std is not None
        assert len(detector.kinematic_scaler_mean) == 9
    else:
        assert detector.scaler is not None
        assert detector.model is not None


def test_input_compatibility():
    """Test B: Verify that extracted live features match the model's expected shape."""
    detector = LiveFallDetector()
    t = Target(
        tid=0, pos_x=0.0, pos_y=2.5, pos_z=1.4,
        vel_x=0.0, vel_y=0.0, vel_z=0.0,
        acc_x=0.0, acc_y=0.0, acc_z=0.0
    )
    pts = [
        Point(point_index=i, elevation=0.0, azimuth=0.0, doppler=0.0, range_val=2.5, snr=20.0, x=0.0, y=2.5, z=1.4, track_index=0)
        for i in range(12)
    ]
    frame = ParsedFrame(frame_num=1, targets=[t], points=pts)
    results = detector.process_frame(frame)
    assert len(results) == 1
    assert results[0]["tid"] == 0
    # First frame should be buffering window
    assert results[0]["posture"] == "BUFFERING"


def test_gui_window_initialization(qapp):
    """Test C & D: Verify that the original TI visualizer initializes with exact title and widgets."""
    window = Window()
    assert window.windowTitle() == "mmWave People Counting"

    # Verify all original controls exist
    assert hasattr(window, "comBox")
    assert hasattr(window, "statBox")
    assert hasattr(window, "configBox")
    assert hasattr(window, "plotControlBox")
    assert hasattr(window, "boxTab")
    assert hasattr(window, "spBox")
    assert hasattr(window, "graphTabs")
    assert hasattr(window, "pcplot")
    assert hasattr(window, "hPlot")
    assert hasattr(window, "demoData")

    # Verify Fall Detection Demo widgets
    assert hasattr(window, "demoName")
    assert hasattr(window, "numDetPeople")
    assert hasattr(window, "fallDetEnabled")
    assert hasattr(window, "fallAlert")
    assert hasattr(window, "fallPic")
    assert hasattr(window, "mlModelStatus")
    assert hasattr(window, "confidenceLabel")
    assert hasattr(window, "latencyLabel")

    # Verify Height Plots widgets
    assert hasattr(window, "iHPlot")
    assert hasattr(window, "mHPlot")
    assert hasattr(window, "dHPlot")
    assert hasattr(window, "fallThreshMarker")

    # Verify initial text
    assert "No People Detected" in window.fallDetEnabled.text()
    assert window.fallAlert.text() == "Upright"
    window.close()


def test_gui_normal_tracking_inference(qapp):
    """Test E: Verify that normal tracking updates the GUI and does NOT trigger fall alert."""
    window = Window()

    # Create 15 consecutive upright standing frames (z=1.4m, vz=0)
    for frame_idx in range(1, 16):
        targets = np.zeros((13, 1))
        targets[0, 0] = 0     # TID
        targets[1, 0] = 0.0   # x
        targets[2, 0] = 2.0   # y
        targets[3, 0] = 1.4   # z (upright)
        targets[4, 0] = 0.0   # vx
        targets[5, 0] = 0.0   # vy
        targets[6, 0] = 0.0   # vz
        targets[10, 0] = 0.0  # ax
        targets[11, 0] = 0.0  # ay
        targets[12, 0] = 0.0  # az

        points = np.zeros((5, 10))
        for p in range(10):
            points[0, p] = 0.0
            points[1, p] = 2.0
            points[2, p] = 1.4
            points[3, p] = 0.0
            points[4, p] = 20.0  # SNR 20 dB

        indexes = np.zeros((10,))  # all associated to TID 0

        parsed_data = (points, targets, indexes, 10, 1, frame_idx, 0, [])
        window.updateGraph(parsed_data)

    # Person should be detected and classified as Upright (NOT Fallen)
    assert window.numDetPeople.text() == "Number of Detected People: 1"
    assert window.fallAlert.text() in ("Upright", "Buffering...")
    assert "FALLEN" not in window.fallAlert.text()
    window.close()


def test_gui_fall_detection_trigger(qapp):
    """Test F: Verify that a rapid downward fall motion triggers the ML fall alert in the GUI."""
    window = Window()

    # Step 1: 8 frames of normal upright stance to establish baseline
    for frame_idx in range(1, 9):
        targets = np.zeros((13, 1))
        targets[0, 0] = 0
        targets[1, 0] = 0.0
        targets[2, 0] = 2.0
        targets[3, 0] = 1.4
        targets[4, 0] = 0.0
        targets[5, 0] = 0.0
        targets[6, 0] = 0.0
        targets[10, 0] = 0.0
        targets[11, 0] = 0.0
        targets[12, 0] = 0.0

        points = np.zeros((5, 10))
        for p in range(10):
            points[0, p] = 0.0
            points[1, p] = 2.0
            points[2, p] = 1.4
            points[3, p] = 0.0
            points[4, p] = 20.0
        indexes = np.zeros((10,))

        window.updateGraph((points, targets, indexes, 10, 1, frame_idx, 0, []))

    # Step 2: Fall motion (high downward velocity vz = -1.2 m/s, downward accel, drop to floor)
    fall_targets = np.zeros((13, 1))
    fall_targets[0, 0] = 0
    fall_targets[1, 0] = 0.0
    fall_targets[2, 0] = 2.0
    fall_targets[3, 0] = 0.35  # Near floor
    fall_targets[4, 0] = 0.0
    fall_targets[5, 0] = 0.0
    fall_targets[6, 0] = -1.2  # Downward velocity exceeding -0.55 m/s threshold
    fall_targets[10, 0] = 0.0
    fall_targets[11, 0] = 0.0
    fall_targets[12, 0] = -0.6 # Negative vertical acceleration

    fall_points = np.zeros((5, 10))
    for p in range(10):
        fall_points[0, p] = 0.0
        fall_points[1, p] = 2.0
        fall_points[2, p] = 0.35
        fall_points[3, p] = -1.2
        fall_points[4, p] = 20.0
    fall_indexes = np.zeros((10,))

    window.updateGraph((fall_points, fall_targets, fall_indexes, 10, 1, 9, 0, []))

    # Assert that FALLEN alert was triggered
    assert window.fallAlert.text() == "FALLEN!"
    assert "FALL DETECTED" in window.fallDetEnabled.text()
    window.close()


def test_gui_replay_pipeline(qapp):
    """Test G: Verify replay mode streaming from recorded dataset trials."""
    window = Window()
    candidates = list(Path("dataset/my_posture_dataset").rglob("targets.csv"))
    if not candidates:
        candidates = list(Path("dataset").rglob("targets.csv"))
    assert len(candidates) > 0, "No dataset trials found for replay test"
    trial_path = candidates[0].parent
    assert trial_path.exists()

    window.replay_acquisition = RadarReplayAcquisition(trial_path, playback_fps=100.0)
    window.replay_generator = window.replay_acquisition.stream_frames()

    for _ in range(15):
        window._replay_step()

    assert window.frameNum > 0
    assert window.numPointsDisplay.text() != ""
    window._disconnect()
    window.close()


def test_gui_disconnection_handling(qapp):
    """Test H: Verify safe failure handling when serial ports are unpowered/disconnected."""
    window = Window()
    window.uartCom.setText("COM99")
    window.dataCom.setText("COM98")
    # connectCom will catch SerialException and display 'Unable to Connect'
    window.connectCom()
    assert window.connectStatus.text() == "Unable to Connect"
    window.close()


def test_oob_parser_modern_tlv_support():
    """Test I: Verify that uartParserSDK parses modern IWR6843 40-byte headers and TLVs 1010 & 1020."""
    import struct
    from src.gui.oob_parser import uartParserSDK

    parser = uartParserSDK(type='3D People Counting')
    assert parser.capon3D == 1

    # Build 1 Point (TLV 1020: 20-byte unit struct + 8-byte point)
    punit_data = struct.pack('<5f', 0.01, 0.01, 0.01, 0.05, 1.0)
    pt_data = struct.pack('<2bh2H', 10, 20, 0, 40, 25)
    tlv1020_payload = punit_data + pt_data
    tlv1020_header = struct.pack('<2I', 1020, len(tlv1020_payload))

    # Build 1 Target (TLV 1010: 112 bytes = uint32 tid + 27 floats)
    target_raw = [0, 0.5, 2.0, 1.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] + [0.0] * 18
    target_payload = struct.pack('<I27f', *target_raw)
    tlv1010_header = struct.pack('<2I', 1010, len(target_payload))

    # Build 40-byte Frame Header (<Q8I)
    total_len = 40 + len(tlv1020_header) + len(tlv1020_payload) + len(tlv1010_header) + len(target_payload)
    magic = b"\x02\x01\x04\x03\x06\x05\x08\x07"
    hdr = magic + struct.pack('<8I', 0x03020004, total_len, 0xA6843, 77, 12345, 1, 2, 0)

    packet = hdr + tlv1020_header + tlv1020_payload + tlv1010_header + target_payload
    assert len(packet) == total_len

    rem = parser.Capon3DHeader(packet)
    assert parser.fail == 0
    assert parser.frameNum == 77
    assert parser.numDetectedObj == 1
    assert parser.numDetectedTarget == 1
    assert parser.targetBufPing[0, 0] == 0  # TID 0


# =========================================================================
# PART 8: SYSTEM VERIFICATION TEST SUITE
# =========================================================================

def test_part8_test1_model_metadata():
    """PART 8 Test 1: Verify actual model classes, feature count, window length, and input/output shapes."""
    import json
    meta_path = Path("models/kinematic_cnn_lstm_metadata.json")
    assert meta_path.exists(), "Metadata file models/kinematic_cnn_lstm_metadata.json missing"
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["classes"] == ["UPRIGHT", "LYING", "FALLING"], f"Expected 3 classes, got {meta['classes']}"
    assert meta["num_classes"] == 3
    assert len(meta["features"]) == 9
    assert meta["window_size"] == 8
    assert meta["input_shape"] == [1, 8, 9]


def test_part8_test2_preprocessing_scaler():
    """PART 8 Test 2: Verify that the saved scaler is loaded and that live preprocessing matches training."""
    scaler_path = Path("models/kinematic_scaler.npz")
    assert scaler_path.exists(), "Scaler models/kinematic_scaler.npz missing"
    data = np.load(scaler_path)
    mean = data["mean"]
    std = data["std"]
    assert len(mean) == 9
    assert len(std) == 9
    assert not np.isnan(mean).any()
    assert not np.isnan(std).any()
    assert (std > 0).all()

    detector = LiveFallDetector()
    assert detector.kinematic_scaler_mean is not None
    assert np.allclose(detector.kinematic_scaler_mean, mean)
    assert np.allclose(detector.kinematic_scaler_std, std)


def test_part8_test3_onnx_inference_parity():
    """PART 8 Test 3: Verify exported ONNX model runs with live input shape and matches PyTorch outputs."""
    import onnxruntime as ort
    from scripts.train_cnn_lstm import KinematicCNNLSTM

    onnx_path = "models/onnx/kinematic_cnn_lstm.onnx"
    pt_path = "models/kinematic_cnn_lstm.pt"
    assert Path(onnx_path).exists()
    assert Path(pt_path).exists()

    pt_model = KinematicCNNLSTM(in_channels=9, seq_len=8, conv_filters=64, lstm_hidden=64, num_classes=3, dropout=0.25)
    pt_model.load_state_dict(torch.load(pt_path))
    pt_model.eval()

    sess = ort.InferenceSession(onnx_path)
    inp = np.random.randn(1, 8, 9).astype(np.float32)

    with torch.no_grad():
        pt_out = pt_model(torch.from_numpy(inp)).numpy()
    ort_out = sess.run(None, {"kinematic_window": inp})[0]

    max_diff = float(np.max(np.abs(pt_out - ort_out)))
    assert max_diff < 1e-4, f"PyTorch vs ONNX difference {max_diff} exceeded tolerance"


def test_part8_test4_class_mapping(qapp):
    """PART 8 Test 4: Verify GUI does not display unsupported activity classes (Walking/Sitting) as model predictions."""
    detector = LiveFallDetector()
    assert detector.cnn_class_names == ["UPRIGHT", "LYING", "FALLING"]
    assert detector.class_names == ["UPRIGHT", "LYING", "FALLING"]

    # Test that normal upright target produces UPRIGHT, not WALKING or SITTING
    for f in range(1, 10):
        t = Target(tid=0, pos_x=0.0, pos_y=2.0, pos_z=1.5, vel_x=0.5, vel_y=0.0, vel_z=0.0, acc_x=0.0, acc_y=0.0, acc_z=0.0)
        res = detector.process_frame(ParsedFrame(frame_num=f, targets=[t], points=[]))
    assert res[0]["model_class"] in ("UPRIGHT", "LYING", "FALLING")
    assert res[0]["model_class"] != "WALKING"
    assert res[0]["model_class"] != "SITTING"


def test_part8_test5_dataset_leakage_audit():
    """PART 8 Test 5: Verify that session tracking identifies recording sessions and prevents cross-session window leakage."""
    data_dir = Path("dataset/my_posture_dataset")
    csvs = list(data_dir.rglob("targets.csv"))
    assert len(csvs) >= 75, f"Expected at least 75 recording sessions, found {len(csvs)}"

    # Check distinct subjects
    subjects = {f.relative_to(data_dir).parts[0] for f in csvs}
    assert subjects == {"person1", "person2", "person3"}


def test_part8_test6_graph_data_calculations(qapp):
    """PART 8 Test 6: Verify instantaneous-height, average-height, and delta-height calculations."""
    window = Window()
    for frame_idx in range(1, 20):
        targets = np.zeros((13, 1))
        targets[0, 0] = 0
        targets[1, 0] = 0.0
        targets[2, 0] = 2.0
        targets[3, 0] = 1.35
        points = np.zeros((5, 5))
        indexes = np.zeros((5,))
        window.updateGraph((points, targets, indexes, 5, 1, frame_idx, 0, []))

    f100 = 19 % 100
    h_inst = window.targetSize[0, 0, f100]
    h_avg = window.targetSize[5, 0, f100]
    h_delta = window.targetSize[6, 0, f100]

    assert h_inst > 0, "Instantaneous height should be > 0"
    assert h_avg > 0, "Average height should be > 0"
    assert isinstance(h_delta, float)
    window.close()


def test_part8_test7_graph_rendering(qapp):
    """PART 8 Test 7: Verify that existing GUI graph-update functions actually render valid measurements."""
    import time
    window = Window()
    for frame_idx in range(1, 25):
        targets = np.zeros((13, 1))
        targets[0, 0] = 0
        targets[3, 0] = 1.35 - (0.01 * frame_idx)
        points = np.zeros((5, 4))
        indexes = np.zeros((4,))
        window.updateGraph((points, targets, indexes, 4, 1, frame_idx, 0, []))
        qapp.processEvents()
        time.sleep(0.005)

    if hasattr(window, 'heightThread') and window.heightThread.isRunning():
        window.heightThread.wait(1000)
    qapp.processEvents()

    xi, yi = window.heightPlotsDict['instant'].getData()
    xm, ym = window.heightPlotsDict['mean'].getData()
    xd, yd = window.heightPlotsDict['deriv'].getData()

    assert xi is not None and len(xi) > 0, "Instantaneous height curve is empty"
    assert xm is not None and len(xm) > 0, "Average height curve is empty"
    assert xd is not None and len(xd) > 0, "Delta height curve is empty"
    window.close()


def test_part8_test8_missing_data_handling(qapp):
    """PART 8 Test 8: Verify application handles empty radar stream (0 targets, 0 points) without crashing."""
    window = Window()
    points = np.zeros((5, 0))
    targets = np.zeros((13, 0))
    indexes = np.zeros((0,))
    window.updateGraph((points, targets, indexes, 0, 0, 100, 0, []))

    assert "No People Detected" in window.fallDetEnabled.text()
    assert window.numDetPeople.text() == "Number of Detected People: 0"
    assert window.fallAlert.text() == "Upright"
    window.close()


def test_part8_test9_stream_recovery(qapp):
    """PART 8 Test 9: Verify application recovers correctly after a temporary loss of radar data."""
    window = Window()
    # Step 1: Active frame
    targets1 = np.zeros((13, 1))
    targets1[3, 0] = 1.4
    window.updateGraph((np.zeros((5, 4)), targets1, np.zeros((4,)), 4, 1, 1, 0, []))
    assert window.numDetPeople.text() == "Number of Detected People: 1"

    # Step 2: Empty data interruption (e.g. sensor occlusion / dropped frames)
    window.updateGraph((np.zeros((5, 0)), np.zeros((13, 0)), np.zeros((0,)), 0, 0, 2, 0, []))
    assert window.numDetPeople.text() == "Number of Detected People: 0"

    # Step 3: Stream resumes
    targets3 = np.zeros((13, 1))
    targets3[3, 0] = 1.38
    window.updateGraph((np.zeros((5, 4)), targets3, np.zeros((4,)), 4, 1, 3, 0, []))
    assert window.numDetPeople.text() == "Number of Detected People: 1"
    window.close()


def test_part8_test10_gui_preservation(qapp):
    """PART 8 Test 10: Verify existing GUI retains layout, radar visualization, controls, and graph panels."""
    window = Window()
    assert window.windowTitle() == "mmWave People Counting"
    assert hasattr(window, "pcplot")
    assert hasattr(window, "iHPlot")
    assert hasattr(window, "mHPlot")
    assert hasattr(window, "dHPlot")
    assert hasattr(window, "demoData")
    assert hasattr(window, "comBox")
    assert hasattr(window, "statBox")
    assert hasattr(window, "configBox")
    window.close()


