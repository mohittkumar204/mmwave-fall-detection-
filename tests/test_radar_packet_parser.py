"""Unit Tests for Radar Packet Parsing, Malformed Packet Recovery, and Empty Frame Handling."""

import struct
import pytest

from src.inference.live_fall_detector import LiveFallDetector
from src.radar.packet_parser import (
    HEADER_STRUCT,
    MAGIC_WORD,
    FrameParseError,
    ParsedFrame,
    Point,
    Target,
    parse_radar_frame,
)


def test_missing_magic_word_raises_error():
    corrupted_data = b"NOT_TI_MAGIC_WORD" + b"\x00" * 40
    with pytest.raises(FrameParseError, match="TI magic word"):
        parse_radar_frame(corrupted_data)


def test_packet_shorter_than_header_raises_error():
    short_data = MAGIC_WORD + b"\x00\x01\x02"
    with pytest.raises(FrameParseError, match="Frame shorter than header"):
        parse_radar_frame(short_data)


def test_incomplete_packet_recovery():
    # Header says 100 bytes, but packet provides only 40 bytes
    header_bytes = struct.pack(
        HEADER_STRUCT,
        0x0708050603040102, # magic
        0x03050000,         # version
        100,                # total_packet_len
        0x6843,             # platform
        1,                  # frame_num
        1000,               # cpu cycles
        0,                  # num objects
        0,                  # num tlvs
        0                   # subframe
    )
    parsed = parse_radar_frame(header_bytes)
    assert parsed.parse_error != "", "Parser should flag incomplete packet rather than crashing"
    assert "Incomplete packet" in parsed.parse_error


def test_live_detector_empty_frame_handling():
    detector = LiveFallDetector()
    empty_frame = ParsedFrame(frame_num=10, targets=[], points=[])
    # Detector must return empty list and NOT throw exception
    res = detector.process_frame(empty_frame)
    assert res == []


def test_live_detector_zero_points_target_handling():
    detector = LiveFallDetector()
    # Target detected but 0 point cloud points associated
    dummy_target = Target(
        tid=0, pos_x=0.0, pos_y=1.5, pos_z=1.2,
        vel_x=0.0, vel_y=0.0, vel_z=0.0,
        acc_x=0.0, acc_y=0.0, acc_z=0.0
    )
    frame = ParsedFrame(frame_num=1, targets=[dummy_target], points=[])
    # Must process cleanly with zero-padded points
    res = detector.process_frame(frame)
    assert len(res) == 1
    assert res[0]["tid"] == 0
    assert res[0]["status"] == "BUFFERING"
