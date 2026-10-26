"""IWR6843 mmWave Radar Packet and TLV Parser.

Decodes UART byte streams from the TI 3D People Tracking Demo (IWR6843ISK-ODS):
- TLV 1010: Target List (TID, x, y, z, vx, vy, vz, ax, ay, az, covariance)
- TLV 1011: Target Index (Point-to-Target association IDs)
- TLV 1012: Target Height (maxZ, minZ per target)
- TLV 1020: Compressed Spherical Points (elevation, azimuth, doppler, range, snr)
- TLV 1021: Presence Indication
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import struct
from typing import List, Optional


MAGIC_WORD = b"\x02\x01\x04\x03\x06\x05\x08\x07"

TLV_TARGET_LIST = 1010
TLV_TARGET_INDEX = 1011
TLV_TARGET_HEIGHT = 1012
TLV_COMPRESSED_POINTS = 1020
TLV_PRESENCE = 1021

HEADER_STRUCT = "<Q8I"
TLV_HEADER_STRUCT = "<2I"
TARGET_STRUCT = "<I27f"
HEIGHT_STRUCT = "<I2f"
TARGET_INDEX_STRUCT = "<B"
POINT_UNIT_STRUCT = "<5f"
COMPRESSED_POINT_STRUCT = "<2bh2H"
PRESENCE_STRUCT = "<I"

HEADER_LEN = struct.calcsize(HEADER_STRUCT)
TLV_HEADER_LEN = struct.calcsize(TLV_HEADER_STRUCT)
TARGET_LEN = struct.calcsize(TARGET_STRUCT)
HEIGHT_LEN = struct.calcsize(HEIGHT_STRUCT)
POINT_UNIT_LEN = struct.calcsize(POINT_UNIT_STRUCT)
COMPRESSED_POINT_LEN = struct.calcsize(COMPRESSED_POINT_STRUCT)


@dataclass(frozen=True)
class Target:
    tid: int
    pos_x: float
    pos_y: float
    pos_z: float
    vel_x: float
    vel_y: float
    vel_z: float
    acc_x: float
    acc_y: float
    acc_z: float
    ec: tuple[float, ...] = ()


@dataclass(frozen=True)
class TargetHeight:
    tid: int
    max_z: float
    min_z: float


@dataclass(frozen=True)
class Point:
    point_index: int
    elevation: float
    azimuth: float
    doppler: float
    range_val: float
    snr: float
    x: float
    y: float
    z: float
    track_index: int = 255


@dataclass
class ParsedFrame:
    magic: int = 0
    version: int = 0
    total_packet_len: int = 0
    platform: int = 0
    frame_num: int = 0
    time_cpu_cycles: int = 0
    num_detected_obj: int = 0
    num_tlvs: int = 0
    sub_frame_num: int = 0
    targets: list[Target] = field(default_factory=list)
    heights: list[TargetHeight] = field(default_factory=list)
    target_indices: list[int] = field(default_factory=list)
    points: list[Point] = field(default_factory=list)
    presence: int = 0
    parse_error: str = ""


class FrameParseError(ValueError):
    pass


def parse_radar_frame(frame_data: bytes) -> ParsedFrame:
    """Parse complete radar UART frame bytes."""
    if len(frame_data) < HEADER_LEN:
        raise FrameParseError(f"Frame shorter than header: {len(frame_data)} bytes")
    if frame_data[: len(MAGIC_WORD)] != MAGIC_WORD:
        raise FrameParseError("Frame does not start with TI magic word")

    hdr = struct.unpack(HEADER_STRUCT, frame_data[:HEADER_LEN])
    parsed = ParsedFrame(
        magic=hdr[0],
        version=hdr[1],
        total_packet_len=hdr[2],
        platform=hdr[3],
        frame_num=hdr[4],
        time_cpu_cycles=hdr[5],
        num_detected_obj=hdr[6],
        num_tlvs=hdr[7],
        sub_frame_num=hdr[8],
    )

    if parsed.total_packet_len > len(frame_data):
        parsed.parse_error = f"Incomplete packet: expected {parsed.total_packet_len}, got {len(frame_data)}"
        return parsed

    offset = HEADER_LEN
    for _ in range(parsed.num_tlvs):
        if offset + TLV_HEADER_LEN > len(frame_data):
            parsed.parse_error = f"TLV header overrun at {offset}"
            break

        tlv_type, tlv_len = struct.unpack(TLV_HEADER_STRUCT, frame_data[offset : offset + TLV_HEADER_LEN])
        offset += TLV_HEADER_LEN

        if offset + tlv_len > len(frame_data):
            parsed.parse_error = f"TLV {tlv_type} length {tlv_len} overruns packet"
            break

        payload = frame_data[offset : offset + tlv_len]
        offset += tlv_len

        if tlv_type == TLV_TARGET_LIST:
            parsed.targets = _parse_target_list(payload)
        elif tlv_type == TLV_TARGET_HEIGHT:
            parsed.heights = _parse_target_heights(payload)
        elif tlv_type == TLV_TARGET_INDEX:
            parsed.target_indices = _parse_target_indices(payload)
        elif tlv_type == TLV_COMPRESSED_POINTS:
            parsed.points = _parse_compressed_points(payload)
        elif tlv_type == TLV_PRESENCE:
            if len(payload) >= 4:
                parsed.presence = struct.unpack(PRESENCE_STRUCT, payload[:4])[0]

    # Associate track indices with points if both are present
    if parsed.points and parsed.target_indices:
        for idx, pt in enumerate(parsed.points):
            if idx < len(parsed.target_indices):
                object.__setattr__(pt, "track_index", parsed.target_indices[idx])

    return parsed


def _parse_target_list(payload: bytes) -> list[Target]:
    targets = []
    count = len(payload) // TARGET_LEN
    for i in range(count):
        raw = struct.unpack_from(TARGET_STRUCT, payload, i * TARGET_LEN)
        targets.append(
            Target(
                tid=raw[0],
                pos_x=raw[1],
                pos_y=raw[2],
                pos_z=raw[3],
                vel_x=raw[4],
                vel_y=raw[5],
                vel_z=raw[6],
                acc_x=raw[7],
                acc_y=raw[8],
                acc_z=raw[9],
                ec=raw[10:],
            )
        )
    return targets


def _parse_target_heights(payload: bytes) -> list[TargetHeight]:
    heights = []
    count = len(payload) // HEIGHT_LEN
    for i in range(count):
        raw = struct.unpack_from(HEIGHT_STRUCT, payload, i * HEIGHT_LEN)
        heights.append(TargetHeight(tid=raw[0], max_z=raw[1], min_z=raw[2]))
    return heights


def _parse_target_indices(payload: bytes) -> list[int]:
    return [int(b) for b in payload]


def _parse_compressed_points(payload: bytes) -> list[Point]:
    if len(payload) < POINT_UNIT_LEN:
        return []
    units = struct.unpack_from(POINT_UNIT_STRUCT, payload, 0)
    elevation_unit, azimuth_unit, doppler_unit, range_unit, snr_unit = units

    points = []
    num_pts = (len(payload) - POINT_UNIT_LEN) // COMPRESSED_POINT_LEN
    for i in range(num_pts):
        offset = POINT_UNIT_LEN + (i * COMPRESSED_POINT_LEN)
        elev_raw, azim_raw, doppler_raw, range_raw, snr_raw = struct.unpack_from(
            COMPRESSED_POINT_STRUCT, payload, offset
        )
        elev = elev_raw * elevation_unit
        azim = azim_raw * azimuth_unit
        doppler = doppler_raw * doppler_unit
        r = range_raw * range_unit
        snr = snr_raw * snr_unit

        # Spherical to Cartesian coordinate transformation
        x = r * math.cos(elev) * math.sin(azim)
        y = r * math.cos(elev) * math.cos(azim)
        z = r * math.sin(elev)

        points.append(
            Point(
                point_index=i,
                elevation=elev,
                azimuth=azim,
                doppler=doppler,
                range_val=r,
                snr=snr,
                x=x,
                y=y,
                z=z,
            )
        )
    return points
