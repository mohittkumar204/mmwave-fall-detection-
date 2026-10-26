"""Ghost Target & Noise Filter Pipeline for IWR6843 mmWave Radar.

Reduces false/ghost tracks to the single most confident real human target
and suppresses low-quality point cloud noise before inference.

Root causes of ghost tracks:
  1. FMCW radar multi-path reflections off walls and furniture
  2. Low-SNR clutter points spawning phantom tracker entries
  3. TI People Tracker not configured tightly enough (gating, allocation)

This filter operates in software after parsing, before fall detection.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Optional
import math
import numpy as np

from src.radar.packet_parser import ParsedFrame, Point, Target


# ─────────────────────────────────────────────────────────────────────
#  Tunable Thresholds  (safe defaults – tighten to reduce false alarms)
# ─────────────────────────────────────────────────────────────────────

# Minimum SNR a point must have to be kept (noise floor removal)
MIN_POINT_SNR: float = 8.0

# Minimum number of associated points a target must have to be kept
# 1 point allows valid distant/obscured targets without dropping real people
MIN_ASSOCIATED_POINTS: int = 1

# Track maturity gate: 1 frame allows immediate buffering without latency
MIN_TRACK_AGE_FRAMES: int = 1

# Maximum range from the sensor at which targets are accepted (m)
MAX_RANGE_M: float = 6.0

# A human centroid cannot be more than 2.3 m above the sensor mount height.
# Anything above is a ceiling/multipath reflection.
MAX_CENTROID_Z_M: float = 2.30

# Minimum centroid height – targets below floor level are noise
MIN_CENTROID_Z_M: float = -0.20

# Maximum allowable speed – real humans in a room do not exceed ~3.0 m/s.
MAX_HUMAN_SPEED_MPS: float = 3.0

# If multiple targets exist and their centroids are within this distance (m),
# merge near-duplicate multipath reflections while preserving distinct targets.
MERGE_DISTANCE_M: float = 0.85

# FALL DETECTION TUNING ─────────────────────────────────────────────
# Downward centroid velocity (m/s) during collapse
FALL_VEL_Z_THRESHOLD: float = -0.65

# Downward vertical acceleration (m/s^2)
FALL_ACC_Z_THRESHOLD: float = -0.30

# Minimum height drop from upright baseline (m) to confirm a fall
FALL_HEIGHT_DROP_M: float = 0.40

# Floor-level threshold (m) for post-impact resting posture
FALL_FLOOR_Z_M: float = 0.55


# ─────────────────────────────────────────────────────────────────────

@dataclass
class TrackState:
    """Per-target quality bookkeeping."""
    tid: int
    age_frames: int = 0
    point_history: deque = None
    last_z: float = 1.2
    baseline_z: float = 1.2
    baseline_set: bool = False

    def __post_init__(self):
        self.point_history = deque(maxlen=6)


class RadarNoiseFilter:
    """Stateful noise/ghost filter.  Call filter_frame() each radar frame."""

    def __init__(
        self,
        min_snr: float = MIN_POINT_SNR,
        min_points: int = MIN_ASSOCIATED_POINTS,
        min_age: int = MIN_TRACK_AGE_FRAMES,
        max_range: float = MAX_RANGE_M,
        max_z: float = MAX_CENTROID_Z_M,
        min_z: float = MIN_CENTROID_Z_M,
        max_speed: float = MAX_HUMAN_SPEED_MPS,
        merge_dist: float = MERGE_DISTANCE_M,
        single_person_mode: bool = True,
    ):
        self.min_snr = min_snr
        self.min_points = min_points
        self.min_age = min_age
        self.max_range = max_range
        self.max_z = max_z
        self.min_z = min_z
        self.max_speed = max_speed
        self.merge_dist = merge_dist
        self.single_person_mode = single_person_mode

        # Track age & quality state per TID
        self._states: dict[int, TrackState] = {}
        self._seen_tids: set[int] = set()
        self._active_tids: set[int] = set()
        self._primary_tid: Optional[int] = None

    # ----------------------------------------------------------------

    def filter_frame(self, frame: ParsedFrame) -> ParsedFrame:
        """Return a new ParsedFrame with noise targets and low-SNR points removed."""
        # Step 1: Filter low-SNR point cloud
        clean_points = self._filter_points(frame.points)

        # Build point count per TID from cleaned points
        pts_by_tid: dict[int, list[Point]] = defaultdict(list)
        unassociated_pts: list[Point] = []
        for pt in clean_points:
            if pt.track_index not in {253, 254, 255}:
                pts_by_tid[pt.track_index].append(pt)
            else:
                unassociated_pts.append(pt)

        # If points lacked track_index (TLV 8 missing or unassigned), associate geometrically within 1.2m
        if unassociated_pts and frame.targets:
            for pt in unassociated_pts:
                best_tid = None
                best_dist = 1.2
                for t in frame.targets:
                    dist = math.sqrt((pt.x - t.pos_x)**2 + (pt.y - t.pos_y)**2 + (pt.z - t.pos_z)**2)
                    if dist < best_dist:
                        best_dist = dist
                        best_tid = t.tid
                if best_tid is not None:
                    pts_by_tid[best_tid].append(pt)

        # Step 2: Update track ages and prune stale tracks
        current_tids = {t.tid for t in frame.targets}
        gone_tids = self._active_tids - current_tids
        for tid in gone_tids:
            if tid in self._states:
                del self._states[tid]
        self._active_tids = current_tids

        for t in frame.targets:
            if t.tid not in self._states:
                self._states[t.tid] = TrackState(tid=t.tid)
            self._states[t.tid].age_frames += 1
            self._states[t.tid].point_history.append(len(pts_by_tid[t.tid]))

        # Step 3: Quality-gate each target
        kept_targets: list[Target] = []
        for t in frame.targets:
            state = self._states.get(t.tid)
            n_pts = len(pts_by_tid[t.tid])

            # -- Geometry gates --
            range_2d = math.sqrt(t.pos_x**2 + t.pos_y**2)
            speed = math.sqrt(t.vel_x**2 + t.vel_y**2 + t.vel_z**2)

            if range_2d > self.max_range:
                continue
            if t.pos_z > self.max_z or t.pos_z < self.min_z:
                continue
            if speed > self.max_speed:
                continue

            # -- Point density gate (only when min_points > 0 and point cloud has points) --
            if self.min_points > 0 and len(clean_points) >= self.min_points:
                avg_pts = (
                    sum(state.point_history) / len(state.point_history)
                    if (state and state.point_history) else n_pts
                )
                if avg_pts < self.min_points and n_pts < self.min_points:
                    continue

            # -- Track maturity gate --
            if state and state.age_frames < self.min_age:
                continue

            kept_targets.append(t)

        # Step 4: Merge near-duplicate ghost targets
        kept_targets = self._merge_duplicates(kept_targets)

        # Step 5: Single-person mode – keep only the best 1 target
        if self.single_person_mode and len(kept_targets) > 1:
            kept_targets = self._pick_best_target(kept_targets, pts_by_tid)

        # Rebuild the frame with filtered data
        kept_tids = {t.tid for t in kept_targets}
        kept_heights = [h for h in frame.heights if h.tid in kept_tids]

        # Patch the ParsedFrame - retain all clean points so 3D visualizer receives the point cloud!
        frame.targets = kept_targets
        frame.points = clean_points
        frame.heights = kept_heights

        return frame


    # ----------------------------------------------------------------

    def _filter_points(self, points: list[Point]) -> list[Point]:
        """Remove low-SNR and out-of-range raw point cloud returns."""
        kept = []
        for pt in points:
            if pt.snr < self.min_snr:
                continue
            if pt.range_val > self.max_range:
                continue
            if pt.z > self.max_z or pt.z < self.min_z:
                continue
            kept.append(pt)
        return kept

    def _merge_duplicates(self, targets: list[Target]) -> list[Target]:
        """If two targets are within merge_dist of each other, discard the second."""
        if len(targets) <= 1:
            return targets
        merged: list[Target] = []
        for t in targets:
            too_close = False
            for m in merged:
                dist = math.sqrt(
                    (t.pos_x - m.pos_x)**2
                    + (t.pos_y - m.pos_y)**2
                    + (t.pos_z - m.pos_z)**2
                )
                if dist < self.merge_dist:
                    too_close = True
                    break
            if not too_close:
                merged.append(t)
        return merged

    def _pick_best_target(
        self,
        targets: list[Target],
        pts_by_tid: dict[int, list[Point]],
    ) -> list[Target]:
        """In single-person mode, keep the one most confident real human target.
        Penalizes ceiling/floor multipath reflections and uses track continuity hysteresis
        to eliminate target-hopping oscillations between adjacent tracks."""
        if not targets:
            self._primary_tid = None
            return []

        def score(t: Target) -> tuple:
            n_pts = len(pts_by_tid.get(t.tid, []))
            rng = math.sqrt(t.pos_x**2 + t.pos_y**2)
            age = self._states[t.tid].age_frames if t.tid in self._states else 0
            # Real human centroid is realistically between 0.3m and 2.2m
            z_weight = 1.8 if (0.3 <= t.pos_z <= 2.2) else 0.3
            # Continuity bonus if this was already the active primary target
            continuity = 6.0 if (self._primary_tid is not None and t.tid == self._primary_tid) else 0.0
            # Weight point count, track maturity, and continuity
            rank_score = z_weight * (n_pts * 2.0 + min(age, 25) * 0.5 + continuity)
            return (rank_score, -rng)

        best_target = max(targets, key=score)
        self._primary_tid = best_target.tid
        return [best_target]
