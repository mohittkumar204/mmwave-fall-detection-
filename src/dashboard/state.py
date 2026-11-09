"""Shared in-memory state store for the ElderGuard dashboard.

Both the GUI (writer) and Flask server (reader) access this module.
All access is thread-safe via a single lock.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Dict, List

_lock = threading.Lock()

_state: Dict[str, Any] = {
    "status": "UNKNOWN",           # "UPRIGHT" | "FALL" | "UNKNOWN"
    "height": 0.0,                 # metres
    "tid": 0,                      # target ID
    "confidence": 0.0,
    "last_update": 0.0,
    "system_start": time.time(),
    "radar_connected": False,
    "esp32_connected": False,
    "telegram_connected": False,
}

_alerts: List[Dict[str, Any]] = []   # newest first
_fall_count: int = 0
_response_times: List[float] = []    # latency samples for avg


def update_status(
    status: str,
    height: float = 0.0,
    tid: int = 0,
    confidence: float = 0.0,
    radar_connected: bool = True,
    esp32_connected: bool = False,
    telegram_connected: bool = False,
    latency_ms: float = 0.0,
) -> None:
    """Called by the GUI every frame to push the latest detection state."""
    global _fall_count
    with _lock:
        prev_status = _state["status"]
        _state["status"] = status
        _state["height"] = round(height, 3)
        _state["tid"] = tid
        _state["confidence"] = round(confidence, 4)
        _state["last_update"] = time.time()
        _state["radar_connected"] = radar_connected
        _state["esp32_connected"] = esp32_connected
        _state["telegram_connected"] = telegram_connected

        if latency_ms > 0:
            _response_times.append(latency_ms)
            if len(_response_times) > 200:
                _response_times.pop(0)

        # Record new fall event (edge-triggered — only on transition to FALL)
        if status == "FALL" and prev_status != "FALL":
            _fall_count += 1
            _alerts.insert(0, {
                "timestamp": time.strftime("%H:%M:%S"),
                "tid": tid,
                "height": round(height, 2),
                "status": "FALL",
            })
            if len(_alerts) > 100:
                _alerts.pop()

        elif status == "UPRIGHT" and prev_status == "FALL":
            # Log recovery event
            _alerts.insert(0, {
                "timestamp": time.strftime("%H:%M:%S"),
                "tid": tid,
                "height": round(height, 2),
                "status": "UPRIGHT",
            })
            if len(_alerts) > 100:
                _alerts.pop()


def get_state() -> Dict[str, Any]:
    """Return a snapshot of the current state (safe copy)."""
    with _lock:
        avg_rt = (sum(_response_times) / len(_response_times)) if _response_times else 0.0
        uptime_s = int(time.time() - _state["system_start"])
        hours, rem = divmod(uptime_s, 3600)
        mins, secs = divmod(rem, 60)
        return {
            **_state,
            "fall_count": _fall_count,
            "avg_response_ms": round(avg_rt, 1),
            "uptime": f"{hours:02d}:{mins:02d}:{secs:02d}",
        }


def get_alerts(limit: int = 20) -> List[Dict[str, Any]]:
    """Return the most recent alert entries."""
    with _lock:
        return list(_alerts[:limit])
