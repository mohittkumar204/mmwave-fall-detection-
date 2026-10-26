"""Camera probing and utility functions."""

from __future__ import annotations

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    cv2 = None
    HAS_CV2 = False


def probe_cameras(max_probe: int = 3) -> list[int]:
    """Find available optical camera device indices with quick frame probe."""
    if not HAS_CV2 or cv2 is None:
        return []
    available = []
    for idx in range(max_probe):
        try:
            test_cap = cv2.VideoCapture(idx)
            if test_cap.isOpened():
                ret, _ = test_cap.read()
                if ret:
                    available.append(idx)
                test_cap.release()
        except Exception:
            pass
    return available
