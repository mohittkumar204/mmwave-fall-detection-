"""Texas Instruments mmWave People Counting & Fall Detection GUI Launcher.

Launches the authentic Texas Instruments 'mmWave People Counting' visualizer
integrated with your trained mmWave fall-detection and posture classification models.

Usage:
    python run_gui.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Pre-import torch before Qt to avoid Windows DLL initialization conflict on Python 3.13
try:
    import torch
except Exception:
    pass

from src.gui.ti_mmwave_fall_gui import main

if __name__ == '__main__':
    main()
