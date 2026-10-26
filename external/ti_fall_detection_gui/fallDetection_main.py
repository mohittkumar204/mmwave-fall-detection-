"""Texas Instruments mmWave People Counting & Fall Detection GUI Entrypoint."""

import sys
from pathlib import Path

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.gui.ti_mmwave_fall_gui import Window, main

if __name__ == '__main__':
    main()
