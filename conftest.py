import sys
import os
from pathlib import Path

# Add project root to sys.path
root = Path(__file__).resolve().parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

# Pre-import torch before Qt to prevent Windows DLL initialization conflict (WinError 1114)
try:
    import torch
except Exception:
    pass

# When running tests in headless/offscreen mode, mock checkOpenGLVersion
# because Windows offscreen platform does not provide a hardware OpenGL context.
try:
    import pyqtgraph.opengl as gl
    gl.GLViewWidget.checkOpenGLVersion = lambda self, *args, **kwargs: None
except Exception:
    pass

import gc
import pytest

@pytest.fixture(autouse=True)
def cleanup_after_test():
    yield
    gc.collect()

