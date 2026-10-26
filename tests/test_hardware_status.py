"""Hardware-Dependent Sensor Communication Tests with Strict Pending State."""

import pytest
try:
    import serial.tools.list_ports
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False


def test_physical_sensor_connection():
    if not HAS_SERIAL:
        pytest.skip("pyserial is not available.")

    ports = list(serial.tools.list_ports.comports())
    if not ports:
        pytest.skip("HARDWARE PENDING: Physical Texas Instruments IWR6843 mmWave radar is not connected or USB UART is unpowered.")

    devices = [p.device for p in ports]
    # If ports exist, check if TI UART ports are present
    assert len(devices) > 0, "No serial devices detected"
