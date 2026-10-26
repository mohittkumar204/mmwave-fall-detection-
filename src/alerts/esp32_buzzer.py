"""ESP32 Buzzer Alert Dispatcher for Real-Time Fall Detection.

Communicates with an ESP32 microcontroller connected via USB Serial (COM port)
or Wireless UDP. When a confirmed fall is detected, it triggers the ESP32 buzzer
to sound an audible emergency alarm.

Features:
- Configurable Beep Patterns: SIREN, DOUBLE_BEEP, PULSE, CONTINUOUS.
- Debounced Triggering: Prevents rapid serial write flooding during consecutive fall frames.
- Acknowledge & Auto-Silence: Automatically silences after hold duration or manual acknowledge.
- Fault-Tolerant: Catches serial disconnections, socket timeouts, and communication errors.
- Non-Blocking & Thread-Safe: Uses locking to prevent serial port contention.
"""

from __future__ import annotations

import os
import socket
import threading
import time
from typing import Any, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import serial
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False


SUPPORTED_BEEP_PATTERNS = ["SIREN", "DOUBLE_BEEP", "PULSE", "CONTINUOUS"]


class Esp32BuzzerAlert:
    """Manages serial or network communication with ESP32 buzzer hardware."""

    def __init__(
        self,
        port: Optional[str] = None,
        baud_rate: Optional[int] = None,
        wifi_ip: Optional[str] = None,
        wifi_port: Optional[int] = None,
        beep_pattern: Optional[str] = None,
        buzz_hold_seconds: Optional[float] = None,
        enabled: Optional[bool] = None,
    ):
        # Resolve configuration from arguments or environment (.env)
        self.enabled = (
            enabled if enabled is not None
            else os.getenv("ESP32_ENABLED", "True").strip().lower() in ("true", "1", "yes")
        )

        self.port = port or os.getenv("ESP32_PORT", "").strip() or None
        self.baud_rate = baud_rate or int(os.getenv("ESP32_BAUD", "115200"))
        self.wifi_ip = wifi_ip or os.getenv("ESP32_WIFI_IP", "").strip() or None
        self.wifi_port = wifi_port or int(os.getenv("ESP32_WIFI_PORT", "4210"))
        self.beep_pattern = (beep_pattern or os.getenv("ESP32_BEEP_PATTERN", "SIREN")).upper()
        if self.beep_pattern not in SUPPORTED_BEEP_PATTERNS:
            self.beep_pattern = "SIREN"

        self.buzz_hold_seconds = (
            buzz_hold_seconds if buzz_hold_seconds is not None
            else float(os.getenv("ESP32_HOLD_SECONDS", "5.0"))
        )

        self.ser: Optional[serial.Serial] = None
        self.udp_sock: Optional[socket.socket] = None
        self.is_connected = False
        self.last_trigger_time = 0.0
        self._last_sent_time = 0.0
        self._last_reconnect_attempt = 0.0
        self.is_buzzing = False
        self._lock = threading.Lock()

        if self.enabled:
            self._connect()

    def _connect(self):
        """Initialize USB Serial or UDP network transport."""
        with self._lock:
            # 1. Try Serial USB connection
            if self.port and HAS_SERIAL:
                try:
                    self.ser = serial.Serial(
                        port=self.port,
                        baudrate=self.baud_rate,
                        timeout=0.1,
                        write_timeout=0.1,
                    )
                    time.sleep(0.5)  # Allow ESP32 boot/reset stabilization
                    self.is_connected = True
                    print(f"[esp32-buzzer] Connected to ESP32 on {self.port} at {self.baud_rate} baud.")
                except Exception as e:
                    self.ser = None
                    self.is_connected = False
                    print(f"[esp32-buzzer] Note: Unable to open ESP32 serial port '{self.port}': {e}")

            # 2. Setup UDP socket if IP provided
            if self.wifi_ip:
                try:
                    self.udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    self.udp_sock.setblocking(False)
                    self.is_connected = True
                    print(f"[esp32-buzzer] Configured WiFi UDP target: {self.wifi_ip}:{self.wifi_port}")
                except Exception as e:
                    self.udp_sock = None
                    print(f"[esp32-buzzer] Failed to initialize UDP socket: {e}")

    def _send_payload(self, msg: bytes) -> bool:
        """Internal helper to dispatch bytes over active transport with error recovery."""
        if not self.enabled:
            return False

        success = False
        with self._lock:
            # Serial dispatch
            if self.ser:
                try:
                    if not self.ser.is_open:
                        self.ser.open()
                    self.ser.write(msg)
                    self.ser.flush()
                    self.is_connected = True
                    success = True
                except Exception as e:
                    self.is_connected = False
                    # Close broken port so reconnect can be attempted later
                    try:
                        self.ser.close()
                    except Exception:
                        pass
                    print(f"[esp32-buzzer] Warning: Serial communication error on {self.port}: {e}")

            # UDP dispatch
            if self.udp_sock and self.wifi_ip:
                try:
                    self.udp_sock.sendto(msg, (self.wifi_ip, self.wifi_port))
                    self.is_connected = True
                    success = True
                except Exception as e:
                    print(f"[esp32-buzzer] Warning: UDP communication error to {self.wifi_ip}: {e}")

        # If not connected, attempt throttled reconnect
        if not success and self.port and (time.time() - self._last_reconnect_attempt > 5.0):
            self._last_reconnect_attempt = time.time()
            self._try_reconnect()

        return success

    def _try_reconnect(self):
        """Attempt non-blocking reconnection if port is configured."""
        if not HAS_SERIAL or not self.port:
            return
        try:
            with self._lock:
                if self.ser is None or not self.ser.is_open:
                    self.ser = serial.Serial(
                        port=self.port,
                        baudrate=self.baud_rate,
                        timeout=0.1,
                        write_timeout=0.1,
                    )
                    self.is_connected = True
                    print(f"[esp32-buzzer] Reconnected to ESP32 on {self.port}.")
        except Exception:
            self.is_connected = False

    def trigger_fall_alarm(self, pattern: Optional[str] = None, duration_seconds: Optional[float] = None) -> bool:
        """Send FALL alert command to ESP32 to activate buzzer.
        
        Debounced: If already buzzing and sent recently, suppresses serial message flooding.
        """
        if not self.enabled:
            return False

        now = time.time()
        self.last_trigger_time = now

        # Debounce: if already buzzing and sent within the last 2.0s, keep state active without flooding serial
        if self.is_buzzing and (now - self._last_sent_time < 2.0):
            return True

        pat = (pattern or self.beep_pattern).upper()
        if pat not in SUPPORTED_BEEP_PATTERNS:
            pat = "SIREN"

        dur_ms = int((duration_seconds or self.buzz_hold_seconds) * 1000)
        # Form formatted payload: FALL:<PATTERN>:<DURATION_MS>\n
        payload = f"FALL:{pat}:{dur_ms}\n".encode("utf-8")

        self.is_buzzing = True
        self._last_sent_time = now
        return self._send_payload(payload)

    def clear_alarm(self, force: bool = False) -> bool:
        """Send CLEAR command to ESP32 to stop buzzer if hold time has passed."""
        if not self.is_buzzing:
            return True

        now = time.time()
        # Ensure minimum hold duration unless forced
        if not force and (now - self.last_trigger_time < self.buzz_hold_seconds):
            return False

        self.is_buzzing = False
        payload = b"CLEAR\n"
        return self._send_payload(payload)

    def acknowledge_alarm(self) -> bool:
        """Immediately silence buzzer and acknowledge emergency event."""
        self.is_buzzing = False
        payload = b"ACK\n"
        return self._send_payload(payload)

    def test_beep(self) -> bool:
        """Send TEST command for a single verification chirp."""
        payload = b"TEST\n"
        return self._send_payload(payload)

    def get_status(self) -> dict[str, Any]:
        """Return diagnostic status dictionary."""
        transport = "NONE"
        if self.ser and self.ser.is_open:
            transport = f"SERIAL ({self.port})"
        elif self.wifi_ip:
            transport = f"WIFI ({self.wifi_ip}:{self.wifi_port})"

        return {
            "enabled": self.enabled,
            "connected": self.is_connected,
            "is_buzzing": self.is_buzzing,
            "transport": transport,
            "pattern": self.beep_pattern,
            "hold_seconds": self.buzz_hold_seconds,
        }

    def close(self):
        """Release connections cleanly."""
        if self.is_buzzing:
            try:
                self.acknowledge_alarm()
            except Exception:
                pass

        with self._lock:
            if self.ser:
                try:
                    self.ser.close()
                except Exception:
                    pass
                self.ser = None

            if self.udp_sock:
                try:
                    self.udp_sock.close()
                except Exception:
                    pass
                self.udp_sock = None

            self.is_connected = False
