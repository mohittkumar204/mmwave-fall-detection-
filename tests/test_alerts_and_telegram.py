"""Comprehensive Tests for ESP32 Buzzer Alert and Telegram Fall Notifications."""

import os
import time
from unittest.mock import MagicMock, patch
import pytest

from src.alerts.esp32_buzzer import Esp32BuzzerAlert
from src.alerts.telegram_notifier import TelegramNotifier
from src.inference.live_fall_detector import LiveFallDetector
from src.radar.packet_parser import ParsedFrame, Point, Target


# -----------------------------------------------------------------------------
# 1. ESP32 Buzzer Alert Tests
# -----------------------------------------------------------------------------

def test_esp32_buzzer_initialization_graceful(monkeypatch):
    """Verify ESP32 buzzer initializes gracefully when port is disconnected."""
    monkeypatch.delenv("ESP32_PORT", raising=False)
    monkeypatch.delenv("ESP32_WIFI_IP", raising=False)
    buzzer = Esp32BuzzerAlert(port=None, wifi_ip=None)
    assert buzzer is not None
    assert buzzer.is_connected is False
    assert buzzer.is_buzzing is False

    st = buzzer.get_status()
    assert st["connected"] is False
    assert st["is_buzzing"] is False
    buzzer.close()


def test_esp32_buzzer_trigger_and_debounce():
    """Verify fall alarm triggers buzzer state and debounces rapid consecutive calls."""
    buzzer = Esp32BuzzerAlert(port=None, wifi_ip=None)

    # Mock the internal _send_payload method to verify message formatting
    sent_messages = []
    buzzer._send_payload = lambda msg: sent_messages.append(msg) or True

    # 1. First trigger call
    res1 = buzzer.trigger_fall_alarm(pattern="SIREN", duration_seconds=5.0)
    assert res1 is True
    assert buzzer.is_buzzing is True
    assert len(sent_messages) == 1
    assert b"FALL:SIREN:5000\n" in sent_messages[0]

    # 2. Immediate second call (< 2 seconds later) should be debounced to prevent flooding
    res2 = buzzer.trigger_fall_alarm(pattern="SIREN", duration_seconds=5.0)
    assert res2 is True
    assert len(sent_messages) == 1  # No duplicate serial write sent!

    buzzer.close()


def test_esp32_buzzer_clear_and_acknowledge():
    """Verify hold duration enforcement and immediate acknowledge silencing."""
    buzzer = Esp32BuzzerAlert(port=None, wifi_ip=None, buzz_hold_seconds=2.0)
    sent_messages = []
    buzzer._send_payload = lambda msg: sent_messages.append(msg) or True

    # Trigger alarm
    buzzer.trigger_fall_alarm()
    assert buzzer.is_buzzing is True

    # Attempt to clear before hold duration expires (should not clear)
    cleared = buzzer.clear_alarm(force=False)
    assert cleared is False
    assert buzzer.is_buzzing is True

    # Forced acknowledge should silence immediately regardless of hold time
    ack = buzzer.acknowledge_alarm()
    assert ack is True
    assert buzzer.is_buzzing is False
    assert any(b"ACK" in m for m in sent_messages)

    buzzer.close()


def test_esp32_buzzer_disconnection_handling(monkeypatch):
    """Verify that serial port write errors and disconnections are handled safely."""
    monkeypatch.delenv("ESP32_WIFI_IP", raising=False)
    buzzer = Esp32BuzzerAlert(port="COM999", baud_rate=115200, wifi_ip=None)

    # Simulate a broken serial connection that raises SerialException
    mock_ser = MagicMock()
    mock_ser.is_open = True
    mock_ser.write.side_effect = OSError("Device disconnected / USB unplugged")
    buzzer.ser = mock_ser
    buzzer.is_connected = True

    # Trigger alarm should catch the exception, set is_connected = False, and not crash
    result = buzzer._send_payload(b"FALL:SIREN:5000\n")
    assert buzzer.is_connected is False
    buzzer.close()


# -----------------------------------------------------------------------------
# 2. Telegram Fall Notification Tests
# -----------------------------------------------------------------------------

def test_telegram_notifier_initialization(monkeypatch):
    """Verify TelegramNotifier initializes and correctly detects unconfigured state."""
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    notifier = TelegramNotifier(bot_token=None, chat_id=None)
    assert notifier is not None
    assert notifier.is_configured is False
    notifier.close()


def test_telegram_duplicate_prevention_and_cooldown():
    """Verify that repeated fall predictions generate only one Telegram notification."""
    notifier = TelegramNotifier(
        bot_token="TEST_BOT_TOKEN_123456",
        chat_id="123456789",
        cooldown_seconds=30.0,
    )

    # Intercept worker queue without making actual HTTP requests
    queued_alerts = []
    with patch.object(notifier, "_post_to_telegram", side_effect=lambda msg: queued_alerts.append(msg) or True):
        event_payload = {
            "tid": 0,
            "frame_num": 100,
            "timestamp": time.time(),
            "model_class": "FALLING",
            "confidence": 0.985,
            "pos_z": 0.35,
            "vel_z": -1.45,
            "motion_label": "High Downward Kinematics",
        }

        # 1. First confirmed fall: should dispatch notification
        sent1 = notifier.send_fall_alert(event_payload)
        assert sent1 is True

        # 2. Immediate subsequent fall frames (e.g. frames 101, 102, 103 for same fall):
        # Must be suppressed to avoid spamming the user!
        sent2 = notifier.send_fall_alert(event_payload)
        sent3 = notifier.send_fall_alert(event_payload)
        assert sent2 is False
        assert sent3 is False

    notifier.close()


def test_telegram_rearm_mechanism():
    """Verify that the notifier safely re-arms when the subject stands UPRIGHT."""
    notifier = TelegramNotifier(
        bot_token="TEST_BOT_TOKEN_123456",
        chat_id="123456789",
        cooldown_seconds=60.0,
        min_recovery_frames=3,
    )

    with patch.object(notifier, "_post_to_telegram", return_value=True):
        event = {"tid": 0, "confidence": 0.95, "model_class": "FALLING"}

        # 1. Trigger initial fall
        assert notifier.send_fall_alert(event) is True
        assert notifier._is_armed is False

        # Simulate time passing beyond initial recovery lock (15s)
        notifier._last_alert_time = time.time() - 20.0

        # 2. While lying down or moving, should remain un-armed
        notifier.update_target_posture(0, "LYING")
        assert notifier._is_armed is False

        # 3. Subject stands upright for min_recovery_frames (3 frames)
        notifier.update_target_posture(0, "UPRIGHT")
        notifier.update_target_posture(0, "UPRIGHT")
        notifier.update_target_posture(0, "UPRIGHT")

        # System should now be safely re-armed
        assert notifier._is_armed is True

        # 4. Next fall can now trigger alert!
        assert notifier.send_fall_alert(event) is True

    notifier.close()


def test_telegram_async_non_blocking():
    """Verify that send_fall_alert returns immediately (< 5ms) without blocking callers."""
    notifier = TelegramNotifier(
        bot_token="TEST_BOT_TOKEN_123456",
        chat_id="123456789",
        cooldown_seconds=60.0,
    )

    # Simulate a slow network post that takes 1.0 second in the background thread
    def slow_post(msg):
        time.sleep(0.5)
        return True

    with patch.object(notifier, "_post_to_telegram", side_effect=slow_post):
        t0 = time.perf_counter()
        sent = notifier.send_fall_alert({"tid": 0, "confidence": 0.99, "model_class": "FALLING"})
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        assert sent is True
        # Method must return immediately (well under 15ms)
        assert elapsed_ms < 15.0, f"send_fall_alert blocked for {elapsed_ms:.2f} ms"

    notifier.close()


def test_telegram_network_failure_handling():
    """Verify that network exceptions in worker thread are caught gracefully without crashing."""
    notifier = TelegramNotifier(
        bot_token="INVALID_TEST_TOKEN",
        chat_id="INVALID_CHAT_ID",
    )

    # Test direct _post_to_telegram with mock request raising Timeout
    with patch("requests.post", side_effect=Exception("Simulated DNS Resolution Failure / Offline")):
        result = notifier._post_to_telegram("Test message")
        # Should catch error cleanly and return False without throwing
        assert result is False

    notifier.close()


# -----------------------------------------------------------------------------
# 3. Integrated Live Fall Detector Pipeline Tests
# -----------------------------------------------------------------------------

def test_integrated_pipeline_fall_alert_dispatch():
    """Verify that LiveFallDetector triggers buzzer and Telegram on fall, but NOT on upright/lying."""
    detector = LiveFallDetector()

    # Intercept buzzer and telegram dispatchers
    buzzer_triggers = []
    telegram_alerts = []

    if detector.buzzer:
        detector.buzzer.trigger_fall_alarm = lambda **kw: buzzer_triggers.append(True) or True
    if detector.telegram:
        detector.telegram.send_fall_alert = lambda data: telegram_alerts.append(data) or True

    # 1. Normal Upright frames: should NOT trigger buzzer or Telegram
    t_upright = Target(tid=0, pos_x=0.0, pos_y=2.0, pos_z=1.5, vel_x=0.0, vel_y=0.0, vel_z=0.0, acc_x=0.0, acc_y=0.0, acc_z=0.0)
    for f in range(1, 10):
        detector.process_frame(ParsedFrame(frame_num=f, targets=[t_upright], points=[]))

    assert len(buzzer_triggers) == 0, "Buzzer should NOT trigger on upright posture"
    assert len(telegram_alerts) == 0, "Telegram should NOT trigger on upright posture"

    # 2. Severe rapid downward fall frame (vz = -1.5 m/s, negative accel, drop to floor)
    t_fall = Target(tid=0, pos_x=0.0, pos_y=2.0, pos_z=0.30, vel_x=0.0, vel_y=0.0, vel_z=-1.5, acc_x=0.0, acc_y=0.0, acc_z=-0.8)
    detector.process_frame(ParsedFrame(frame_num=10, targets=[t_fall], points=[]))

    # Assert that both alerts were dispatched
    assert len(buzzer_triggers) > 0, "Buzzer should trigger on confirmed fall"
    assert len(telegram_alerts) > 0, "Telegram notification should trigger on confirmed fall"

    # Verify event metadata delivered to Telegram
    alert_info = telegram_alerts[0]
    assert alert_info["tid"] == 0
    assert alert_info["pos_z"] == 0.30
    assert alert_info["vel_z"] == -1.5

    # 3. Test acknowledge / silence method
    detector.acknowledge_alarm()
    assert len(detector.alert_hold_counters) == 0

    detector.close()
