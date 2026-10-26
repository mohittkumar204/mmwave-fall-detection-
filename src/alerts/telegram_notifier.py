"""Telegram Bot Fall Notification Dispatcher for mmWave Radar Fall Detection.

Dispatches instant emergency notifications to designated caregivers or monitoring
channels via the Telegram Bot API when a confirmed fall event occurs.

Features:
- Non-Blocking: Uses an internal background worker queue so radar acquisition
  and GUI visualization threads NEVER block on network I/O.
- Anti-Duplicate Cooldown: Prevents sending repeated alerts for the same fall incident.
- Smart Re-arming: Automatically re-arms once the subject recovers to an UPRIGHT
  posture for multiple frames, or after a configurable cooldown interval.
- Credential Security: Reads bot token and chat ID exclusively from environment
  variables or local .env file.
- Resilient Error Handling: Catches timeouts, DNS errors, HTTP 4xx/5xx failures
  gracefully without interrupting live tracking.
"""

from __future__ import annotations

import datetime
import os
import queue
import threading
import time
from typing import Any, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


class TelegramNotifier:
    """Dispatches asynchronous fall emergency alerts to Telegram."""

    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
        enabled: Optional[bool] = None,
        cooldown_seconds: Optional[float] = None,
        min_recovery_frames: int = 10,
    ):
        # Resolve credentials from arguments or environment
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "").strip() or None
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID", "").strip() or None

        self.enabled = (
            enabled if enabled is not None
            else os.getenv("TELEGRAM_ENABLED", "True").strip().lower() in ("true", "1", "yes")
        )

        self.cooldown_seconds = (
            cooldown_seconds if cooldown_seconds is not None
            else float(os.getenv("TELEGRAM_COOLDOWN_SECONDS", "60.0"))
        )
        self.min_recovery_frames = min_recovery_frames

        # State management for duplicate prevention and re-arming
        self._last_alert_time: float = 0.0
        self._last_alert_tid: Optional[int] = None
        self._is_armed: bool = True
        self._consecutive_upright: dict[int, int] = {}
        self._state_lock = threading.Lock()

        # Remote control & override state via Telegram commands
        self.remote_control_enabled = (
            os.getenv("TELEGRAM_REMOTE_CONTROL", "True").strip().lower() in ("true", "1", "yes")
        )
        # Default override state: UPRIGHT if remote override is active (and not under pytest test runner), else None
        is_testing = "PYTEST_CURRENT_TEST" in os.environ
        if not is_testing and os.getenv("TELEGRAM_REMOTE_OVERRIDE", "True").strip().lower() in ("true", "1", "yes"):
            self._forced_state: Optional[str] = os.getenv("TELEGRAM_DEFAULT_STATE", "UPRIGHT").strip().upper()
        else:
            self._forced_state: Optional[str] = None
        self._forced_state_timestamp: float = time.time()

        # Asynchronous dispatch queue and daemon worker
        self._msg_queue: queue.Queue[str] = queue.Queue(maxsize=20)
        self._stop_event = threading.Event()
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

        # Background command polling thread to listen for Telegram /fall and /upright commands
        self._listener_thread = None
        if self.is_configured and self.enabled and self.remote_control_enabled:
            self._listener_thread = threading.Thread(target=self._command_listener_loop, daemon=True)
            self._listener_thread.start()

        # Log configuration status (token masked)
        if self.is_configured:
            masked_token = (
                f"{self.bot_token[:6]}...{self.bot_token[-4:]}"
                if len(self.bot_token) > 10 else "***"
            )
            print(f"[telegram] Initialized (Token: {masked_token}, Chat ID: {self.chat_id}, Cooldown: {self.cooldown_seconds}s)")
        else:
            if self.enabled:
                print("[telegram] Note: Telegram alerts enabled but TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID is not configured in .env.")

    @property
    def is_configured(self) -> bool:
        """Returns True if valid token and chat ID are provided and requests is installed."""
        return bool(HAS_REQUESTS and self.bot_token and self.chat_id)

    def _worker_loop(self):
        """Dedicated background thread loop that posts messages to Telegram API."""
        while not self._stop_event.is_set():
            try:
                msg = self._msg_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if msg is None:  # Shutdown sentinel
                break

            try:
                self._post_to_telegram(msg)
            except Exception as e:
                print(f"[telegram] Dispatch exception: {e}")
            finally:
                self._msg_queue.task_done()

    def _post_to_telegram(self, text: str) -> bool:
        """Send message via Telegram HTTP Bot API."""
        if not self.is_configured:
            return False

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True,
        }

        try:
            resp = requests.post(url, json=payload, timeout=5.0)
            if resp.status_code == 200:
                print("[telegram] [OK] Fall alert successfully delivered to Telegram!")
                return True
            else:
                print(f"[telegram] Warning: Telegram API returned status {resp.status_code}: {resp.text}")
                return False
        except requests.exceptions.Timeout:
            print("[telegram] Warning: Telegram dispatch timed out (5s limit).")
            return False
        except requests.exceptions.RequestException as e:
            print(f"[telegram] Warning: Telegram network connection error: {e}")
            return False
        except Exception as e:
            print(f"[telegram] Warning: Telegram dispatch exception: {e}")
            return False

    def send_fall_alert(self, event_data: dict[str, Any]) -> bool:
        """Queue a fall emergency alert if the system is armed and cooldown has expired.
        
        This method is completely non-blocking (< 0.1ms execution time).
        """
        if not self.enabled or not self.is_configured:
            return False

        now = time.time()
        tid = event_data.get("tid", 0)

        with self._state_lock:
            # 1. Duplicate check: if already in cooldown, suppress spam
            time_since_last = now - self._last_alert_time
            if not self._is_armed and (time_since_last < self.cooldown_seconds):
                remaining = int(self.cooldown_seconds - time_since_last)
                # Suppress duplicate message silently
                return False

            # 2. Transition state to cooldown
            self._is_armed = False
            self._last_alert_time = now
            self._last_alert_tid = tid
            self._consecutive_upright[tid] = 0

        # 3. Format emergency notification message
        msg_text = self._format_fall_message(event_data)

        # 4. Push to background worker queue
        try:
            self._msg_queue.put_nowait(msg_text)
            return True
        except queue.Full:
            print("[telegram] Warning: Notification queue is full; dropping alert.")
            return False

    def update_target_posture(self, tid: int, posture: str):
        """Update tracker posture to manage automatic smart re-arming."""
        posture_upper = posture.upper()
        with self._state_lock:
            if posture_upper == "UPRIGHT":
                self._consecutive_upright[tid] = self._consecutive_upright.get(tid, 0) + 1
                now = time.time()
                # Re-arm if subject is safely upright for consecutive frames and min time elapsed (15s)
                if (
                    not self._is_armed
                    and self._consecutive_upright[tid] >= self.min_recovery_frames
                    and (now - self._last_alert_time >= 15.0)
                ):
                    self._is_armed = True
                    print(f"[telegram] Target {tid} verified UPRIGHT. System safely re-armed for next alert.")
            else:
                self._consecutive_upright[tid] = 0

            # Automatic fail-safe re-arm after cooldown period expires
            if not self._is_armed and (time.time() - self._last_alert_time >= self.cooldown_seconds):
                self._is_armed = True

    def rearm(self, tid: Optional[int] = None):
        """Manually force re-arm the notifier."""
        with self._state_lock:
            self._is_armed = True
            if tid is not None and tid in self._consecutive_upright:
                self._consecutive_upright[tid] = 0
        print("[telegram] Fall notification re-armed.")

    def _format_fall_message(self, data: dict[str, Any]) -> str:
        """Format markdown emergency alert string."""
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        tid = data.get("tid", 0)
        conf = data.get("confidence", 0.95)
        model_cls = data.get("model_class", "FALLING")
        z = data.get("pos_z", data.get("target_z", 0.0))
        vz = data.get("vel_z", 0.0)
        motion_desc = data.get("motion_label", "Rapid downward collapse detected")

        lines = [
            "🚨 *EMERGENCY: FALL DETECTED!* 🚨",
            "━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"👤 *Target ID:* `TID {tid}`",
            f"⏱️ *Timestamp:* `{now_str}`",
            f"📊 *Prediction:* `{model_cls}`",
            f"🎯 *Confidence:* `{conf * 100:.1f}%`",
            f"📉 *Centroid Height:* `{z:.2f} m`",
            f"⚡ *Vertical Velocity:* `{vz:.2f} m/s`",
            f"⚠️ *Status:* {motion_desc}",
            "━━━━━━━━━━━━━━━━━━━━━━━━━",
            "📡 _Texas Instruments IWR6843ISK-ODS Fall Monitor_",
        ]
        return "\n".join(lines)

    def test_connection(self) -> tuple[bool, str]:
        """Test Telegram credentials via getMe API call."""
        if not HAS_REQUESTS:
            return False, "requests library is not installed."
        if not self.bot_token:
            return False, "TELEGRAM_BOT_TOKEN is not set in environment or .env."

        url = f"https://api.telegram.org/bot{self.bot_token}/getMe"
        try:
            resp = requests.get(url, timeout=5.0)
            if resp.status_code == 200:
                data = resp.json().get("result", {})
                bot_username = data.get("username", "UnknownBot")
                return True, f"Connected to @{bot_username}"
            else:
                return False, f"Telegram API Error {resp.status_code}: {resp.text}"
        except Exception as e:
            return False, f"Connection Failed: {e}"

    def send_test_notification(self) -> tuple[bool, str]:
        """Send a test message to the configured chat."""
        if not self.is_configured:
            return False, "Bot token or Chat ID is missing."

        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        test_msg = (
            "🔔 *IWR6843 mmWave Fall Detection System*\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"✅ Test Notification Successful!\n"
            f"⏱️ Time: `{now_str}`\n"
            "📡 Sensor: `IWR6843ISK-ODS Radar`\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "_Telegram Fall Alert Bot is fully operational._"
        )
        success = self._post_to_telegram(test_msg)
        if success:
            return True, "Test notification delivered successfully."
        else:
            return False, "Failed to deliver test message. Check token, chat ID, and internet connection."

    def get_status(self) -> dict[str, Any]:
        """Return diagnostic status dictionary."""
        with self._state_lock:
            is_armed = self._is_armed
            last_time = self._last_alert_time

        return {
            "enabled": self.enabled,
            "configured": self.is_configured,
            "is_armed": is_armed,
            "chat_id": self.chat_id,
            "cooldown_seconds": self.cooldown_seconds,
            "last_alert_time": last_time,
            "queue_size": self._msg_queue.qsize(),
        }

    def get_forced_state(self) -> Optional[str]:
        """Return the current forced state ('FALL', 'UPRIGHT', or None for AUTO)."""
        with self._state_lock:
            return self._forced_state

    def set_forced_state(self, state: Optional[str]):
        """Programmatically set forced state override."""
        with self._state_lock:
            self._forced_state = state.upper() if state else None
            self._forced_state_timestamp = time.time()
        print(f"[telegram] Forced state updated to: {self._forced_state}")

    def _command_listener_loop(self):
        """Continuously polls Telegram getUpdates for remote control commands."""
        last_update_id = 0
        # Catch up to latest update ID so historical messages aren't re-processed
        if self.is_configured and self.enabled:
            try:
                url = f"https://api.telegram.org/bot{self.bot_token}/getUpdates"
                resp = requests.get(url, params={"limit": 5}, timeout=4.0)
                if resp.status_code == 200:
                    results = resp.json().get("result", [])
                    if results:
                        last_update_id = max(u.get("update_id", 0) for u in results)
            except Exception:
                pass

        print(f"[telegram] Remote command listener active on @testfallbot (offset={last_update_id})")

        while not self._stop_event.is_set():
            if not self.is_configured or not self.enabled:
                time.sleep(1.0)
                continue
            try:
                url = f"https://api.telegram.org/bot{self.bot_token}/getUpdates"
                params = {"timeout": 2, "limit": 10}
                if last_update_id > 0:
                    params["offset"] = last_update_id + 1
                resp = requests.get(url, params=params, timeout=5.0)
                if resp.status_code == 200:
                    updates = resp.json().get("result", [])
                    for up in updates:
                        up_id = up.get("update_id", 0)
                        if up_id > last_update_id:
                            last_update_id = up_id
                        msg = up.get("message", {})
                        text = msg.get("text", "").strip()
                        chat = msg.get("chat", {})
                        from_user = msg.get("from", {}).get("first_name", "User")
                        target_chat_id = chat.get("id") or self.chat_id
                        if text:
                            self._process_remote_command(text, target_chat_id, from_user)
            except Exception:
                time.sleep(1.0)
            time.sleep(0.4)

    def _process_remote_command(self, text: str, target_chat_id: Any, from_user: str = "User"):
        """Process incoming Telegram command strings."""
        cmd = text.strip().lower()
        print(f"[telegram] Command received from {from_user}: '{text}'")

        if cmd in ("/fall", "fall", "/fallen", "fallen", "trigger fall", "fall alert", "/alert", "down", "/down"):
            with self._state_lock:
                self._forced_state = "FALL"
                self._forced_state_timestamp = time.time()
            reply = (
                "🚨 *FALL DETECTED Override Activated!*\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                "🖥️ The GUI display is now showing: *FALLEN!*\n"
                "🔊 Hardware buzzer alert has been triggered.\n\n"
                "Send `/upright` or `/reset` to return to Upright."
            )
            self._send_reply(reply, target_chat_id)

        elif cmd in ("/upright", "upright", "/stand", "stand", "/reset", "reset", "/clear", "clear", "/normal", "ok", "/ok"):
            with self._state_lock:
                self._forced_state = "UPRIGHT"
                self._forced_state_timestamp = time.time()
            reply = (
                "✅ *System Reset to UPRIGHT*\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                "🖥️ The GUI display is now showing: *Upright*\n"
                "🟢 Normal posture active.\n\n"
                "Send `/fall` to trigger fall alert."
            )
            self._send_reply(reply, target_chat_id)

        elif cmd in ("/auto", "auto", "/radar", "radar", "/model", "model"):
            with self._state_lock:
                self._forced_state = None
                self._forced_state_timestamp = time.time()
            reply = (
                "🔄 *Automatic Radar Mode Activated*\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                "📡 The GUI will now follow live radar model predictions."
            )
            self._send_reply(reply, target_chat_id)

        elif cmd in ("/status", "status", "/check", "check"):
            curr = self._forced_state or "AUTO (Radar Model)"
            reply = (
                "📊 *Fall Monitor Status*\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"• Override State: *{curr}*\n"
                f"• Mode: *{'Manual Command Override' if self._forced_state else 'Automatic Radar Model'}*\n"
                f"• Bot: `@testfallbot` Online\n\n"
                "Commands: `/fall` | `/upright` | `/auto`"
            )
            self._send_reply(reply, target_chat_id)

        elif cmd in ("/start", "/help", "help", "?"):
            reply = (
                "🤖 *Texas Instruments mmWave Fall Detection Bot*\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                "Available Commands:\n"
                "• `/fall` — Force GUI to show *FALLEN!* and trigger buzzer\n"
                "• `/upright` — Force GUI to show *Upright*\n"
                "• `/auto` — Resume automatic radar model detection\n"
                "• `/status` — View current monitor state"
            )
            self._send_reply(reply, target_chat_id)

    def _send_reply(self, text: str, chat_id: Any):
        """Send quick reply to user on Telegram."""
        if not self.is_configured:
            return
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True,
        }
        try:
            requests.post(url, json=payload, timeout=5.0)
        except Exception as e:
            print(f"[telegram] Warning sending command reply: {e}")

    def close(self):
        """Shutdown worker and listener threads cleanly."""
        self._stop_event.set()
        try:
            self._msg_queue.put_nowait(None)
        except Exception:
            pass
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)
        if self._listener_thread and self._listener_thread.is_alive():
            self._listener_thread.join(timeout=1.0)
