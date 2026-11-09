# 🛡️ ElderGuard — mmWave Radar Fall Detection System

> Real-time fall detection for elderly care using Texas Instruments IWR6843 mmWave radar, edge AI, and IoT alerts.

---

## 📌 Overview

ElderGuard is a non-contact, privacy-preserving fall detection system that uses **60GHz millimeter-wave radar** to monitor a person's posture and instantly detect falls. No cameras. No wearables. Just radar.

When a fall is detected:
- 🚨 **ESP32 buzzer** sounds an alarm (2 seconds)
- 📱 **Telegram alert** is sent instantly
- 🌐 **Web dashboard** updates in real-time
- 🖥️ **Desktop GUI** shows live radar data and fall status

---

## 🧠 How It Works

```
TI IWR6843 Radar  →  Point Cloud Data  →  Height Analysis  →  Fall Decision
                                                   ↓
                                         ESP32 Buzzer (2s)
                                         Telegram Alert Bot
                                         Web Dashboard (port 5050)
```

**Fall is detected when:**
- Person's height drops to ≤ 0.68m (lying/fallen position)
- OR the delta height curve crosses the fall threshold (rapid drop)

---

## 🔧 Hardware

| Component | Details |
|---|---|
| **Radar Sensor** | Texas Instruments IWR6843ISK (60GHz mmWave) |
| **Microcontroller** | ESP32 (buzzer trigger via serial) |
| **Buzzer** | Active buzzer on ESP32 GPIO |
| **PC** | Windows 10/11, Python 3.13 |
| **Connectivity** | USB (radar), USB (ESP32), WiFi (dashboard + Telegram) |

---

## 🚀 Features

- ✅ **Real-time fall detection** — 0.68m height threshold + delta height cross detection
- ✅ **Web dashboard** — live height chart, alert log, QR code for phone access
- ✅ **Telegram bots** — instant alerts + remote `/fall` `/upright` commands
- ✅ **ESP32 edge buzzer** — autonomous 2-second siren, no PC dependency
- ✅ **ML model** — CNN-LSTM trained on custom posture dataset (standing/sitting/lying/walking)
- ✅ **Privacy first** — no cameras, no images, radar only
- ✅ **Multi-target** — tracks multiple people simultaneously

---

## 📁 Project Structure

```
mmwavefalldetection/
├── src/
│   ├── gui/                    # PyQt5 desktop GUI + radar parser
│   ├── inference/              # Live fall detector (CNN-LSTM + height gate)
│   ├── alerts/                 # Telegram notifier + ESP32 buzzer
│   └── dashboard/              # Flask web dashboard (SSE, Chart.js)
├── models/
│   ├── kinematic_cnn_lstm.pt   # Trained PyTorch model
│   └── onnx/                   # ONNX exported models for inference
├── datasets/
│   └── my_posture_dataset/     # Custom recorded posture data (6 people)
├── scripts/
│   └── train_cnn_lstm.py       # Model training script
├── tests/                      # Unit + integration tests
└── run_gui.bat                 # One-click launcher (Windows)
```

---

## ⚙️ Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure `.env`
```env
TELEGRAM_BOT_TOKEN=your_alert_bot_token
TELEGRAM_CHAT_ID=your_chat_id
TELEGRAM_COMMAND_BOT_TOKEN=your_command_bot_token
TELEGRAM_COMMAND_CHAT_ID=your_command_chat_id
ESP32_PORT=COM3
ESP32_ENABLED=True
```

### 3. Run
```bash
python src/gui/ti_mmwave_fall_gui.py
```

Then open **http://localhost:5050** for the web dashboard.

---

## 📊 Web Dashboard

| Panel | Description |
|---|---|
| Status card | UPRIGHT (green) / FALL DETECTED (red, animated) |
| Height chart | Live rolling graph with 0.68m threshold line |
| Alert log | Timestamped history of every fall & recovery |
| System health | Radar / ESP32 / Telegram connection status |
| QR code | Scan to open dashboard on any phone on the same WiFi |

---

## 📱 Telegram Commands

| Command | Bot | Action |
|---|---|---|
| `/fall` | @ElderguardAlertBot | Force FALL state (demo/test) |
| `/upright` | @ElderguardAlertBot | Instantly reset to UPRIGHT |

---

## 🏆 Project Expo

Built for the **IEI Project Expo** as a demonstration of:
- Edge AI and sensor fusion
- Real-time IoT alerting
- Non-invasive elderly care technology

---

## 👨‍💻 Authors

- **Mohit Kumar** — mohittkumar204
- **Navaneeth MP** — navaneethmp007

---

## 📄 License

MIT License — free to use, modify, and share.
