/*
 * ==============================================================================
 * ESP32 Emergency Fall Detection Buzzer & Visual Alarm Firmware
 * Features:
 *   - Audible Buzzer Alert (Siren, Double Beep, Pulse, Continuous)
 *   - Direct Wi-Fi Telegram Alerts via HTTPS GET Requests
 *   - Auto-Off Safety Watchdog & Acknowledge/Silence Commands
 *   - Dual Control: USB Serial (115200) + Wireless Wi-Fi / UDP
 * ==============================================================================
 *
 * Hardware Wiring:
 *   - Buzzer Positive (+) -> GPIO 18 (or GPIO 25 / GPIO 4)
 *   - Buzzer Negative (-) -> GND
 *   - Visual Alarm LED    -> GPIO 2 (Onboard blue LED)
 *   - USB Data Cable      -> PC USB Port (115200 Baud)
 * ==============================================================================
 */

#include <Arduino.h>
#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <HTTPClient.h>

// ================= Pin Configuration =================
#define BUZZER_PIN 18             // Output pin connected to buzzer (+)
#define LED_PIN 2                 // Visual alarm LED (onboard blue LED on GPIO 2)
#define USE_PASSIVE_BUZZER false  // Set to true for passive buzzer (tone PWM); false for active buzzer
#define BUZZER_FREQ_HZ 2700       // Frequency in Hz for passive buzzer

// ================= Wi-Fi & Telegram Configuration =================
#define ENABLE_WIFI true          // Set to true to enable Wi-Fi & Telegram alerts

// Replace with your local 2.4 GHz Wi-Fi credentials
const char* WIFI_SSID = "YOUR_WIFI_SSID";
const char* WIFI_PASS = "YOUR_WIFI_PASSWORD";

// Telegram Bot Credentials (Configured)
const char* TELEGRAM_BOT_TOKEN = "8503249788:AAHcXWu47YXPunhjJ8rWP1dAuy8ZUr2o-4k";
const char* TELEGRAM_CHAT_ID   = "2112600021";

// ================= Alarm State & Timing =================
enum BeepPattern {
  PATTERN_SIREN,
  PATTERN_DOUBLE_BEEP,
  PATTERN_PULSE,
  PATTERN_CONTINUOUS
};

bool alarmActive = false;
BeepPattern activePattern = PATTERN_SIREN;
unsigned long lastAlarmTrigger = 0;
unsigned long autoOffDurationMs = 10000; // 10 seconds auto-shutoff

unsigned long lastBeepToggle = 0;
bool beepState = false;
int doubleBeepStep = 0;

// URL encode helper for HTTPS GET query parameters
String urlEncode(const String& str) {
  String encoded = "";
  char c;
  char code0;
  char code1;
  for (unsigned int i = 0; i < str.length(); i++) {
    c = str.charAt(i);
    if (isalnum(c)) {
      encoded += c;
    } else if (c == ' ') {
      encoded += "%20";
    } else if (c == '\n') {
      encoded += "%0A";
    } else {
      code1 = (c & 0xf) + '0';
      if ((c & 0xf) > 9) {
        code1 = (c & 0xf) - 10 + 'A';
      }
      c = (c >> 4) & 0xf;
      code0 = c + '0';
      if (c > 9) {
        code0 = c - 10 + 'A';
      }
      encoded += '%';
      encoded += code0;
      encoded += code1;
    }
  }
  return encoded;
}

// Send HTTPS GET request to Telegram Bot API
bool sendTelegramAlertHttps(const String& message) {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("[ESP32] Wi-Fi not connected. Cannot send Telegram alert.");
    return false;
  }

  Serial.println("[ESP32] Sending HTTPS GET alert to Telegram...");
  WiFiClientSecure client;
  client.setInsecure(); // Lightweight TLS without CA bundle overhead
  client.setTimeout(6000);

  HTTPClient https;
  String url = "https://api.telegram.org/bot" + String(TELEGRAM_BOT_TOKEN) +
               "/sendMessage?chat_id=" + String(TELEGRAM_CHAT_ID) +
               "&text=" + urlEncode(message) +
               "&parse_mode=Markdown";

  bool success = false;
  if (https.begin(client, url)) {
    int httpCode = https.GET();
    if (httpCode == HTTP_CODE_OK) {
      Serial.println("[ESP32] [OK] Telegram alert delivered successfully via HTTPS GET!");
      success = true;
    } else {
      Serial.printf("[ESP32] Telegram API returned code: %d\n", httpCode);
      if (httpCode < 0) {
        Serial.printf("[ESP32] HTTPS GET error: %s\n", https.errorToString(httpCode).c_str());
      }
    }
    https.end();
  } else {
    Serial.println("[ESP32] Failed to initialize HTTPS connection to Telegram.");
  }
  return success;
}

void triggerBuzzer(bool state) {
  digitalWrite(LED_PIN, state ? HIGH : LOW);
  if (USE_PASSIVE_BUZZER) {
    if (state) {
      tone(BUZZER_PIN, BUZZER_FREQ_HZ);
    } else {
      noTone(BUZZER_PIN);
    }
  } else {
    digitalWrite(BUZZER_PIN, state ? HIGH : LOW);
  }
}

void parseCommand(String cmd) {
  cmd.trim();
  cmd.toUpperCase();
  if (cmd.length() == 0) return;

  if (cmd.startsWith("FALL")) {
    alarmActive = true;
    lastAlarmTrigger = millis();
    lastBeepToggle = millis();
    beepState = false;
    doubleBeepStep = 0;

    int firstColon = cmd.indexOf(':');
    if (firstColon != -1) {
      int secondColon = cmd.indexOf(':', firstColon + 1);
      String patternStr = "";
      String durStr = "";
      if (secondColon != -1) {
        patternStr = cmd.substring(firstColon + 1, secondColon);
        durStr = cmd.substring(secondColon + 1);
      } else {
        patternStr = cmd.substring(firstColon + 1);
      }

      if (patternStr == "DOUBLE_BEEP") {
        activePattern = PATTERN_DOUBLE_BEEP;
      } else if (patternStr == "PULSE") {
        activePattern = PATTERN_PULSE;
      } else if (patternStr == "CONTINUOUS") {
        activePattern = PATTERN_CONTINUOUS;
      } else {
        activePattern = PATTERN_SIREN;
      }

      if (durStr.length() > 0) {
        long d = durStr.toInt();
        if (d > 500 && d <= 60000) {
          autoOffDurationMs = d;
        }
      }
    } else {
      activePattern = PATTERN_SIREN;
    }

    Serial.println("[ESP32] >>> FALL ALARM ACTIVATED! BUZZER ON <<<");

  } else if (cmd == "CLEAR" || cmd == "ACK") {
    alarmActive = false;
    triggerBuzzer(false);
    Serial.println("[ESP32] Alarm Cleared/Acknowledged. System Normal.");

  } else if (cmd == "TEST") {
    Serial.println("[ESP32] Playing test chirp...");
    triggerBuzzer(true);
    delay(150);
    triggerBuzzer(false);
    delay(80);
    triggerBuzzer(true);
    delay(150);
    triggerBuzzer(false);

  } else if (cmd == "TEST_TELEGRAM") {
    Serial.println("[ESP32] Test Telegram Command Received.");
    String testMsg = "*ESP32 Fall Detection Hardware Alert*\n"
                     "------------------------------------\n"
                     "Test Alert Sent via ESP32 Wi-Fi HTTPS GET!\n"
                     "Status: Operational\n"
                     "Buzzer Pin: GPIO 18\n"
                     "Sensor: IWR6843ISK-ODS";
    sendTelegramAlertHttps(testMsg);

  } else if (cmd == "PING") {
    Serial.println("PONG");

  } else if (cmd == "STATUS") {
    Serial.print("[ESP32] Status: ");
    Serial.print(alarmActive ? "ALARM_ACTIVE" : "IDLE");
    Serial.print(" | Pattern: ");
    switch (activePattern) {
      case PATTERN_SIREN: Serial.print("SIREN"); break;
      case PATTERN_DOUBLE_BEEP: Serial.print("DOUBLE_BEEP"); break;
      case PATTERN_PULSE: Serial.print("PULSE"); break;
      case PATTERN_CONTINUOUS: Serial.print("CONTINUOUS"); break;
    }
    Serial.print(" | AutoOff: ");
    Serial.print(autoOffDurationMs);
    Serial.print("ms | WiFi: ");
    Serial.println(WiFi.status() == WL_CONNECTED ? WiFi.localIP().toString() : "DISCONNECTED");
  }
}

void updateBuzzerPattern() {
  if (!alarmActive) return;

  unsigned long now = millis();

  // Safety auto-off watchdog
  if (now - lastAlarmTrigger > autoOffDurationMs) {
    alarmActive = false;
    triggerBuzzer(false);
    Serial.println("[ESP32] Alarm auto-timed out.");
    return;
  }

  // Non-blocking pattern generator
  switch (activePattern) {
    case PATTERN_SIREN: {
      unsigned long interval = beepState ? 120 : 80;
      if (now - lastBeepToggle >= interval) {
        lastBeepToggle = now;
        beepState = !beepState;
        triggerBuzzer(beepState);
      }
      break;
    }

    case PATTERN_DOUBLE_BEEP: {
      unsigned long stepInterval = 100;
      if (doubleBeepStep == 0) stepInterval = 100;
      else if (doubleBeepStep == 1) stepInterval = 80;
      else if (doubleBeepStep == 2) stepInterval = 100;
      else if (doubleBeepStep == 3) stepInterval = 600;

      if (now - lastBeepToggle >= stepInterval) {
        lastBeepToggle = now;
        doubleBeepStep = (doubleBeepStep + 1) % 4;
        bool state = (doubleBeepStep == 0 || doubleBeepStep == 2);
        triggerBuzzer(state);
      }
      break;
    }

    case PATTERN_PULSE: {
      if (now - lastBeepToggle >= 250) {
        lastBeepToggle = now;
        beepState = !beepState;
        triggerBuzzer(beepState);
      }
      break;
    }

    case PATTERN_CONTINUOUS: {
      triggerBuzzer(true);
      break;
    }
  }
}

void setup() {
  Serial.begin(115200);
  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(LED_PIN, OUTPUT);
  triggerBuzzer(false);

  // Power-on self-test chirp
  for (int i = 0; i < 2; i++) {
    triggerBuzzer(true);
    delay(70);
    triggerBuzzer(false);
    delay(70);
  }

  Serial.println("\n=======================================================");
  Serial.println("[ESP32] IWR6843 Fall Detection Alarm Module Ready");
  Serial.println("[ESP32] Listening on USB Serial (115200 Baud)");
  Serial.println("=======================================================");

#if ENABLE_WIFI
  if (String(WIFI_SSID) != "YOUR_WIFI_SSID") {
    WiFi.mode(WIFI_STA);
    WiFi.begin(WIFI_SSID, WIFI_PASS);
    Serial.print("[ESP32] Connecting to WiFi (");
    Serial.print(WIFI_SSID);
    Serial.print(")");
    int attempts = 0;
    while (WiFi.status() != WL_CONNECTED && attempts < 20) {
      delay(500);
      Serial.print(".");
      attempts++;
    }
    if (WiFi.status() == WL_CONNECTED) {
      Serial.println("\n[ESP32] WiFi Connected! IP: " + WiFi.localIP().toString());
    } else {
      Serial.println("\n[ESP32] WiFi connection timed out. Serial mode active.");
    }
  } else {
    Serial.println("[ESP32] Note: Set WIFI_SSID in sketch to enable ESP32 Wi-Fi Telegram alerts.");
  }
#endif
}

void loop() {
  if (Serial.available() > 0) {
    String cmd = Serial.readStringUntil('\n');
    parseCommand(cmd);
  }

  updateBuzzerPattern();
}
