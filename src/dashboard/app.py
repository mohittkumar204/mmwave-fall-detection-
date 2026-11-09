"""ElderGuard Flask Dashboard Server.

Runs in a background daemon thread alongside the PyQt5 GUI.
Serves a live web dashboard at http://localhost:5050 (and local network IP).
Uses Server-Sent Events (SSE) for real-time push — no WebSocket library needed.
"""
from __future__ import annotations

import base64
import io
import json
import os
import socket
import threading
import time

from flask import Flask, Response, jsonify, render_template_string

from src.dashboard.state import get_alerts, get_state

# ── Flask app ────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["SECRET_KEY"] = "elderguard-2025"

_LOCAL_IP: str = "127.0.0.1"
_PORT: int = 5050


def _get_local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def _make_qr_base64(url: str) -> str:
    """Generate a QR code PNG and return it as a base64 data URI."""
    try:
        import qrcode
        qr = qrcode.QRCode(box_size=6, border=2)
        qr.add_data(url)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return ""


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    url = f"http://{_LOCAL_IP}:{_PORT}"
    qr_src = _make_qr_base64(url)
    return render_template_string(_HTML_TEMPLATE, qr_src=qr_src, url=url)


@app.route("/api/status")
def api_status():
    return jsonify(get_state())


@app.route("/api/alerts")
def api_alerts():
    return jsonify(get_alerts())


@app.route("/stream")
def stream():
    """Server-Sent Events endpoint — browser receives a push every 400 ms."""
    def generate():
        while True:
            payload = json.dumps(get_state())
            yield f"data: {payload}\n\n"
            time.sleep(0.4)

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ── HTML Template ─────────────────────────────────────────────────────────────
_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>ElderGuard — Fall Detection</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" />
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css" />
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <style>
    :root {
      --green: #16a34a;
      --red:   #dc2626;
      --bg:    #0d1117;
      --card:  #161b22;
      --border:#30363d;
      --text:  #e6edf3;
      --muted: #8b949e;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { background: var(--bg); color: var(--text); font-family: 'Segoe UI', system-ui, sans-serif; min-height: 100vh; }

    /* Navbar */
    .navbar { background: var(--card); border-bottom: 1px solid var(--border); padding: 0.75rem 1.5rem; display: flex; align-items: center; justify-content: space-between; }
    .navbar .brand { display: flex; align-items: center; gap: 0.6rem; font-size: 1.05rem; font-weight: 600; letter-spacing: 0.02em; }
    .navbar .brand i { font-size: 1.25rem; color: var(--green); }
    .live-pill { display: flex; align-items: center; gap: 0.4rem; font-size: 0.78rem; color: var(--muted); }
    .live-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--green); animation: pulse 1.8s infinite; }
    @keyframes pulse { 0%,100% { opacity:1; } 50% { opacity:0.4; } }

    /* Grid */
    .main { padding: 1.25rem 1.5rem; display: grid; grid-template-columns: 1fr 1fr 1fr; grid-template-rows: auto auto auto; gap: 1rem; }

    /* Card */
    .card { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 1.1rem 1.25rem; }
    .card-label { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); margin-bottom: 0.4rem; }

    /* Status card */
    .status-card { grid-column: span 2; display: flex; align-items: center; gap: 1.5rem; }
    .status-icon { font-size: 3.5rem; line-height: 1; }
    .status-text .badge-status { display: inline-block; padding: 0.3rem 0.9rem; border-radius: 4px; font-size: 1.05rem; font-weight: 700; letter-spacing: 0.04em; }
    .status-text .sub { font-size: 0.82rem; color: var(--muted); margin-top: 0.35rem; }
    .status-upright .status-icon { color: var(--green); }
    .status-upright .badge-status { background: rgba(22,163,74,0.15); color: var(--green); border: 1px solid rgba(22,163,74,0.4); }
    .status-fall .status-icon { color: var(--red); animation: shake 0.4s infinite; }
    .status-fall .badge-status { background: rgba(220,38,38,0.15); color: var(--red); border: 1px solid rgba(220,38,38,0.4); }
    @keyframes shake { 0%,100% { transform:translateX(0); } 25% { transform:translateX(-3px); } 75% { transform:translateX(3px); } }
    .status-unknown .status-icon { color: var(--muted); }
    .status-unknown .badge-status { background: rgba(139,148,158,0.12); color: var(--muted); border: 1px solid var(--border); }

    /* QR card */
    .qr-card { display: flex; flex-direction: column; align-items: center; justify-content: center; }
    .qr-card img { width: 90px; height: 90px; border-radius: 4px; margin-bottom: 0.4rem; }
    .qr-url { font-size: 0.7rem; color: var(--muted); word-break: break-all; text-align: center; }

    /* Stats row */
    .stat-value { font-size: 1.65rem; font-weight: 700; line-height: 1.15; }
    .stat-sub    { font-size: 0.75rem; color: var(--muted); margin-top: 0.2rem; }
    .stat-icon   { font-size: 1.5rem; float: right; opacity: 0.25; margin-top: -0.25rem; }

    /* Chart */
    .chart-card { grid-column: span 2; }
    canvas { max-height: 200px; }

    /* Alerts table */
    .alerts-card { grid-column: span 1; overflow: hidden; }
    .alerts-card .card-label { margin-bottom: 0.75rem; }
    .alert-table { width: 100%; border-collapse: collapse; font-size: 0.78rem; }
    .alert-table th { color: var(--muted); font-weight: 500; text-align: left; padding: 0 0 0.5rem; border-bottom: 1px solid var(--border); }
    .alert-table td { padding: 0.45rem 0; border-bottom: 1px solid rgba(48,54,61,0.5); vertical-align: middle; }
    .badge-fall   { background: rgba(220,38,38,0.15); color: var(--red); border: 1px solid rgba(220,38,38,0.35); padding: 0.1rem 0.45rem; border-radius: 3px; font-size: 0.7rem; font-weight: 600; }
    .badge-upright{ background: rgba(22,163,74,0.12); color: var(--green); border: 1px solid rgba(22,163,74,0.3); padding: 0.1rem 0.45rem; border-radius: 3px; font-size: 0.7rem; }

    /* System status */
    .sys-card { grid-column: span 3; }
    .sys-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 0.75rem; }
    .sys-item { display: flex; align-items: center; gap: 0.6rem; }
    .sys-dot  { width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0; }
    .dot-ok   { background: var(--green); }
    .dot-off  { background: var(--muted); }
    .dot-err  { background: var(--red); }
    .sys-label{ font-size: 0.78rem; color: var(--muted); }
    .sys-val  { font-size: 0.78rem; font-weight: 500; }
  </style>
</head>
<body>

<!-- Navbar -->
<nav class="navbar">
  <div class="brand">
    <i class="bi bi-shield-check"></i>
    ElderGuard &mdash; Fall Detection System
  </div>
  <div class="live-pill">
    <span class="live-dot" id="liveDot"></span>
    <span id="liveLabel">Connecting…</span>
  </div>
</nav>

<!-- Main grid -->
<div class="main">

  <!-- Status Card -->
  <div class="card status-card status-unknown" id="statusCard">
    <div class="status-icon" id="statusIcon"><i class="bi bi-person-standing"></i></div>
    <div class="status-text">
      <div class="card-label">Person Status</div>
      <span class="badge-status" id="statusBadge">Waiting for data…</span>
      <div class="sub" id="statusSub">Height: — &nbsp;|&nbsp; TID: — &nbsp;|&nbsp; Confidence: —</div>
    </div>
  </div>

  <!-- QR Card -->
  <div class="card qr-card">
    <div class="card-label" style="text-align:center">Scan to open on phone</div>
    {% if qr_src %}
    <img src="{{ qr_src }}" alt="QR Code" />
    {% else %}
    <div style="font-size:0.75rem;color:var(--muted);padding:1rem;text-align:center">Install qrcode<br>to show QR</div>
    {% endif %}
    <div class="qr-url">{{ url }}</div>
  </div>

  <!-- Falls Today -->
  <div class="card">
    <i class="bi bi-bell-fill stat-icon" style="color:#f97316"></i>
    <div class="card-label">Falls Today</div>
    <div class="stat-value" id="statFalls">—</div>
    <div class="stat-sub">Total detected events</div>
  </div>

  <!-- Avg Response -->
  <div class="card">
    <i class="bi bi-stopwatch stat-icon" style="color:#38bdf8"></i>
    <div class="card-label">Avg Response</div>
    <div class="stat-value" id="statResponse">—</div>
    <div class="stat-sub">Model inference time</div>
  </div>

  <!-- Uptime -->
  <div class="card">
    <i class="bi bi-clock-history stat-icon" style="color:#a78bfa"></i>
    <div class="card-label">System Uptime</div>
    <div class="stat-value" id="statUptime">—</div>
    <div class="stat-sub">Since GUI launched</div>
  </div>

  <!-- Height chart -->
  <div class="card chart-card">
    <div class="card-label" style="margin-bottom:0.75rem">Live Height (metres)</div>
    <canvas id="heightChart"></canvas>
  </div>

  <!-- Alert log -->
  <div class="card alerts-card">
    <div class="card-label">Alert History</div>
    <table class="alert-table">
      <thead><tr><th>Time</th><th>TID</th><th>Height</th><th>Status</th></tr></thead>
      <tbody id="alertBody"></tbody>
    </table>
  </div>

  <!-- System status -->
  <div class="card sys-card">
    <div class="card-label" style="margin-bottom:0.75rem">System Health</div>
    <div class="sys-grid">
      <div class="sys-item">
        <span class="sys-dot" id="dotRadar"></span>
        <div>
          <div class="sys-val">mmWave Radar</div>
          <div class="sys-label" id="lblRadar">Checking…</div>
        </div>
      </div>
      <div class="sys-item">
        <span class="sys-dot" id="dotEsp"></span>
        <div>
          <div class="sys-val">ESP32 Buzzer</div>
          <div class="sys-label" id="lblEsp">Checking…</div>
        </div>
      </div>
      <div class="sys-item">
        <span class="sys-dot" id="dotTg"></span>
        <div>
          <div class="sys-val">Telegram Bot</div>
          <div class="sys-label" id="lblTg">Checking…</div>
        </div>
      </div>
      <div class="sys-item">
        <span class="sys-dot dot-ok" id="dotDash"></span>
        <div>
          <div class="sys-val">Dashboard</div>
          <div class="sys-label">Online &mdash; port {{ url.split(':')[-1] }}</div>
        </div>
      </div>
    </div>
  </div>

</div><!-- /.main -->

<script>
// ── Chart setup ──────────────────────────────────────────────────────────────
const MAX_POINTS = 80;
const heights = [];
const labels  = [];

const ctx = document.getElementById('heightChart').getContext('2d');
const chart = new Chart(ctx, {
  type: 'line',
  data: {
    labels,
    datasets: [{
      label: 'Height (m)',
      data: heights,
      borderColor: '#38bdf8',
      backgroundColor: 'rgba(56,189,248,0.07)',
      borderWidth: 1.5,
      pointRadius: 0,
      fill: true,
      tension: 0.35,
    },{
      label: 'Fall threshold (0.68 m)',
      data: new Array(MAX_POINTS).fill(0.68),
      borderColor: 'rgba(220,38,38,0.55)',
      borderWidth: 1,
      borderDash: [6,4],
      pointRadius: 0,
      fill: false,
    }]
  },
  options: {
    animation: false,
    responsive: true,
    maintainAspectRatio: true,
    plugins: { legend: { labels: { color: '#8b949e', boxWidth: 14, font: { size: 11 } } } },
    scales: {
      x: { display: false },
      y: {
        min: 0, max: 2.2,
        grid: { color: 'rgba(48,54,61,0.6)' },
        ticks: { color: '#8b949e', font: { size: 10 }, stepSize: 0.5 },
      }
    }
  }
});

function pushHeight(h) {
  const t = new Date().toLocaleTimeString('en-GB', { hour12: false });
  heights.push(h);
  labels.push(t);
  if (heights.length > MAX_POINTS) { heights.shift(); labels.shift(); }
  chart.options.data = chart.data;
  chart.update('none');
}

// ── SSE listener ─────────────────────────────────────────────────────────────
const evtSrc = new EventSource('/stream');
let lastUpdate = Date.now();

evtSrc.onmessage = function(e) {
  lastUpdate = Date.now();
  const d = JSON.parse(e.data);
  updateStatus(d);
  updateStats(d);
  updateHealth(d);
  pushHeight(d.height || 0);
  document.getElementById('liveLabel').textContent = 'Live';
  document.getElementById('liveDot').style.background = '#16a34a';
};

evtSrc.onerror = function() {
  document.getElementById('liveLabel').textContent = 'Reconnecting…';
  document.getElementById('liveDot').style.background = '#f97316';
};

// ── Status card ───────────────────────────────────────────────────────────────
function updateStatus(d) {
  const card  = document.getElementById('statusCard');
  const icon  = document.getElementById('statusIcon');
  const badge = document.getElementById('statusBadge');
  const sub   = document.getElementById('statusSub');

  card.className = 'card status-card ';
  if (d.status === 'FALL') {
    card.className += 'status-fall';
    icon.innerHTML  = '<i class="bi bi-exclamation-triangle-fill"></i>';
    badge.textContent = '⚠ FALL DETECTED';
  } else if (d.status === 'UPRIGHT') {
    card.className += 'status-upright';
    icon.innerHTML  = '<i class="bi bi-person-standing"></i>';
    badge.textContent = 'UPRIGHT';
  } else {
    card.className += 'status-unknown';
    icon.innerHTML  = '<i class="bi bi-person"></i>';
    badge.textContent = 'Waiting for data…';
  }

  const conf = d.confidence ? (d.confidence * 100).toFixed(1) + '%' : '—';
  const ht   = d.height ? d.height.toFixed(2) + ' m' : '—';
  sub.innerHTML = `Height: <strong>${ht}</strong> &nbsp;|&nbsp; TID: ${d.tid ?? '—'} &nbsp;|&nbsp; Confidence: ${conf}`;
}

// ── Stats ─────────────────────────────────────────────────────────────────────
function updateStats(d) {
  document.getElementById('statFalls').textContent    = d.fall_count ?? '—';
  document.getElementById('statResponse').textContent = d.avg_response_ms ? d.avg_response_ms.toFixed(1) + ' ms' : '—';
  document.getElementById('statUptime').textContent   = d.uptime ?? '—';
}

// ── System health ─────────────────────────────────────────────────────────────
function dot(id, ok) {
  document.getElementById(id).className = 'sys-dot ' + (ok ? 'dot-ok' : 'dot-off');
}
function lbl(id, text) { document.getElementById(id).textContent = text; }

function updateHealth(d) {
  dot('dotRadar', d.radar_connected);
  lbl('lblRadar', d.radar_connected ? 'Connected' : 'Not connected');
  dot('dotEsp',   d.esp32_connected);
  lbl('lblEsp',   d.esp32_connected ? 'Connected' : 'Not connected');
  dot('dotTg',    d.telegram_connected);
  lbl('lblTg',    d.telegram_connected ? 'Online' : 'Offline');
}

// ── Alert log (poll separately at lower rate) ──────────────────────────────────
function refreshAlerts() {
  fetch('/api/alerts')
    .then(r => r.json())
    .then(rows => {
      const tbody = document.getElementById('alertBody');
      tbody.innerHTML = rows.slice(0, 12).map(r => `
        <tr>
          <td>${r.timestamp}</td>
          <td>${r.tid}</td>
          <td>${r.height} m</td>
          <td><span class="${r.status === 'FALL' ? 'badge-fall' : 'badge-upright'}">${r.status}</span></td>
        </tr>`).join('');
    }).catch(() => {});
}
refreshAlerts();
setInterval(refreshAlerts, 3000);
</script>
</body>
</html>"""


# ── Server launcher ───────────────────────────────────────────────────────────

def start_dashboard(port: int = 5050) -> None:
    """Start the Flask dashboard in a background daemon thread.
    
    Call once from the GUI's __init__. The thread is a daemon so it exits
    automatically when the main (GUI) process exits.
    """
    global _LOCAL_IP, _PORT
    _LOCAL_IP = _get_local_ip()
    _PORT = port

    import logging
    log = logging.getLogger("werkzeug")
    log.setLevel(logging.ERROR)  # Silence Flask request logs in the GUI console

    def _run():
        app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False, threaded=True)

    t = threading.Thread(target=_run, name="dashboard-server", daemon=True)
    t.start()
    print(f"[dashboard] ElderGuard dashboard running at http://{_LOCAL_IP}:{port}")
