from flask import Flask, jsonify, render_template, request
import random
import time
import uuid
from datetime import datetime, timedelta
from collections import defaultdict, deque
import numpy as np
import joblib
import json
import os

model = joblib.load("model.pkl")
anomaly_model = joblib.load("anomaly_model.pkl")
app = Flask(__name__)

SESSIONS_FILE = "sessions_store.json"
AUDIT_LOG_FILE = "audit_log.json"

# Empirical score_samples() range from the training set, used to normalize
# IsolationForest's raw output into a 0-100 "anomaly score" (higher = more
# anomalous). Recalculate these if the model is retrained on new data.
ANOMALY_RAW_MIN = -0.6215549883977712
ANOMALY_RAW_MAX = -0.3824187043857521

# --- Simulated session store ---
SESSIONS = {}

# --- IP velocity tracking (for credential-stuffing / scalping detection) ---
# Maps ip_address -> deque of datetime objects for recent sessions seen.
IP_ACTIVITY = defaultdict(deque)
IP_WINDOW_MINUTES = 5
IP_VELOCITY_THRESHOLD = 3  # >= this many sessions from one IP in the window

# --- Audit trail of every mitigation decision made ---
AUDIT_LOG = deque(maxlen=200)

# Small pools of simulated source IPs. Bots draw from a much smaller pool so
# the same handful of addresses show up repeatedly (mimicking a botnet /
# credential-stuffing burst); humans get a wide spread of addresses.
BOT_IP_POOL = [f"185.220.101.{n}" for n in (4, 7, 12, 19)]

def random_human_ip():
    return f"{random.randint(1,223)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"

def generate_session(force_bot=None):
    """Generate a simulated user session with behavioral signals."""
    session_id = str(uuid.uuid4())[:8].upper()
    is_bot = force_bot if force_bot is not None else random.random() < 0.35

    if is_bot:
        # Bot behavioral patterns: too fast, too uniform, suspicious timing
        session_duration = round(random.uniform(0.5, 8.0), 2)
        action_count = random.randint(80, 300)
        avg_click_interval_ms = round(random.uniform(50, 250), 1)
        scroll_variance = round(random.uniform(0.01, 0.08), 3)
        skip_rate = round(random.uniform(0.7, 1.0), 2)
        unique_ips = random.randint(1, 2)
        captcha_solved = random.choice([True, False])
        browser_headless = random.choice([True, True, False])
        mouse_movement_score = round(random.uniform(0.0, 0.25), 2)
        typing_rhythm_variance = round(random.uniform(0.0, 0.05), 3)
        threat_type = random.choice([
            "OAT-016 Skewing", "OAT-005 Scalping",
            "OAT-013 Credential Stuffing", "Synthetic Identity"
        ])
    else:
        # Human behavioral patterns: natural variance
        session_duration = round(random.uniform(5.0, 45.0), 2)
        action_count = random.randint(5, 60)
        avg_click_interval_ms = round(random.uniform(400, 3000), 1)
        scroll_variance = round(random.uniform(0.2, 0.9), 3)
        skip_rate = round(random.uniform(0.05, 0.45), 2)
        unique_ips = 1
        captcha_solved = True
        browser_headless = False
        mouse_movement_score = round(random.uniform(0.6, 1.0), 2)
        typing_rhythm_variance = round(random.uniform(0.15, 0.65), 3)
        threat_type = None

    ip_address = random.choice(BOT_IP_POOL) if is_bot and random.random() < 0.6 else random_human_ip()

    # Compute Bot Probability Score (0-100)
    score = compute_bot_score(
        session_duration, action_count, avg_click_interval_ms,
        scroll_variance, skip_rate, browser_headless,
        mouse_movement_score, typing_rhythm_variance
    )

    # Add some noise to make it realistic
    score = min(100, max(0, score + random.randint(-5, 5)))

    anomaly_features = np.array([[
        session_duration, action_count, avg_click_interval_ms,
        scroll_variance, skip_rate, mouse_movement_score,
        typing_rhythm_variance, int(browser_headless)
    ]])
    anomaly_score, is_anomaly = compute_anomaly_score(anomaly_features)
    needs_review = is_anomaly and score < 35

    ip_recent_count, is_velocity_attack = record_ip_activity(ip_address)
    mitigation_action, mitigation_reason = decide_mitigation(score, is_velocity_attack, needs_review)

    platform = random.choice([
        "Spotify", "BookMyShow", "YouTube Music",
        "Ticketmaster", "JioSaavn", "Apple Music"
    ])

    now = datetime.now()
    ts = now - timedelta(seconds=random.randint(0, 300))

    session = {
        "session_id": session_id,
        "timestamp": ts.strftime("%H:%M:%S"),
        "platform": platform,
        "is_bot": is_bot,
        "bot_score": score,
        "risk_level": risk_label(score),
        "signals": {
            "session_duration_min": session_duration,
            "action_count": action_count,
            "avg_click_interval_ms": avg_click_interval_ms,
            "scroll_variance": scroll_variance,
            "skip_rate": skip_rate,
            "unique_ips": unique_ips,
            "captcha_solved": captcha_solved,
            "browser_headless": browser_headless,
            "mouse_movement_score": mouse_movement_score,
            "typing_rhythm_variance": typing_rhythm_variance,
        },
        "threat_type": threat_type,
        "flagged": score >= 65,
        "ip_address": ip_address,
        "ip_recent_count": ip_recent_count,
        "is_velocity_attack": is_velocity_attack,
        "anomaly_score": anomaly_score,
        "needs_review": needs_review,
        "mitigation_action": mitigation_action,
        "mitigation_reason": mitigation_reason,
    }

    SESSIONS[session_id] = session
    log_audit_entry(session)
    return session

FEATURE_COLUMNS = [
    "session_duration_min", "action_count", "click_interval_ms",
    "scroll_variance", "skip_rate", "mouse_movement_score",
    "typing_rhythm_variance", "browser_headless"
]

def compute_bot_score(session_duration, action_count, click_interval,
                      scroll_var, skip_rate, headless,
                      mouse_score, typing_var):
    features = np.array([[
        session_duration,
        action_count,
        click_interval,
        scroll_var,
        skip_rate,
        mouse_score,
        typing_var,
        int(headless)
    ]])
    proba = model.predict_proba(features)[0]
    bot_probability = proba[1]
    return round(bot_probability * 100)

def compute_anomaly_score(feature_row):
    """Run the unsupervised IsolationForest cross-check.

    Returns (anomaly_score 0-100, is_anomaly bool). This model never sees
    the is_bot label during training, so it acts as an independent second
    opinion on the supervised RandomForest score.
    """
    raw = anomaly_model.score_samples(feature_row)[0]
    normalized = (ANOMALY_RAW_MAX - raw) / (ANOMALY_RAW_MAX - ANOMALY_RAW_MIN) * 100
    anomaly_score = round(min(100, max(0, normalized)))
    is_anomaly = anomaly_model.predict(feature_row)[0] == -1
    return anomaly_score, bool(is_anomaly)


def record_ip_activity(ip_address, when=None):
    """Log a session timestamp for this IP and return whether it currently
    looks like a velocity-based attack (many sessions in a short window),
    e.g. credential stuffing or ticket scalping bursts."""
    when = when or datetime.now()
    history = IP_ACTIVITY[ip_address]
    history.append(when)

    cutoff = when - timedelta(minutes=IP_WINDOW_MINUTES)
    while history and history[0] < cutoff:
        history.popleft()

    recent_count = len(history)
    is_velocity_attack = recent_count >= IP_VELOCITY_THRESHOLD
    return recent_count, is_velocity_attack


def decide_mitigation(bot_score, is_velocity_attack, needs_review):
    """Rule-based mitigation engine. Combines the supervised bot score with
    the IP-velocity signal and the anomaly cross-check to pick an action,
    the same way a real SOC playbook would escalate a session."""
    if is_velocity_attack and bot_score >= 50:
        return "BLOCK", "Credential-stuffing / scalping pattern: multiple high-risk sessions from same IP"
    if bot_score >= 75:
        return "BLOCK", "Bot score in CRITICAL range"
    if bot_score >= 55:
        return "CHALLENGE", "Bot score elevated — CAPTCHA challenge issued"
    if is_velocity_attack:
        return "CHALLENGE", "Unusual session velocity from this IP — CAPTCHA challenge issued"
    if needs_review:
        return "REVIEW", "Supervised model says low-risk but anomaly detector disagrees — flagged for analyst review"
    return "ALLOW", "Behavior within normal range"


def log_audit_entry(session):
    entry = {
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "session_id": session["session_id"],
        "ip_address": session.get("ip_address", "unknown"),
        "platform": session["platform"],
        "bot_score": session["bot_score"],
        "anomaly_score": session.get("anomaly_score"),
        "ip_recent_count": session.get("ip_recent_count"),
        "action": session["mitigation_action"],
        "reason": session["mitigation_reason"],
    }
    AUDIT_LOG.appendleft(entry)
    save_audit_log()


def save_audit_log():
    with open(AUDIT_LOG_FILE, "w") as f:
        json.dump(list(AUDIT_LOG), f, indent=2)


def load_audit_log():
    global AUDIT_LOG
    if os.path.exists(AUDIT_LOG_FILE):
        with open(AUDIT_LOG_FILE, "r") as f:
            AUDIT_LOG = deque(json.load(f), maxlen=200)


@app.route("/api/track", methods=["POST"])
def api_track():
    data = request.get_json()
    
    features = np.array([[
        data["session_duration_min"],
        data["action_count"],
        data["click_interval_ms"],
        data["scroll_variance"],
        data["skip_rate"],
        data["mouse_movement_score"],
        data["typing_rhythm_variance"],
        int(data["browser_headless"])
    ]])
    
    print(f"Features: {features}")
    print(f"Proba: {model.predict_proba(features)}")

    proba = model.predict_proba(features)[0]
    bot_score = round(proba[1] * 100)

    anomaly_score, is_anomaly = compute_anomaly_score(features)
    needs_review = is_anomaly and bot_score < 35

    ip_address = data.get("ip_address") or request.remote_addr or random_human_ip()
    ip_recent_count, is_velocity_attack = record_ip_activity(ip_address)
    mitigation_action, mitigation_reason = decide_mitigation(bot_score, is_velocity_attack, needs_review)

    session = {
        "session_id": data.get("session_id", "LIVE"),
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "platform": data.get("platform", "StreamSafe"),
        "is_bot": bot_score >= 65,
        "bot_score": bot_score,
        "risk_level": risk_label(bot_score),
        "signals": data,
        "threat_type": "OAT-016 Skewing" if bot_score >= 65 else None,
        "flagged": bot_score >= 65,
        "ip_address": ip_address,
        "ip_recent_count": ip_recent_count,
        "is_velocity_attack": is_velocity_attack,
        "anomaly_score": anomaly_score,
        "needs_review": needs_review,
        "mitigation_action": mitigation_action,
        "mitigation_reason": mitigation_reason,
    }
    
    SESSIONS[session["session_id"]] = session
    log_audit_entry(session)
    save_sessions()
    return jsonify({
        "bot_score": bot_score,
        "risk_level": session["risk_level"],
        "anomaly_score": anomaly_score,
        "mitigation_action": mitigation_action,
        "mitigation_reason": mitigation_reason,
    })

def save_sessions():
    with open(SESSIONS_FILE, "w") as f:
        json.dump(SESSIONS, f, indent=2)

def load_sessions():
    global SESSIONS
    if os.path.exists(SESSIONS_FILE):
        with open(SESSIONS_FILE, "r") as f:
            SESSIONS = json.load(f)
    else:
        SESSIONS = {}
        save_sessions()


@app.route("/stream")
def stream():
    return render_template("streamsafe.html")

@app.route("/api/metrics")
def api_metrics():
    with open("metrics.json","r")as f:
        return jsonify(json.load(f))

def risk_label(score):
    if score >= 75:
        return "CRITICAL"
    elif score >= 55:
        return "HIGH"
    elif score >= 35:
        return "MEDIUM"
    else:
        return "LOW"


# Pre-populate some sessions
load_sessions()
load_audit_log()


# --- Routes ---

@app.route("/")
def dashboard():
    return render_template("dashboard.html")


@app.route("/api/sessions")
def api_sessions():
    """Return all sessions sorted by timestamp desc."""
    sessions = list(SESSIONS.values())
    sessions.sort(key=lambda s: s["timestamp"], reverse=True)
    return jsonify(sessions)


@app.route("/api/stats")
def api_stats():
    """Return aggregated stats for dashboard cards."""
    sessions = list(SESSIONS.values())
    total = len(sessions)
    bots = sum(1 for s in sessions if s["flagged"])
    humans = total - bots
    avg_score = round(sum(s["bot_score"] for s in sessions) / total, 1) if total else 0

    threat_counts = {}
    for s in sessions:
        if s["threat_type"]:
            t = s["threat_type"]
            threat_counts[t] = threat_counts.get(t, 0) + 1

    platform_counts = {}
    for s in sessions:
        p = s["platform"]
        platform_counts[p] = platform_counts.get(p, 0) + 1

    score_bins = {"0-25": 0, "26-50": 0, "51-75": 0, "76-100": 0}
    for s in sessions:
        sc = s["bot_score"]
        if sc <= 25:
            score_bins["0-25"] += 1
        elif sc <= 50:
            score_bins["26-50"] += 1
        elif sc <= 75:
            score_bins["51-75"] += 1
        else:
            score_bins["76-100"] += 1

    mitigation_counts = {"BLOCK": 0, "CHALLENGE": 0, "REVIEW": 0, "ALLOW": 0}
    for s in sessions:
        action = s.get("mitigation_action", "ALLOW")
        mitigation_counts[action] = mitigation_counts.get(action, 0) + 1
    needs_review_count = sum(1 for s in sessions if s.get("needs_review"))

    return jsonify({
        "total_sessions": total,
        "flagged_bots": bots,
        "human_sessions": humans,
        "avg_bot_score": avg_score,
        "detection_rate": round((bots / total * 100), 1) if total else 0,
        "threat_breakdown": threat_counts,
        "platform_breakdown": platform_counts,
        "score_distribution": score_bins,
        "mitigation_breakdown": mitigation_counts,
        "needs_review_count": needs_review_count,
    })


@app.route("/api/audit")
def api_audit():
    """Return the mitigation audit trail (most recent first)."""
    return jsonify(list(AUDIT_LOG))


@app.route("/api/simulate", methods=["POST"])
def api_simulate():
    """Simulate a new incoming session."""
    data = request.get_json(silent=True) or {}
    force_bot = data.get("force_bot", None)
    session = generate_session(force_bot=force_bot)
    save_sessions()
    return jsonify(session)


@app.route("/api/session/<session_id>")
def api_session_detail(session_id):
    """Return detailed signals for a specific session."""
    session = SESSIONS.get(session_id.upper())
    if not session:
        return jsonify({"error": "Session not found"}), 404
    return jsonify(session)


@app.route("/api/reset", methods=["POST"])
def api_reset():
    """Reset and regenerate all sessions."""
    global SESSIONS, AUDIT_LOG, IP_ACTIVITY
    SESSIONS = {}
    AUDIT_LOG = deque(maxlen=200)
    IP_ACTIVITY = defaultdict(deque)
    for _ in range(20):
        generate_session()
    save_sessions()
    save_audit_log()
    return jsonify({"status": "reset", "count": len(SESSIONS)})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
