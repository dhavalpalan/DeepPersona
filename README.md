# DeepPersona — Security Dashboard

Synthetic Identity & Bot Detection Platform for the Media & Entertainment Industry.

## Quick Setup

```bash
# 1. Install dependencies
pip install flask

# 2. Run the app
python app.py

# 3. Open in browser
http://localhost:5000
```

## What's Running

- **`app.py`** — Flask backend with behavioral scoring engine and REST APIs
- **`templates/dashboard.html`** — Dark-themed live Security Dashboard

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Security Dashboard UI |
| `/api/sessions` | GET | All sessions (sorted by time) |
| `/api/stats` | GET | Aggregated stats for dashboard cards & charts |
| `/api/simulate` | POST | Inject a new simulated session |
| `/api/session/<id>` | GET | Detailed signals for one session |
| `/api/audit` | GET | Mitigation audit trail (most recent decisions first) |
| `/api/reset` | POST | Reset all session data |

### Simulate a session

```bash
# Random session
curl -X POST http://localhost:5000/api/simulate

# Force a bot
curl -X POST http://localhost:5000/api/simulate \
     -H "Content-Type: application/json" \
     -d '{"force_bot": true}'

# Force a human
curl -X POST http://localhost:5000/api/simulate \
     -d '{"force_bot": false}' -H "Content-Type: application/json"
```

## How Bot Scoring Works

The `compute_bot_score()` function in `app.py` analyzes 8 behavioral signals:

| Signal | Bot Pattern | Human Pattern |
|---|---|---|
| Session Duration | < 2 min | 5–45 min |
| Action Count | 80–300 | 5–60 |
| Click Interval | 50–250 ms | 400–3000 ms |
| Scroll Variance | 0.01–0.08 | 0.2–0.9 |
| Skip Rate | 70–100% | 5–45% |
| Browser Headless | Often YES | NO |
| Mouse Movement Score | 0.0–0.25 | 0.6–1.0 |
| Typing Rhythm Variance | 0.0–0.05 | 0.15–0.65 |

Scores ≥ 65 are flagged as bots. Model: `RandomForestClassifier` (10 trees, max depth 5), trained in `train_model.py` on `sessions.csv`.

## Threat Response Engine (Mitigation + IP Velocity)

Beyond just showing a bot score, the system now makes an actual decision and logs it:

1. **IP session-velocity tracking** — every session is tagged with a source IP. If 3+ sessions arrive from the same IP within a 5-minute window, that's treated as a velocity-based attack pattern (credential stuffing / ticket scalping bursts), regardless of any single session's score.
2. **Mitigation decision engine** (`decide_mitigation()` in `app.py`) — combines the bot score with the IP-velocity signal to pick one of four actions:
   - **BLOCK** — critical bot score, or a velocity attack combined with an elevated score
   - **CHALLENGE** — elevated bot score or unusual velocity → CAPTCHA issued
   - **REVIEW** — see anomaly cross-check below
   - **ALLOW** — behavior within normal range
3. **Audit trail** — every decision (timestamp, session ID, IP, scores, action, reason) is logged and persisted to `audit_log.json`, viewable via `/api/audit` and the dashboard's "Mitigation Audit Trail" table.

## Unsupervised Anomaly Cross-Check

`train_model.py` also trains an `IsolationForest` (`anomaly_model.py` → `anomaly_model.pkl`) on the same 8 features — but **without** the `is_bot` label. This acts as an independent second opinion on the supervised RandomForest:

- Every session gets an **anomaly score (0–100)**, alongside its supervised bot score.
- If the supervised model calls a session low-risk (bot score < 35) but the IsolationForest flags it as an outlier, the session is marked **`needs_review`** — a proxy for behavior that doesn't match known bot patterns from training but still looks statistically unusual (i.e., a possible novel/zero-day evasion attempt).
- On the held-out test set, the unsupervised model alone agrees with the ground-truth label ~79% of the time — clearly weaker than the supervised RandomForest (99.6%) used alone, which is expected and worth noting in the report: it's not meant to replace the classifier, but to catch what it might miss.

## OWASP Threat Coverage

- **OAT-005 Scalping** — Ticket bots purchasing before humans
- **OAT-013 Credential Stuffing** — Bot-driven login attacks
- **OAT-016 Skewing** — Fake streams inflating play counts
- **Synthetic Identity** — Fabricated accounts mimicking real users

## Project Developer

 **Dhaval** — Project Lead, Architecture, Backend, ML Model
