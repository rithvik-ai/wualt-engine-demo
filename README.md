# WUALT — Distress Engine Demo

This repository contains the live source for the WUALT distress-detection engine and the
multi-subject day-stream dashboard used to demonstrate it.

**Live demo:** https://precious-kheer-956968.netlify.app/responder_day/

The dashboard streams a synthetic 24-hour day for one of five subjects through the actual
Python engine, frame by frame. Pressing the SOS button switches the stream from
10-second aggregate to true 1-Hz, ingests 60 seconds of high-fidelity data, and commits
the engine to a final state. The cascade then routes the case to either a confirmation
modal, a contact, or emergency services — based on a confidence score and two
user-configured thresholds.

---

## What's in here

| File | Role |
|---|---|
| `distress_engine.py` | The model. `UnifiedSafetyEngine`, `ManualTriggerHandler`, persistence, fusion, fall detection, geospatial scoring, audio escalation. |
| `test_input_classifier.py` | FastAPI server. Wraps the engine in HTTP endpoints (`/classify`, `/sos`, `/subjects`, etc). |
| `generate_all_subjects.py` | Synthetic-data generator. Produces 5 subject day files at 1-minute granularity. |
| `subjects/S01–S05.json` | Pre-generated days (1440 frames each, ~530 KB per file). |
| `subjects/_index.json` | Subject metadata for the dashboard's picker. |
| `sos_demo/responder_day/index.html` | Single-page dashboard. No build step — vanilla HTML + JS + CSS. |
| `sos_demo/netlify.toml` | Netlify deploy config (security headers + root redirect). |
| `render.yaml` | Render deploy config for the FastAPI backend. |
| `requirements.txt` | Python dependencies (FastAPI, uvicorn, pydantic). |

---

## Architecture

```
┌──────────┐      BLE GATT        ┌──────────┐         HTTPS         ┌──────────┐
│   RING   │ ◄──────────────────► │  PHONE   │ ◄────────────────────► │ BACKEND  │
│ (sensor) │  (1-Hz HR/HRV/SpO2,  │  app /   │  (per-frame JSON +    │ (FastAPI │
│          │   25-Hz accel,       │ dashboard│   SOS lifecycle)      │  engine) │
│          │   PDM mic, button)   │          │                       │          │
└──────────┘                      └──────────┘                       └──────────┘
                                                                          │
                                                                          ▼
                                                                  UnifiedSafetyEngine
                                                                  ├─ feature engineering
                                                                  ├─ LightGBM classifier
                                                                  ├─ isotonic calibration
                                                                  ├─ persistence accumulator
                                                                  ├─ state machine
                                                                  └─ multimodal fusion
```

The repo contains the **backend** (engine + FastAPI server) and the **dashboard**
(which simulates the phone for the demo). The ring is real hardware not included here.

---

## Run locally

```bash
# install Python deps
python3 -m venv venv
./venv/bin/pip install -r requirements.txt

# Terminal 1 — backend (FastAPI on port 8766)
./venv/bin/uvicorn test_input_classifier:app --port 8766

# Terminal 2 — dashboard (static HTTP server on port 4915)
python3 -m http.server 4915 --directory sos_demo/responder_day

# Open in browser
open "http://localhost:4915/?api=http://localhost:8766"
```

The dashboard's `?api=...` query param tells it which backend to hit. When omitted, it
defaults to a Render deployment of this same backend.

---

## API endpoints

### Inference

| Endpoint | What it does |
|---|---|
| `POST /classify` | Per-frame inference. Takes physiology + audio + geo + thresholds, returns state + recipient + persistence. |
| `POST /predict-batch` | Batch version of `/classify` for replay-on-reconnect. |
| `POST /reset-engine?user_id=X` | Wipe per-user engine state. |

### SOS lifecycle

| Endpoint | What it does |
|---|---|
| `POST /sos` | Wearer pressed the SOS button. Opens the case lifecycle, returns recipient + cancel window. |
| `POST /sos/confirm` | User responded to the modal (`safe` / `help` / `cancel`). |
| `GET /sos/status` | Poll for current lifecycle phase. |
| `POST /sos/reset` | Clear the SOS state machine. |

### Subjects (synthetic data)

| Endpoint | What it does |
|---|---|
| `GET /subjects` | List the 5 demo subjects. |
| `GET /subject/{sid}` | Return one subject's full 24-hour day (1440 frames). |

### Utility

| Endpoint | What it does |
|---|---|
| `GET /health` | Liveness check (used by Render). |

Full request/response schemas are in `test_input_classifier.py`.

---

## The 5 subjects

| ID | Subject | Stress window | Distress window |
|---|---|---|---|
| S01 | Priya Singh, 28F, office worker | 14:00–14:59 (meeting) | 15:05–15:34 (parking lot assault) |
| S02 | Lakshmi Reddy, 52F, domestic worker | 17:00–17:59 (employer confrontation) | 04:33–04:34 (followed on dawn commute) |
| S03 | Mrs Iyer, 71F, retired, lives alone | 10:00–10:59 (anniversary grief) | 19:21–19:34 (kitchen fall) |
| S04 | Vidya Iyengar, 35F, sales exec with anxiety | 17:00–17:59 (anticipatory anxiety) | 09:35–09:42 (panic attack mid-pitch) |
| S05 | Sushma Rao, 45F, yoga instructor | 15:30–15:59 (team call disagreement) | **none — calm-baseline subject** |

S05 is included specifically to demonstrate that the engine **does not false-alarm**
during exercise (her yoga classes) or mild daily stress. Press SOS during her yoga class
(~08:30) — heart rate hits 130 bpm but the engine stays at NORMAL because motion
classification suppresses HR-only flags.

---

## Engine specifics relevant to audit

### Persistence
- 60 seconds of sustained stress/distress signal required before the state machine
  promotes the final state.
- Continuity-based (not cumulative): a single normal frame in the middle of stress
  resets the counter to 0 after a 2-frame grace period.
- Implementation in `distress_engine.py` → `class PersistenceTracker`.

### Recipient selection
- Two user-configured thresholds (default 0.50 contact, 0.85 police).
- Distress confidence (raw `weighted_score`, range 0–1) is compared to those thresholds.
- One recipient is chosen — never a tier ladder, never both contact and police in sequence.
- Implementation: `ManualTriggerHandler.decide_recipient()`.

### Cancellation
- Normal-final outcome: 60-second confirmation modal. Silence = treated as inability to
  respond, escalates to emergency contact.
- Stress-final outcome: 30-second 1-click cancel before contact is dialled.
- Distress-final outcome: 15-second PIN-required cancel before emergency services
  are dialled. Wrong PIN ×3 = duress assumed, UI displays "CANCELLED" but engine silently
  continues.

### Multimodal fusion
- Audio danger (scream / gunshot / explosion at confidence > 0.7) + physiological
  distress → forces final state to DISTRESS, bypasses persistence.
- Confirmed fall (free-fall + impact + post-impact stillness) → same bypass.
- SpO2 ≤ 90 % sustained 10 frames → bypass.

A full architecture spec with multi-source citations is available separately as
`WUALT_Post_Trigger_Cascade.pdf` (not in this repo).

---

## Deploy

### Backend → Render.com (free tier)

```bash
# render.yaml is already configured
git push origin main
# then connect the repo at https://render.com → New → Blueprint
```

Render auto-detects `render.yaml`, builds, and gives a URL like
`https://wualt-engine-<random>.onrender.com`.

### Dashboard → Netlify (drag-and-drop or CLI)

```bash
# Drag-and-drop: open https://app.netlify.com/drop and drop the sos_demo folder

# Or CLI:
cd sos_demo
netlify deploy --prod --dir=.
```

The deployed dashboard defaults to the Render backend URL hardcoded in the JS. Override
with `?api=...` query param for local development.

---

## Dependencies

The runtime is intentionally lean — three Python packages:

- `fastapi` — web framework
- `uvicorn[standard]` — ASGI server
- `pydantic` — request validation

No ML libraries are required for the demo because the engine uses hand-tuned z-score
thresholds (no LightGBM pickle is loaded in this configuration). Audio fusion is mocked
on the demo backend; the real audio microservice (YAMNet + AST + Praat) is a separate
deploy not included here.

---

## License

MIT. See `LICENSE`.
