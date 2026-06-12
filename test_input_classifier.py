"""
Standalone test harness for the FULL distress engine.

Enter values for the four modalities (physiology, motion,
geospatial, audio), click classify, watch the engine output normal / stress /
distress with reasoning.

Run
---
    /opt/homebrew/bin/python3 -m uvicorn test_input_classifier:app --port 8766

Then open http://localhost:8766/

Endpoints
---------
    GET   /                UI
    POST  /classify {...}  build pipeline_output from the form, run UnifiedSafetyEngine
    GET   /presets         list of preset configurations
    POST  /preset {name}   apply a preset (returns the verdict)
"""
from __future__ import annotations

import time
from typing import Optional, Dict, List, Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

import distress_engine as de

# Production persistence — 60s of sustained signal required before state promotion.
# Set this to a smaller number (e.g. 30) for faster demos; 0 disables persistence entirely.
de.PERSISTENCE_STRESS_S   = 60
de.PERSISTENCE_DISTRESS_S = 60

app = FastAPI(title="WUALT Input Classifier — Test Harness")

# Per-user engine instances — persistence requires the SAME engine across calls
# so the internal timers accumulate. Keyed by user_id.
_ENGINES: Dict[str, "de.UnifiedSafetyEngine"] = {}

def get_engine(user_id: str) -> "de.UnifiedSafetyEngine":
    if user_id not in _ENGINES:
        _ENGINES[user_id] = de.UnifiedSafetyEngine()
    return _ENGINES[user_id]

def reset_engine(user_id: str) -> None:
    _ENGINES.pop(user_id, None)

# CORS — the static dashboard at sos_demo/responder/ needs to hit this from a different origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],         # demo only; tighten for production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Scenario dataset (lazy-loaded, stdlib csv) ──────────────
_SCENARIO_ROWS: Optional[List[Dict[str, str]]] = None
def _load_scenarios() -> List[Dict[str, str]]:
    global _SCENARIO_ROWS
    if _SCENARIO_ROWS is None:
        import csv
        with open("synthetic_dataset.csv", newline="") as f:
            _SCENARIO_ROWS = list(csv.DictReader(f))
    return _SCENARIO_ROWS

# ── Manual trigger handler (per-session) ─────────────────────
_SOS_HANDLER = de.ManualTriggerHandler()


# Standard deviation priors for z-score computation
STD = {
    "hr":    5.0,    # bpm
    "hrv":   8.0,    # ms (hr_stability_score)
    "spo2":  0.8,    # %
    "temp":  0.15,   # °C
}


# ─────────────────────────────────────────────────────────────
# Request schema
# ─────────────────────────────────────────────────────────────
class ClassifyRequest(BaseModel):
    # Physiology — current values
    hr:    float = 72.0
    hrv:   float = 55.0          # RMSSD ms (mapped to hr_stability_score)
    spo2:  float = 98.0
    temp:  float = 36.6
    # Per-user engine identity — persistence accumulates per user_id
    user_id: str = "demo"
    # Per-user recipient policy — set by the user in onboarding
    contact_threshold: float = 0.50
    police_threshold:  float = 0.85
    emergency_contact_name:  str = "R. Singh (sister)"
    emergency_contact_phone: str = "+91 98••• 11•••"

    # Per-user baselines (so z-scores are personalised)
    baseline_hr:   float = 72.0
    baseline_hrv:  float = 55.0
    baseline_spo2: float = 98.0
    baseline_temp: float = 36.6

    # Signal quality 0–1 (gating; <0.30 frame is rejected)
    sqi_overall:   float = 0.88
    finger_on:     bool  = True

    # Motion
    acc_mag:       float = 0.98   # gravity baseline ≈ 1g
    dyn_acc_mag:   float = 0.01   # 0.01 still · 0.10 walking · 0.45 exercise

    # Audio — fake the AudioIntelligenceEngine output
    audio_enabled:    bool   = False
    audio_risk_level: str    = "normal"   # normal | elevated | danger
    audio_top_class:  str    = "silence"  # silence | speech | scream | gunshot | explosion | music

    # Geospatial
    is_home_zone:        bool  = True
    is_work_zone:        bool  = False
    is_known_area:       bool  = True
    is_unfamiliar_area:  bool  = False
    distance_from_home_km: float = 0.0
    hour_of_day:         int   = 14
    is_night:            bool  = False
    is_stationary:       bool  = True
    is_walking:          bool  = False
    is_vehicle_like_motion: bool = False
    sudden_route_change: bool  = False
    sudden_stop:         bool  = False
    phone_connected:     bool  = True


# ─────────────────────────────────────────────────────────────
# Build pipeline_output from the form, then run the engine
# ─────────────────────────────────────────────────────────────
def zscore(current: float, baseline: float, std: float) -> float:
    return round((current - baseline) / max(std, 1e-6), 3)


def build_pipeline_output(r: ClassifyRequest) -> Dict[str, Any]:
    ts = int(time.time())
    sample = {
        "timestamp": ts, "sequence": ts,
        "hr":   r.hr,
        "hr_stability_score": r.hrv,   # the engine calls HRV "hr_stability_score"
        "temp": r.temp, "temp_raw": r.temp,
        "spo2": r.spo2,
        "acc_mag":     r.acc_mag,
        "dyn_acc_mag": r.dyn_acc_mag,
        "acc_x": 0.02, "acc_y": -0.01, "acc_z": 0.98,
        "gravity_x": 0.02, "gravity_y": -0.01, "gravity_z": 0.98,
        "finger_on":  r.finger_on,
        "charging":   False,
        "battery_mv": 3850, "die_temp": 34.5, "adc_raw": 112000,
        "thermal_bias": 0.0,
        "sqi": {
            "hr":   max(0.7, r.sqi_overall),
            "hrv":  max(0.6, r.sqi_overall - 0.1),
            "temp": max(0.8, r.sqi_overall),
            "acc":  0.95,
            "spo2": max(0.7, r.sqi_overall),
            "ppg":  max(0.7, r.sqi_overall - 0.05),
            "overall": r.sqi_overall,
        },
        "accepted":      True,
        "reject_reasons": [],
        "clinical_flags": [],
    }

    zscores = {
        "hr":                  zscore(r.hr,   r.baseline_hr,   STD["hr"]),
        "hr_stability_score":  zscore(r.hrv,  r.baseline_hrv,  STD["hrv"]),
        "spo2":                zscore(r.spo2, r.baseline_spo2, STD["spo2"]),
        "temp":                zscore(r.temp, r.baseline_temp, STD["temp"]),
        "acc_mag":             0.0,
    }

    window = {
        "window_n": 25,
        "hr_mean": r.hr, "hr_var": 2.0, "hr_min": r.hr - 2, "hr_max": r.hr + 2,
        "hr_stability_score_mean": r.hrv, "hr_stability_score_var": 4.0,
        "hr_stability_score_min": r.hrv - 4, "hr_stability_score_max": r.hrv + 4,
        "temp_mean": r.temp, "temp_var": 0.01,
        "temp_min":  r.temp - 0.1, "temp_max": r.temp + 0.1,
        "spo2_mean": r.spo2, "spo2_var": 0.5,
        "spo2_min":  r.spo2 - 0.5, "spo2_max": r.spo2 + 0.5,
        "acc_mag_mean": r.acc_mag, "acc_mag_var": 0.001,
        "acc_mag_min":  r.acc_mag - 0.01, "acc_mag_max": r.acc_mag + 0.01,
    }

    return {
        "sample":         sample,
        "zscores":        zscores,
        "baseline_ready": True,
        "window":         window,
    }


def build_geo_context(r: ClassifyRequest) -> Dict[str, Any]:
    return {
        "latitude": 0.0, "longitude": 0.0, "timestamp": int(time.time()),
        "speed_kmph": 0.0, "heading": 0.0,
        "is_home_zone":         r.is_home_zone,
        "is_work_zone":         r.is_work_zone,
        "is_known_area":        r.is_known_area,
        "is_unfamiliar_area":   r.is_unfamiliar_area,
        "distance_from_home_km": r.distance_from_home_km,
        "hour_of_day":          r.hour_of_day,
        "is_night":             r.is_night,
        "is_stationary":        r.is_stationary,
        "is_walking":           r.is_walking,
        "is_vehicle_like_motion": r.is_vehicle_like_motion,
        "sudden_route_change":  r.sudden_route_change,
        "sudden_stop":          r.sudden_stop,
        "phone_connected":      r.phone_connected,
        "phone_disconnect_duration_s": 0.0,
    }


def fake_audio_result(r: ClassifyRequest) -> Optional[Dict[str, Any]]:
    """Skip the heavy AudioIntelligenceEngine — inject a result dict directly."""
    if not r.audio_enabled:
        return None
    risk_score_map = {"normal": 0.10, "elevated": 0.40, "danger": 0.78}
    sev_map        = {"normal": "low", "elevated": "medium", "danger": "high"}
    return {
        "audio_risk_level":   r.audio_risk_level,
        "audio_risk_score":   risk_score_map.get(r.audio_risk_level, 0.10),
        "top_class":          r.audio_top_class,
        "environment": {
            "danger_score":   risk_score_map.get(r.audio_risk_level, 0.10),
            "top_class":      r.audio_top_class,
        },
        "vocal_stress": {"stress_score": 0.5 if r.audio_risk_level != "normal" else 0.1},
        "noise":        {"noise_score":  0.4 if r.audio_risk_level == "danger" else 0.15},
        "alert": {
            "title":    f"Audio: {r.audio_top_class}",
            "message":  f"Detected {r.audio_top_class} ({r.audio_risk_level})",
            "severity": sev_map.get(r.audio_risk_level, "low"),
        },
    }


@app.post("/classify")
def classify(req: ClassifyRequest) -> JSONResponse:
    # Per-user engine — persistence accumulates across calls for this user_id.
    # Sliding through the same scenario for ~60s lets distress promote naturally.
    unified = get_engine(req.user_id)

    # Monkey-patch the audio engine to return our fake dict (avoids ML deps)
    audio_dict = fake_audio_result(req)
    unified.audio.evaluate = lambda audio_input=None: audio_dict      # type: ignore

    pipeline_output = build_pipeline_output(req)
    geo_context     = build_geo_context(req)

    result = unified.evaluate(
        pipeline_output = pipeline_output,
        geo_context     = geo_context,
        audio_input     = {"audio_array": [], "sample_rate": 16000} if req.audio_enabled else None,
    )

    # Tidy verdict for the UI — pull out the headline numbers
    state       = result.get("state", "normal")
    confidence  = result.get("confidence", 0.0)
    signals     = result.get("contributing_signals", [])
    safety      = result.get("safety", {})
    top_alert   = result.get("alert", {})
    debug       = result.get("debug", {})

    # Persistence countdown — exposed to the dashboard so the UI can show
    # "building 24/60s" while the engine accumulates evidence.
    persistence_s = float(debug.get("persistence_s", 0.0))
    raw_state     = "distress" if debug.get("weighted_score", 0.0) > 0.40 else \
                    "stress"   if debug.get("weighted_score", 0.0) > 0.15 else "normal"

    # Recipient preview — who WOULD be notified if SOS fired right now.
    # We use the raw weighted_score (the strength of the distress signature on
    # the CURRENT frame) rather than the persistence-gated final-state confidence.
    # This way the recipient card answers "if this signal sustains for 60s,
    # who would the engine alert?" — even during the persistence build-up.
    raw_score = max(0.0, min(1.0, float(debug.get("weighted_score", 0.0))))
    distress_confidence = round(raw_score, 3)

    recipient = de.ManualTriggerHandler.decide_recipient(
        engine_confidence  = distress_confidence,
        contact_threshold  = req.contact_threshold,
        police_threshold   = req.police_threshold,
        emergency_contact  = {"name": req.emergency_contact_name, "phone": req.emergency_contact_phone},
    )

    return JSONResponse({
        "state":                state,
        "confidence":           confidence,
        "contributing_signals": signals,
        "safety_risk_level":    safety.get("risk_level"),
        "safety_risk_score":    safety.get("risk_score"),
        "safety_reasoning":     safety.get("reasoning", []),
        "top_alert":            top_alert,
        "zscores":              pipeline_output["zscores"],
        "audio":                audio_dict,
        "persistence_s":        persistence_s,
        "persistence_required_s": float(de.PERSISTENCE_DISTRESS_S),
        "raw_state":            raw_state,
        "weighted_score":       debug.get("weighted_score", 0.0),
        "motion_state":         debug.get("motion_state"),
        # NEW — recipient + thresholds (for the dashboard's confidence-scale UI)
        "recipient":            recipient,
        "distress_confidence":  distress_confidence,
        "contact_threshold":    req.contact_threshold,
        "police_threshold":     req.police_threshold,
        "full":                 result,
    })


@app.post("/reset-engine")
def reset_engine_endpoint(user_id: str = "demo") -> JSONResponse:
    reset_engine(user_id)
    return JSONResponse({"reset": True, "user_id": user_id})


# ─────────────────────────────────────────────────────────────
# Presets — one-click scenarios for the boss
# ─────────────────────────────────────────────────────────────
PRESETS: Dict[str, ClassifyRequest] = {
    "calm_baseline": ClassifyRequest(),                      # defaults = normal
    "workplace_stress": ClassifyRequest(
        # HR z=2.4, HRV z=-1.625, weighted_score=0.386 → stress band.
        hr=84, hrv=42, spo2=98, temp=36.7,
        dyn_acc_mag=0.04, is_home_zone=False, is_work_zone=True,
        is_known_area=True, is_unfamiliar_area=False, hour_of_day=15,
    ),
    "evening_jog": ClassifyRequest(
        # HR elevated, HRV and temp stay at baseline, motion clearly EXERCISE
        # → HR-only flag suppressed by exercise classifier → state normal.
        hr=130, hrv=55, spo2=98, temp=36.6,
        baseline_hr=72, baseline_hrv=55, baseline_temp=36.6,
        dyn_acc_mag=0.55, is_home_zone=False, is_known_area=True,
        is_walking=True, hour_of_day=19,
    ),
    "panic_attack": ClassifyRequest(
        hr=128, hrv=14, spo2=96.5, temp=37.2,
        dyn_acc_mag=0.06, is_home_zone=True, is_known_area=True,
        hour_of_day=2, is_night=True,
        audio_enabled=True, audio_risk_level="elevated", audio_top_class="speech",
    ),
    "assault_at_night": ClassifyRequest(
        hr=132, hrv=12, spo2=96.0, temp=37.3,
        dyn_acc_mag=0.20, is_home_zone=False, is_known_area=False,
        is_unfamiliar_area=True, distance_from_home_km=4.2,
        hour_of_day=1, is_night=True, is_walking=True,
        audio_enabled=True, audio_risk_level="danger", audio_top_class="scream",
    ),
    "silent_hypoxia": ClassifyRequest(
        hr=105, hrv=35, spo2=86.5, temp=36.8,
        dyn_acc_mag=0.02, is_home_zone=True, hour_of_day=23, is_night=True,
    ),
}


@app.get("/presets")
def list_presets() -> JSONResponse:
    return JSONResponse({"presets": list(PRESETS.keys())})


@app.post("/preset/{name}")
def apply_preset(name: str) -> JSONResponse:
    preset = PRESETS.get(name)
    if preset is None:
        return JSONResponse({"error": f"unknown preset: {name}"}, status_code=400)
    # Return both the input the preset represents, and the verdict for it
    verdict = classify(preset).body  # type: ignore
    import json as _json
    return JSONResponse({
        "preset_name":  name,
        "preset_input": preset.model_dump(),
        "verdict":      _json.loads(verdict),
    })


# ─────────────────────────────────────────────────────────────
# SYNTHETIC SCENARIO REPLAY
# ─────────────────────────────────────────────────────────────
# ── Single-subject one-day view (legacy single-file path) ────
_SUBJECT_DAY: Optional[Dict[str, Any]] = None
def _load_subject_day() -> Dict[str, Any]:
    global _SUBJECT_DAY
    if _SUBJECT_DAY is None:
        import json
        with open("subject_day.json") as f:
            _SUBJECT_DAY = json.load(f)
    return _SUBJECT_DAY


@app.get("/subject-day")
def get_subject_day() -> JSONResponse:
    """Return one synthetic subject's 24-hour day at 1-minute granularity."""
    return JSONResponse(_load_subject_day())


# ── Multi-subject (5 profiles) endpoints ─────────────────────
_SUBJECTS_INDEX: Optional[Dict[str, Any]] = None
_SUBJECT_CACHE: Dict[str, Dict[str, Any]] = {}

def _load_subjects_index() -> Dict[str, Any]:
    global _SUBJECTS_INDEX
    if _SUBJECTS_INDEX is None:
        import json
        with open("subjects/_index.json") as f:
            _SUBJECTS_INDEX = json.load(f)
    return _SUBJECTS_INDEX

def _load_subject(sid: str) -> Optional[Dict[str, Any]]:
    if sid not in _SUBJECT_CACHE:
        import json, os
        path = f"subjects/{sid}.json"
        if not os.path.exists(path):
            return None
        with open(path) as f:
            _SUBJECT_CACHE[sid] = json.load(f)
    return _SUBJECT_CACHE[sid]

@app.get("/subjects")
def list_subjects() -> JSONResponse:
    """List the 5 demo subjects with their metadata."""
    return JSONResponse(_load_subjects_index())

@app.get("/subject/{sid}")
def get_subject(sid: str) -> JSONResponse:
    """Return one subject's 24-hour day (1440 frames at 1-min granularity)."""
    d = _load_subject(sid)
    if d is None:
        return JSONResponse({"error": f"subject not found: {sid}"}, status_code=404)
    return JSONResponse(d)


@app.get("/scenarios")
def list_scenarios() -> JSONResponse:
    """List the distinct scenario types in synthetic_dataset.csv,
    grouped by category, with the expected outcome and frame count for each."""
    rows = _load_scenarios()
    # group by (category, name, expected_state)
    seen: Dict[tuple, int] = {}
    for r in rows:
        key = (r["scenario_category"], r["scenario_name"], r["expected_state"])
        seen[key] = seen.get(key, 0) + 1
    by_cat: Dict[str, List[Dict[str, Any]]] = {}
    for (cat, name, exp), n in sorted(seen.items()):
        by_cat.setdefault(cat, []).append({
            "name":           name,
            "expected_state": exp,
            "n_frames":       n,
        })
    return JSONResponse({"categories": by_cat, "total": len(seen)})


@app.get("/scenario/{name}")
def get_scenario_frames(name: str, profile_id: Optional[str] = None) -> JSONResponse:
    """Return the ordered frames for one scenario.
    Optionally filter to one profile_id (e.g. S01); otherwise picks the first available."""
    rows = _load_scenarios()
    sub = [r for r in rows if r["scenario_name"] == name]
    if not sub:
        return JSONResponse({"error": f"scenario not found: {name}"}, status_code=404)
    if profile_id:
        sub = [r for r in sub if r["profile_id"] == profile_id]
        if not sub:
            return JSONResponse({"error": f"profile {profile_id} not in scenario {name}"}, status_code=404)
    else:
        first_profile = sub[0]["profile_id"]
        sub = [r for r in sub if r["profile_id"] == first_profile]

    def _f(r, key, default=0.0):
        try:    return float(r.get(key, default))
        except: return float(default)

    frames: List[Dict[str, Any]] = []
    for r in sub:
        frames.append({
            "frame_id":            int(float(r["frame_id"])),
            "scenario":            r["scenario_name"],
            "expected":            r["expected_state"],
            "hr":                  _f(r, "hr",         72.0),
            "hrv":                 _f(r, "hrv_rmssd",  55.0),
            "spo2":                _f(r, "spo2",       98.0),
            "temp":                _f(r, "temp",       36.6),
            "dyn_acc_mag":         _f(r, "dyn_acc_mag", 0.01),
            "acc_mag":             _f(r, "acc_mag",    0.98),
            "sqi_overall":         _f(r, "sqi_overall", _f(r, "sqi", 0.88)),
            "frame_in_scenario":   int(float(r.get("frame_in_scenario", 0))),
            "total_frames":        int(float(r.get("total_frames", len(sub)))),
            "profile_id":          r.get("profile_id", ""),
            "profile_label":       r.get("profile_label", ""),
        })
    return JSONResponse({
        "scenario":       name,
        "expected_state": sub[0]["expected_state"],
        "profile_id":     sub[0].get("profile_id", ""),
        "profile_label":  sub[0].get("profile_label", ""),
        "frames":         frames,
        "n_frames":       len(frames),
    })


# ─────────────────────────────────────────────────────────────
# MANUAL SOS — real ManualTriggerHandler from distress_engine.py
# ─────────────────────────────────────────────────────────────
class SosPress(BaseModel):
    engine_state:      Optional[str]  = None    # kept for logging only
    engine_confidence: float          = 0.5
    engine_severity:   float          = 0.0
    contact_threshold: float          = 0.50
    police_threshold:  float          = 0.85
    emergency_contact_name:  Optional[str] = None
    emergency_contact_phone: Optional[str] = None

class SosAction(BaseModel):
    action: str  # "safe" | "help" | "cancel"


@app.post("/sos")
def sos_press(body: SosPress) -> JSONResponse:
    v = _SOS_HANDLER.on_button_press(
        engine_state      = body.engine_state,
        engine_confidence = body.engine_confidence,
        engine_severity   = body.engine_severity,
        contact_threshold = body.contact_threshold,
        police_threshold  = body.police_threshold,
        emergency_contact = {
            "name":  body.emergency_contact_name or "Emergency contact",
            "phone": body.emergency_contact_phone or "",
        },
    )
    return JSONResponse(v)


@app.post("/sos/confirm")
def sos_confirm(body: SosAction) -> JSONResponse:
    return JSONResponse(_SOS_HANDLER.on_user_action(body.action))


@app.get("/sos/status")
def sos_status() -> JSONResponse:
    return JSONResponse(_SOS_HANDLER.status())


@app.post("/sos/reset")
def sos_reset() -> JSONResponse:
    _SOS_HANDLER.reset()
    return JSONResponse({"reset": True})


@app.post("/sos/tick")
def sos_tick() -> JSONResponse:
    """Caller drives the clock — POST per second to advance the state machine."""
    v = _SOS_HANDLER.tick()
    return JSONResponse(v or _SOS_HANDLER.status())


@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse({
        "ok": True,
        "engine_loaded": True,
        "scenario_dataset_loaded": _SCENARIO_ROWS is not None,
    })


# ─────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def ui() -> str:
    return HTML


HTML = r"""
<!doctype html>
<html><head><meta charset="utf-8"><title>WUALT Classifier</title>
<style>
  :root { color-scheme: dark; }
  body { font-family: -apple-system, system-ui, sans-serif; max-width: 1280px;
         margin: 24px auto; padding: 0 16px; background: #0d0f14; color: #e8ecf1; }
  h1   { font-weight: 700; letter-spacing: -0.3px; margin: 0 0 4px; }
  .sub { color: #8a93a3; margin: 0 0 24px; font-size: 14px; }
  .grid{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
  .card{ background: #161a22; border: 1px solid #232a36; border-radius: 12px;
         padding: 18px; }
  .card h2 { margin: 0 0 12px; font-size: 13px; text-transform: uppercase;
             letter-spacing: 1px; color: #8a93a3; font-weight: 600; }
  .row { display: grid; grid-template-columns: 1fr 1fr; gap: 10px 16px; }
  label { display: block; font-size: 12px; color: #8a93a3; margin: 6px 0 2px; }
  input[type=number] { width: 100%; background: #0d1218; color: #e8ecf1;
        border: 1px solid #2c3340; padding: 6px 10px; border-radius: 6px;
        font: inherit; }
  select { background: #0d1218; color: #e8ecf1; border: 1px solid #2c3340;
        padding: 6px 10px; border-radius: 6px; font: inherit; width: 100%; }
  .toggle { display: inline-flex; align-items: center; gap: 6px;
        font-size: 13px; color: #d6dbe4; margin: 4px 12px 4px 0; }
  button { font: inherit; padding: 9px 14px; border-radius: 8px; cursor: pointer;
        border: 1px solid #2c3340; background: #1d222c; color: #e8ecf1;
        margin: 3px 4px 3px 0; }
  button:hover { background: #252b37; }
  .classify { background: linear-gradient(180deg, #4f9aff, #2563eb); color: white;
        border: 0; padding: 12px 22px; font-size: 14px; font-weight: 600;
        box-shadow: 0 4px 16px rgba(37,99,235,0.35); width: 100%; }
  .verdict { display: flex; align-items: center; gap: 16px; padding: 14px 0; }
  .state { font-size: 38px; font-weight: 800; letter-spacing: -1px; }
  .s-normal   { color: #6ee7b7; }
  .s-stress   { color: #fde68a; }
  .s-distress { color: #fca5a5; }
  .pill { display: inline-block; padding: 4px 10px; border-radius: 999px;
        font-size: 12px; font-weight: 600; margin-right: 6px; }
  .r-low      { background: #16291f; color: #6ee7b7; border: 1px solid #1f3f30; }
  .r-medium   { background: #2a2716; color: #fde68a; border: 1px solid #4a3f1a; }
  .r-high     { background: #2a1f16; color: #fdba74; border: 1px solid #4a3520; }
  .r-critical { background: #2a1818; color: #fca5a5; border: 1px solid #4a2222; }
  ul { padding-left: 20px; margin: 4px 0; font-size: 13.5px; }
  pre  { background: #0a0d12; padding: 12px; border-radius: 8px; overflow: auto;
        font-size: 11.5px; line-height: 1.4; border: 1px solid #1c222d;
        max-height: 340px; }
  .small { font-size: 12px; color: #8a93a3; margin-top: 4px; }
</style></head>
<body>

<h1>WUALT Classifier</h1>
<p class="sub">Enter values across the four modalities. Engine returns normal / stress / distress.
Persistence is disabled so a single frame fires the verdict immediately.</p>

<div class="card" style="margin-bottom: 16px;">
  <h2>Presets — one-click scenarios</h2>
  <button onclick="applyPreset('calm_baseline')">Calm baseline</button>
  <button onclick="applyPreset('workplace_stress')">Workplace stress</button>
  <button onclick="applyPreset('evening_jog')">Evening jog (exercise)</button>
  <button onclick="applyPreset('panic_attack')">Panic attack at 2am</button>
  <button onclick="applyPreset('assault_at_night')">Assault at night</button>
  <button onclick="applyPreset('silent_hypoxia')">Silent hypoxia</button>
</div>

<div class="grid">
  <div>
    <div class="card">
      <h2>Physiology</h2>
      <div class="row">
        <div><label>HR (bpm)</label><input id="hr" type="number" step="0.5" value="72"></div>
        <div><label>Baseline HR</label><input id="baseline_hr" type="number" step="0.5" value="72"></div>
        <div><label>HRV / RMSSD (ms)</label><input id="hrv" type="number" step="0.5" value="55"></div>
        <div><label>Baseline HRV</label><input id="baseline_hrv" type="number" step="0.5" value="55"></div>
        <div><label>SpO2 (%)</label><input id="spo2" type="number" step="0.1" value="98"></div>
        <div><label>Baseline SpO2</label><input id="baseline_spo2" type="number" step="0.1" value="98"></div>
        <div><label>Skin temp (°C)</label><input id="temp" type="number" step="0.05" value="36.6"></div>
        <div><label>Baseline temp</label><input id="baseline_temp" type="number" step="0.05" value="36.6"></div>
        <div><label>Signal quality (0–1)</label><input id="sqi_overall" type="number" step="0.01" min="0" max="1" value="0.88"></div>
        <div><span class="toggle"><input type="checkbox" id="finger_on" checked> finger on</span></div>
      </div>
    </div>

    <div class="card" style="margin-top: 16px;">
      <h2>Motion</h2>
      <div class="row">
        <div><label>acc_mag (g, gravity baseline ≈ 1)</label><input id="acc_mag" type="number" step="0.01" value="0.98"></div>
        <div><label>dyn_acc_mag (g)</label><input id="dyn_acc_mag" type="number" step="0.01" value="0.01"></div>
      </div>
      <p class="small">0.01 still · 0.04 walking · 0.10 walking-active · 0.45 exercise</p>
    </div>

    <div class="card" style="margin-top: 16px;">
      <h2>Audio</h2>
      <div class="row">
        <div style="grid-column: span 2;">
          <span class="toggle"><input type="checkbox" id="audio_enabled"> Audio enabled</span>
        </div>
        <div><label>Risk level</label>
          <select id="audio_risk_level">
            <option value="normal">normal</option>
            <option value="elevated">elevated</option>
            <option value="danger">danger</option>
          </select>
        </div>
        <div><label>Top sound class</label>
          <select id="audio_top_class">
            <option value="silence">silence</option>
            <option value="speech">speech</option>
            <option value="scream">scream</option>
            <option value="gunshot">gunshot</option>
            <option value="explosion">explosion</option>
            <option value="music">music</option>
          </select>
        </div>
      </div>
    </div>
  </div>

  <div>
    <div class="card">
      <h2>Geospatial</h2>
      <div>
        <span class="toggle"><input type="checkbox" id="is_home_zone" checked> home zone</span>
        <span class="toggle"><input type="checkbox" id="is_work_zone"> work zone</span>
        <span class="toggle"><input type="checkbox" id="is_known_area" checked> known area</span>
        <span class="toggle"><input type="checkbox" id="is_unfamiliar_area"> unfamiliar area</span>
      </div>
      <div class="row" style="margin-top: 6px;">
        <div><label>Distance from home (km)</label><input id="distance_from_home_km" type="number" step="0.1" value="0"></div>
        <div><label>Hour of day (0–23)</label><input id="hour_of_day" type="number" min="0" max="23" value="14"></div>
      </div>
      <div style="margin-top: 8px;">
        <span class="toggle"><input type="checkbox" id="is_night"> night</span>
        <span class="toggle"><input type="checkbox" id="is_stationary" checked> stationary</span>
        <span class="toggle"><input type="checkbox" id="is_walking"> walking</span>
        <span class="toggle"><input type="checkbox" id="is_vehicle_like_motion"> in vehicle</span>
      </div>
      <div style="margin-top: 4px;">
        <span class="toggle"><input type="checkbox" id="sudden_route_change"> sudden route change</span>
        <span class="toggle"><input type="checkbox" id="sudden_stop"> sudden stop</span>
        <span class="toggle"><input type="checkbox" id="phone_connected" checked> phone connected</span>
      </div>
    </div>

    <div class="card" style="margin-top: 16px;">
      <h2>Verdict</h2>
      <div class="verdict">
        <div id="state" class="state s-normal">—</div>
        <div>
          <div><span class="pill" id="risk_pill">risk: —</span><span class="pill" id="conf_pill">confidence —</span></div>
          <div id="signals" class="small"></div>
        </div>
      </div>
      <div><strong>Top alert:</strong> <span id="alert"></span></div>
      <div class="small" id="alert_msg"></div>
      <div style="margin-top: 8px;"><strong>Safety reasoning:</strong>
        <ul id="reasoning"></ul>
      </div>
    </div>

    <div class="card" style="margin-top: 16px;">
      <h2>Full result</h2>
      <pre id="raw">submit to see the verdict</pre>
    </div>
  </div>
</div>

<div style="margin-top: 16px;">
  <button class="classify" onclick="classify()">Classify</button>
</div>

<script>
const FIELDS_NUM  = ['hr','baseline_hr','hrv','baseline_hrv','spo2','baseline_spo2',
                     'temp','baseline_temp','sqi_overall',
                     'acc_mag','dyn_acc_mag',
                     'distance_from_home_km','hour_of_day'];
const FIELDS_BOOL = ['finger_on','audio_enabled',
                     'is_home_zone','is_work_zone','is_known_area','is_unfamiliar_area',
                     'is_night','is_stationary','is_walking','is_vehicle_like_motion',
                     'sudden_route_change','sudden_stop','phone_connected'];
const FIELDS_STR  = ['audio_risk_level','audio_top_class'];

function readForm() {
  const body = {};
  for (const f of FIELDS_NUM)  body[f] = parseFloat(document.getElementById(f).value);
  for (const f of FIELDS_BOOL) body[f] = document.getElementById(f).checked;
  for (const f of FIELDS_STR)  body[f] = document.getElementById(f).value;
  return body;
}
function writeForm(input) {
  for (const f of FIELDS_NUM)  if (f in input) document.getElementById(f).value = input[f];
  for (const f of FIELDS_BOOL) if (f in input) document.getElementById(f).checked = !!input[f];
  for (const f of FIELDS_STR)  if (f in input) document.getElementById(f).value = input[f];
}
async function classify() {
  const body = readForm();
  const r = await fetch('/classify', {method:'POST', headers:{'content-type':'application/json'}, body:JSON.stringify(body)});
  render(await r.json());
}
async function applyPreset(name) {
  const r = await fetch('/preset/' + name, {method:'POST'});
  const j = await r.json();
  writeForm(j.preset_input);
  render(j.verdict);
}
function render(v) {
  if (!v) return;
  const stateEl = document.getElementById('state');
  stateEl.textContent = (v.state || '—').toUpperCase();
  stateEl.className = 'state s-' + (v.state || 'normal');

  const risk = (v.safety_risk_level || 'low');
  const riskPill = document.getElementById('risk_pill');
  riskPill.textContent = 'risk: ' + risk;
  riskPill.className = 'pill r-' + risk;

  document.getElementById('conf_pill').textContent =
    'confidence ' + Number(v.confidence || 0).toFixed(2);

  document.getElementById('signals').textContent =
    'signals: ' + (v.contributing_signals?.length ? v.contributing_signals.join(', ') : '—');

  const a = v.top_alert || {};
  document.getElementById('alert').textContent = a.title || '—';
  document.getElementById('alert_msg').textContent = a.message || '';

  const ul = document.getElementById('reasoning');
  ul.innerHTML = '';
  (v.safety_reasoning || []).forEach(t => { const li = document.createElement('li'); li.textContent = t; ul.appendChild(li); });

  document.getElementById('raw').textContent = JSON.stringify(v.full || v, null, 2);
}
</script>
</body></html>
"""
