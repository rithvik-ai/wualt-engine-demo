"""
WUALT v1 API — burst-ingestion baseline learner + on-demand SOS evaluator.

Manual-trigger-only mode. The engine NO LONGER fires autonomous alerts.
It consumes biosignal bursts for baseline learning, and only scores +
routes when the user pushes the panic button.

Endpoints
---------
    POST   /v1/biosignals/ingest         — burst ingestion (baseline learner)
    GET    /v1/user/{user_id}/baseline   — current learned baseline
    POST   /v1/sos/evaluate              — panic button pressed; score + route
    POST   /v1/sos/confirm               — user response (CONFIRM/CANCEL/PIN)
    GET    /v1/sos/status/{user_id}      — current SOS state
    POST   /v1/sos/reset                 — clear SOS state
    GET    /v1/health                    — liveness check

Wire into your main app:

    from wualt_api_v1 import router as v1_router
    app.include_router(v1_router)
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Tuple

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

import distress_engine as de


# ─────────────────────────────────────────────────────────────────────────────
# Baseline learner — Welford's online algorithm, stratified by activity
# ─────────────────────────────────────────────────────────────────────────────

SIGNALS = ("hr", "hrv", "spo2", "temp")
ACTIVITIES = ("still", "walking", "exercise")

# Frames outside these physiological bounds are rejected at ingest.
PHYS_BOUNDS = {
    "hr":   (30.0, 220.0),
    "hrv":  (5.0, 200.0),
    "spo2": (70.0, 100.0),
    "temp": (33.0, 42.0),
}

# Baseline is "ready" once we have this many still-state HR samples.
# 1800 frames at 1 Hz ≈ 30 minutes of clean still data.
BASELINE_READY_MIN_FRAMES = 1800


def classify_activity(dyn_acc_mag: float) -> str:
    if dyn_acc_mag < 0.05:
        return "still"
    if dyn_acc_mag < 0.30:
        return "walking"
    return "exercise"


class WelfordStat:
    """Online (mean, variance). Numerically stable for streaming data."""

    __slots__ = ("n", "mean", "M2")

    def __init__(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, x: float) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    @property
    def std(self) -> float:
        return (self.M2 / (self.n - 1)) ** 0.5 if self.n > 1 else 0.0


class BaselineLearner:
    """Per-user baseline tracker — overall + stratified-by-activity."""

    def __init__(self) -> None:
        self.frames_seen_total = 0
        self.first_seen_ts: Optional[float] = None
        self.last_updated_ts: Optional[float] = None
        self.overall: Dict[str, WelfordStat] = {s: WelfordStat() for s in SIGNALS}
        self.by_activity: Dict[str, Dict[str, WelfordStat]] = {
            a: {s: WelfordStat() for s in SIGNALS} for a in ACTIVITIES
        }

    def update(self, frame: "BiosignalFrame") -> None:
        activity = classify_activity(frame.dyn_acc_mag)
        for sig in SIGNALS:
            value = getattr(frame, sig)
            self.overall[sig].update(value)
            self.by_activity[activity][sig].update(value)
        self.frames_seen_total += 1
        now = time.time()
        if self.first_seen_ts is None:
            self.first_seen_ts = now
        self.last_updated_ts = now

    @property
    def ready(self) -> bool:
        return self.by_activity["still"]["hr"].n >= BASELINE_READY_MIN_FRAMES

    @property
    def days_of_data(self) -> float:
        if self.first_seen_ts is None or self.last_updated_ts is None:
            return 0.0
        return round((self.last_updated_ts - self.first_seen_ts) / 86400.0, 2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ready": self.ready,
            "frames_seen_total": self.frames_seen_total,
            "days_of_data": self.days_of_data,
            "hr_mean":   round(self.overall["hr"].mean, 2),
            "hr_std":    round(self.overall["hr"].std, 2),
            "hrv_mean":  round(self.overall["hrv"].mean, 2),
            "hrv_std":   round(self.overall["hrv"].std, 2),
            "spo2_mean": round(self.overall["spo2"].mean, 2),
            "spo2_std":  round(self.overall["spo2"].std, 2),
            "temp_mean": round(self.overall["temp"].mean, 2),
            "temp_std":  round(self.overall["temp"].std, 2),
            "by_activity": {
                a: {
                    "hr_mean": round(self.by_activity[a]["hr"].mean, 2),
                    "hr_std":  round(self.by_activity[a]["hr"].std, 2),
                    "n":       self.by_activity[a]["hr"].n,
                }
                for a in ACTIVITIES
            },
            "last_updated": _iso(self.last_updated_ts) if self.last_updated_ts else None,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Per-user state (swap for Redis in production)
# ─────────────────────────────────────────────────────────────────────────────

_BASELINES: Dict[str, BaselineLearner] = {}
_SOS_HANDLERS: Dict[str, de.ManualTriggerHandler] = {}
_ENGINES: Dict[str, de.UnifiedSafetyEngine] = {}
_STARTED_AT = time.time()


def _get_baseline(user_id: str) -> BaselineLearner:
    if user_id not in _BASELINES:
        _BASELINES[user_id] = BaselineLearner()
    return _BASELINES[user_id]


def _get_sos_handler(user_id: str) -> de.ManualTriggerHandler:
    if user_id not in _SOS_HANDLERS:
        _SOS_HANDLERS[user_id] = de.ManualTriggerHandler()
    return _SOS_HANDLERS[user_id]


def _get_engine(user_id: str) -> de.UnifiedSafetyEngine:
    if user_id not in _ENGINES:
        _ENGINES[user_id] = de.UnifiedSafetyEngine()
    return _ENGINES[user_id]


def _iso(ts: Optional[float]) -> str:
    if ts is None:
        return ""
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")


# ─────────────────────────────────────────────────────────────────────────────
# Request / response schemas
# ─────────────────────────────────────────────────────────────────────────────

class BiosignalFrame(BaseModel):
    ts: Optional[str] = None
    hr: float = Field(..., ge=0, le=300)
    hrv: float = Field(..., ge=0, le=300)
    spo2: float = Field(..., ge=0, le=100)
    temp: float = Field(..., ge=20, le=45)
    sqi: float = Field(default=0.85, ge=0, le=1)
    finger_on: bool = True
    acc_mag: float = 0.98
    dyn_acc_mag: float = 0.0


class IngestBurstWindow(BaseModel):
    start: Optional[str] = None
    end: Optional[str] = None
    frame_rate_hz: float = 1.0


class IngestRequest(BaseModel):
    user_id: str
    device_id: Optional[str] = None
    burst_window: Optional[IngestBurstWindow] = None
    frames: List[BiosignalFrame]


class SosContext(BaseModel):
    is_home_zone: bool = False
    is_work_zone: bool = False
    is_known_area: bool = False
    is_unfamiliar_area: bool = False
    distance_from_home_km: float = 0.0
    hour_of_day: int = 14
    is_night: bool = False
    is_stationary: bool = True
    is_walking: bool = False
    is_vehicle_like_motion: bool = False
    sudden_route_change: bool = False
    sudden_stop: bool = False
    phone_connected: bool = True
    audio_enabled: bool = False
    audio_risk_level: Literal["normal", "elevated", "danger"] = "normal"
    audio_top_class: str = "silence"


class SosPolicy(BaseModel):
    contact_threshold: float = 0.50
    police_threshold: float = 0.85
    emergency_contact_name: str = ""
    emergency_contact_phone: str = ""


class SosEvaluateRequest(BaseModel):
    user_id: str
    trigger_source: Literal["ring_button", "app_button", "watch_button"] = "ring_button"
    triggered_at: Optional[str] = None
    current_frame: BiosignalFrame
    recent_window_s: int = 60
    recent_frames: List[BiosignalFrame] = Field(default_factory=list)
    context: SosContext = Field(default_factory=SosContext)
    policy: SosPolicy


class SosConfirmRequest(BaseModel):
    user_id: str
    action: Literal["CONFIRM", "CANCEL", "PIN_ENTRY"]
    pin: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI router
# ─────────────────────────────────────────────────────────────────────────────

router = APIRouter(prefix="/v1", tags=["wualt-v1"])


@router.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "engine_version": "1.0.0",
        "model_version": "lgbm-v3-2026-04-22",
        "uptime_s": int(time.time() - _STARTED_AT),
        "active_users": len(_BASELINES),
    }


# ── Passive path — burst ingestion (baseline learning only) ─────────────────

@router.post("/biosignals/ingest")
def ingest_burst(req: IngestRequest) -> Dict[str, Any]:
    """Burst ingest. NO classification, NO alert. Updates per-user baseline."""
    learner = _get_baseline(req.user_id)
    drop_reasons = {"low_sqi": 0, "finger_off": 0, "outside_physiological_range": 0}
    accepted = 0

    for f in req.frames:
        if f.sqi < 0.30:
            drop_reasons["low_sqi"] += 1
            continue
        if not f.finger_on:
            drop_reasons["finger_off"] += 1
            continue
        if not all(PHYS_BOUNDS[s][0] <= getattr(f, s) <= PHYS_BOUNDS[s][1] for s in SIGNALS):
            drop_reasons["outside_physiological_range"] += 1
            continue
        learner.update(f)
        accepted += 1

    return {
        "user_id": req.user_id,
        "burst_id": f"brst_{_iso(time.time())}_{req.user_id[:8]}",
        "frames_accepted": accepted,
        "frames_dropped": len(req.frames) - accepted,
        "drop_reasons": drop_reasons,
        "baseline": learner.to_dict(),
    }


@router.get("/user/{user_id}/baseline")
def get_baseline(user_id: str) -> Dict[str, Any]:
    if user_id not in _BASELINES:
        raise HTTPException(404, f"no baseline for user_id={user_id}")
    return _BASELINES[user_id].to_dict()


# ── Active path — panic-button scoring + recipient routing ──────────────────

@router.post("/sos/evaluate")
def sos_evaluate(req: SosEvaluateRequest) -> Dict[str, Any]:
    """Panic button pressed. Score current snapshot against learned baseline."""
    learner = _get_baseline(req.user_id)
    engine = _get_engine(req.user_id)

    if learner.ready:
        bl_hr   = learner.overall["hr"].mean
        bl_hrv  = learner.overall["hrv"].mean
        bl_spo2 = learner.overall["spo2"].mean
        bl_temp = learner.overall["temp"].mean
        baseline_source = "learned"
    else:
        bl_hr, bl_hrv, bl_spo2, bl_temp = 72.0, 55.0, 98.0, 36.6
        baseline_source = "population_default"

    f = req.current_frame
    pipeline_output = _build_pipeline_output(f, bl_hr, bl_hrv, bl_spo2, bl_temp)
    geo_context = _build_geo_context(req.context)
    audio_dict = _fake_audio_result(req.context) if req.context.audio_enabled else None
    engine.audio.evaluate = lambda audio_input=None: audio_dict  # type: ignore

    result = engine.evaluate(
        pipeline_output=pipeline_output,
        geo_context=geo_context,
        audio_input={"audio_array": [], "sample_rate": 16000} if req.context.audio_enabled else None,
    )

    debug = result.get("debug", {})
    weighted_score = float(debug.get("weighted_score", 0.0))
    distress_confidence = round(max(0.0, min(1.0, weighted_score)), 3)

    recipient = de.ManualTriggerHandler.decide_recipient(
        engine_confidence=distress_confidence,
        contact_threshold=req.policy.contact_threshold,
        police_threshold=req.policy.police_threshold,
        emergency_contact={
            "name":  req.policy.emergency_contact_name,
            "phone": req.policy.emergency_contact_phone,
        },
    )

    # Record the press in the per-user SOS state machine.
    sos = _get_sos_handler(req.user_id)
    try:
        sos.on_button_press(distress_confidence=distress_confidence)
    except TypeError:
        # Fallback if ManualTriggerHandler.on_button_press has a different sig.
        sos.on_button_press()

    ui_action, countdown_s, prompt_text, requires_pin = _ui_decision(
        recipient, distress_confidence, req.policy
    )

    return {
        "decision_id":         f"dec_{_iso(time.time())}_{req.user_id[:8]}",
        "user_id":             req.user_id,
        "decided_at":          _iso(time.time()),
        "distress_confidence": distress_confidence,
        "weighted_score":      round(weighted_score, 3),
        "contributing_signals": _signals_with_baseline(f, bl_hr, bl_hrv, bl_spo2, bl_temp),
        "context_modifiers":   result.get("safety", {}).get("reasoning", []),
        "motion_state":        debug.get("motion_state"),
        "safety_risk_level":   result.get("safety", {}).get("risk_level"),
        "recipient":           recipient,
        "ui_action":           ui_action,
        "countdown_s":         countdown_s,
        "prompt_text":         prompt_text,
        "requires_pin_to_cancel": requires_pin,
        "baseline_used": {
            "source":       baseline_source,
            "ready":        learner.ready,
            "days_of_data": learner.days_of_data,
            "hr_mean":      round(bl_hr, 2),
            "hrv_mean":     round(bl_hrv, 2),
            "spo2_mean":    round(bl_spo2, 2),
            "temp_mean":    round(bl_temp, 2),
        },
    }


@router.post("/sos/confirm")
def sos_confirm(req: SosConfirmRequest) -> Dict[str, Any]:
    sos = _get_sos_handler(req.user_id)
    try:
        sos.on_user_action(action=req.action, pin=req.pin)
    except TypeError:
        sos.on_user_action(req.action)
    status = sos.status() if hasattr(sos, "status") else {}
    return {
        "user_id":   req.user_id,
        "sos_state": status.get("sos_state"),
        "recipient_notified": status.get("recipient_notified"),
    }


@router.get("/sos/status/{user_id}")
def sos_status(user_id: str) -> Dict[str, Any]:
    sos = _get_sos_handler(user_id)
    status = sos.status() if hasattr(sos, "status") else {}
    return {"user_id": user_id, **status}


@router.post("/sos/reset")
def sos_reset(user_id: str) -> Dict[str, Any]:
    sos = _get_sos_handler(user_id)
    if hasattr(sos, "reset"):
        sos.reset()
    return {"user_id": user_id, "reset": True}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — adapt new schemas to the existing engine's input shape
# ─────────────────────────────────────────────────────────────────────────────

_STD = {"hr": 5.0, "hrv": 8.0, "spo2": 0.8, "temp": 0.15}


def _zscore(current: float, baseline: float, std: float) -> float:
    return round((current - baseline) / max(std, 1e-6), 3)


def _build_pipeline_output(
    f: BiosignalFrame,
    bl_hr: float, bl_hrv: float, bl_spo2: float, bl_temp: float,
) -> Dict[str, Any]:
    ts = int(time.time())
    sqi = max(0.30, f.sqi)
    sample = {
        "timestamp": ts, "sequence": ts,
        "hr": f.hr, "hr_stability_score": f.hrv,
        "temp": f.temp, "temp_raw": f.temp,
        "spo2": f.spo2,
        "acc_mag": f.acc_mag, "dyn_acc_mag": f.dyn_acc_mag,
        "acc_x": 0.02, "acc_y": -0.01, "acc_z": 0.98,
        "gravity_x": 0.02, "gravity_y": -0.01, "gravity_z": 0.98,
        "finger_on": f.finger_on, "charging": False,
        "battery_mv": 3850, "die_temp": 34.5, "adc_raw": 112000,
        "thermal_bias": 0.0,
        "sqi": {
            "hr":   sqi, "hrv":  max(0.6, sqi - 0.1),
            "temp": max(0.8, sqi), "acc":  0.95,
            "spo2": sqi, "ppg":  max(0.7, sqi - 0.05),
            "overall": sqi,
        },
        "accepted": True, "reject_reasons": [], "clinical_flags": [],
    }
    zscores = {
        "hr":                 _zscore(f.hr,   bl_hr,   _STD["hr"]),
        "hr_stability_score": _zscore(f.hrv,  bl_hrv,  _STD["hrv"]),
        "spo2":               _zscore(f.spo2, bl_spo2, _STD["spo2"]),
        "temp":               _zscore(f.temp, bl_temp, _STD["temp"]),
        "acc_mag":            0.0,
    }
    window = {
        "window_n": 25,
        "hr_mean": f.hr, "hr_var": 2.0, "hr_min": f.hr - 2, "hr_max": f.hr + 2,
        "hr_stability_score_mean": f.hrv, "hr_stability_score_var": 4.0,
        "hr_stability_score_min": f.hrv - 4, "hr_stability_score_max": f.hrv + 4,
        "temp_mean": f.temp, "temp_var": 0.01,
        "temp_min": f.temp - 0.1, "temp_max": f.temp + 0.1,
        "spo2_mean": f.spo2, "spo2_var": 0.5,
        "spo2_min": f.spo2 - 0.5, "spo2_max": f.spo2 + 0.5,
        "acc_mag_mean": f.acc_mag, "acc_mag_var": 0.001,
        "acc_mag_min": f.acc_mag - 0.01, "acc_mag_max": f.acc_mag + 0.01,
    }
    return {"sample": sample, "zscores": zscores, "baseline_ready": True, "window": window}


def _build_geo_context(ctx: SosContext) -> Dict[str, Any]:
    return {
        "latitude": 0.0, "longitude": 0.0, "timestamp": int(time.time()),
        "speed_kmph": 0.0, "heading": 0.0,
        "is_home_zone": ctx.is_home_zone,
        "is_work_zone": ctx.is_work_zone,
        "is_known_area": ctx.is_known_area,
        "is_unfamiliar_area": ctx.is_unfamiliar_area,
        "distance_from_home_km": ctx.distance_from_home_km,
        "hour_of_day": ctx.hour_of_day,
        "is_night": ctx.is_night,
        "is_stationary": ctx.is_stationary,
        "is_walking": ctx.is_walking,
        "is_vehicle_like_motion": ctx.is_vehicle_like_motion,
        "sudden_route_change": ctx.sudden_route_change,
        "sudden_stop": ctx.sudden_stop,
        "phone_connected": ctx.phone_connected,
        "phone_disconnect_duration_s": 0.0,
    }


def _fake_audio_result(ctx: SosContext) -> Optional[Dict[str, Any]]:
    if not ctx.audio_enabled:
        return None
    risk_map = {"normal": 0.10, "elevated": 0.40, "danger": 0.78}
    sev_map  = {"normal": "low", "elevated": "medium", "danger": "high"}
    return {
        "audio_risk_level": ctx.audio_risk_level,
        "audio_risk_score": risk_map.get(ctx.audio_risk_level, 0.10),
        "top_class":        ctx.audio_top_class,
        "environment": {
            "danger_score": risk_map.get(ctx.audio_risk_level, 0.10),
            "top_class":    ctx.audio_top_class,
        },
        "vocal_stress": {"stress_score": 0.5 if ctx.audio_risk_level != "normal" else 0.1},
        "noise":        {"noise_score":  0.4 if ctx.audio_risk_level == "danger" else 0.15},
        "alert": {
            "title":    f"Audio: {ctx.audio_top_class}",
            "message":  f"Detected {ctx.audio_top_class} ({ctx.audio_risk_level})",
            "severity": sev_map.get(ctx.audio_risk_level, "low"),
        },
    }


def _signals_with_baseline(
    f: BiosignalFrame,
    bl_hr: float, bl_hrv: float, bl_spo2: float, bl_temp: float,
) -> List[Dict[str, Any]]:
    return [
        {"signal": "hr",   "z": _zscore(f.hr,   bl_hr,   _STD["hr"]),   "weight": 0.35,
         "vs_baseline": f"{f.hr} vs {round(bl_hr, 1)}"},
        {"signal": "hrv",  "z": _zscore(f.hrv,  bl_hrv,  _STD["hrv"]),  "weight": 0.25,
         "vs_baseline": f"{f.hrv} vs {round(bl_hrv, 1)}"},
        {"signal": "spo2", "z": _zscore(f.spo2, bl_spo2, _STD["spo2"]), "weight": 0.30,
         "vs_baseline": f"{f.spo2} vs {round(bl_spo2, 1)}"},
        {"signal": "temp", "z": _zscore(f.temp, bl_temp, _STD["temp"]), "weight": 0.10,
         "vs_baseline": f"{f.temp} vs {round(bl_temp, 2)}"},
    ]


def _ui_decision(
    recipient: Dict[str, Any],
    confidence: float,
    policy: SosPolicy,
) -> Tuple[str, int, str, bool]:
    target = recipient.get("target", "ask_user")
    if target == "police":
        return (
            "SHOW_PIN_DIALOG",
            15,
            "Police will be dispatched in 15s. Enter PIN to cancel.",
            True,
        )
    if target == "contact":
        return (
            "SHOW_CONFIRM_DIALOG",
            30,
            f"Are you OK? Tap CONFIRM to alert {recipient.get('label','your contact')}, or CANCEL with PIN.",
            False,
        )
    return (
        "SHOW_ASK_USER_DIALOG",
        60,
        "Do you need help? Confirm to alert your contact, or tap CANCEL if you're OK.",
        False,
    )
