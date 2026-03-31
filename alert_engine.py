"""
alert_engine.py — Context-Aware Medical Alert Engine
Generates clinically meaningful alerts with cooldown and sustained-reading filters.
"""

import time
from dataclasses import dataclass
from typing import Optional
from collections import deque

ALERT_COOLDOWN_MIN: int = 10
_COOLDOWN_SECS: int = ALERT_COOLDOWN_MIN * 60
SUSTAINED_READINGS_REQUIRED: int = 3


@dataclass
class MedicalAlert:
    patient_id:      str
    severity:        str
    event:           str
    description:     str
    vitals_snapshot: dict

    def to_dict(self) -> dict:
        return {
            "patient_id":  self.patient_id,
            "severity":    self.severity,
            "event":       self.event,
            "description": self.description,
            "vitals":      self.vitals_snapshot,
        }


class PatientTrendTracker:
    def __init__(self, window: int = 5):
        self.window = window
        self._history: deque = deque(maxlen=window)

    def push(self, vitals: dict) -> None:
        self._history.append(vitals)

    def trend(self, key: str) -> Optional[float]:
        vals = [v[key] for v in self._history if key in v]
        if len(vals) < 2:
            return None
        deltas = [vals[i+1] - vals[i] for i in range(len(vals)-1)]
        return sum(deltas) / len(deltas)

    def latest(self, key: str, default=None):
        if not self._history:
            return default
        return self._history[-1].get(key, default)

    def mean(self, key: str, default=None):
        vals = [v[key] for v in self._history if key in v]
        if not vals:
            return default
        return sum(vals) / len(vals)


def _check_tachycardia(pid, tracker, vitals):
    hr = vitals.get("heart_rate", 0)
    rr = vitals.get("respiratory_rate", 0)
    rr_trend = tracker.trend("respiratory_rate") or 0.0
    if hr > 120 and rr_trend > 0.2:
        return MedicalAlert(pid, "HIGH", "Tachycardia + Rising RR",
            f"HR {hr:.0f} bpm with RR trending upward (+{rr_trend:.2f}/step). "
            "May indicate compensatory response to reduced cardiac output.",
            {"hr": hr, "rr": rr})
    if hr > 130:
        return MedicalAlert(pid, "HIGH", "Severe Tachycardia",
            f"HR {hr:.0f} bpm. Persistent HR >130 requires urgent evaluation.",
            {"hr": hr})
    return None


def _check_hypoxia(pid, tracker, vitals):
    spo2 = vitals.get("oxygen_saturation", 100)
    hr   = vitals.get("heart_rate", 70)
    spo2_trend = tracker.trend("oxygen_saturation") or 0.0
    hr_trend   = tracker.trend("heart_rate") or 0.0
    if spo2 < 90:
        return MedicalAlert(pid, "CRITICAL", "Critical Hypoxia",
            f"SpO₂ {spo2:.1f}% — below 90%. Immediate O₂ and airway assessment required.",
            {"spo2": spo2, "hr": hr})
    if spo2_trend < -0.3 and hr_trend > 0.3:
        return MedicalAlert(pid, "HIGH", "Possible Hypoxia",
            f"SpO₂ falling ({spo2_trend:.2f}/step) with compensatory tachycardia.",
            {"spo2": spo2, "hr": hr})
    return None


def _check_respiratory_distress(pid, tracker, vitals):
    rr   = vitals.get("respiratory_rate", 16)
    spo2 = vitals.get("oxygen_saturation", 98)
    if rr > 25 and spo2 < 94:
        return MedicalAlert(pid, "CRITICAL", "Respiratory Distress",
            f"RR {rr:.0f}/min with SpO₂ {spo2:.1f}%. Urgent review required.",
            {"rr": rr, "spo2": spo2})
    if rr > 24:
        return MedicalAlert(pid, "WARNING", "Tachypnoea",
            f"Respiratory rate {rr:.0f}/min (normal 12-20).",
            {"rr": rr})
    return None


def _check_fever(pid, tracker, vitals):
    temp = vitals.get("body_temperature", 37.0)
    hr   = vitals.get("heart_rate", 70)
    if temp > 39.0 and hr > 100:
        return MedicalAlert(pid, "HIGH", "Fever with Tachycardia",
            f"Temp {temp:.2f}°C with HR {hr:.0f} bpm. Investigate for sepsis.",
            {"temp": temp, "hr": hr})
    if temp > 38.0:
        return MedicalAlert(pid, "WARNING", "Possible Fever",
            f"Temperature {temp:.2f}°C. Monitor for escalation.",
            {"temp": temp})
    if temp < 36.0:
        return MedicalAlert(pid, "WARNING", "Hypothermia Warning",
            f"Temperature {temp:.2f}°C — below normal. Consider sepsis.",
            {"temp": temp})
    return None


def _check_hypotension(pid, tracker, vitals):
    sbp = vitals.get("systolic_bp", 120)
    hr  = vitals.get("heart_rate", 70)
    if sbp < 90:
        return MedicalAlert(pid, "CRITICAL", "Hypotensive Shock",
            f"Systolic BP {sbp:.0f} mmHg. IV fluids and urgent review required.",
            {"sbp": sbp, "hr": hr})
    if sbp < 100 and hr > 100:
        return MedicalAlert(pid, "HIGH", "Hypotension + Tachycardia",
            f"BP {sbp:.0f} mmHg with HR {hr:.0f} bpm. Possible early shock.",
            {"sbp": sbp, "hr": hr})
    return None


def _check_hrv_collapse(pid, tracker, vitals):
    hrv      = vitals.get("hrv", 0.07)
    hrv_mean = tracker.mean("hrv", hrv)
    if hrv < 0.02:
        return MedicalAlert(pid, "HIGH", "HRV Collapse",
            f"HRV {hrv:.4f} — critically low autonomic variability.",
            {"hrv": hrv})
    if hrv_mean > 0 and hrv < hrv_mean * 0.5 and hrv < 0.04:
        return MedicalAlert(pid, "WARNING", "Acute HRV Reduction",
            f"HRV dropped to {hrv:.4f} (mean {hrv_mean:.4f}).",
            {"hrv": hrv})
    return None


def _check_hypertensive_crisis(pid, tracker, vitals):
    sbp = vitals.get("systolic_bp", 120)
    dbp = vitals.get("diastolic_bp", 80)
    if sbp > 180 or dbp > 110:
        return MedicalAlert(pid, "CRITICAL", "Hypertensive Crisis",
            f"BP {sbp:.0f}/{dbp:.0f} mmHg. Risk of stroke/MI. Urgent management.",
            {"sbp": sbp, "dbp": dbp})
    return None


def _check_critical_combination(pid, tracker, vitals):
    flags, issues = 0, []
    hr   = vitals.get("heart_rate", 70)
    spo2 = vitals.get("oxygen_saturation", 98)
    rr   = vitals.get("respiratory_rate", 16)
    sbp  = vitals.get("systolic_bp", 120)
    temp = vitals.get("body_temperature", 37)
    if hr > 120 or hr < 45:   flags += 1; issues.append(f"HR {hr:.0f}")
    if spo2 < 92:              flags += 1; issues.append(f"SpO₂ {spo2:.1f}%")
    if rr > 24 or rr < 8:     flags += 1; issues.append(f"RR {rr:.0f}")
    if sbp < 90 or sbp > 180: flags += 1; issues.append(f"BP {sbp:.0f}")
    if temp > 39.5 or temp < 35.5: flags += 1; issues.append(f"Temp {temp:.1f}°C")
    if flags >= 3:
        return MedicalAlert(pid, "CRITICAL", "Multi-Organ Deterioration",
            f"{flags} vitals simultaneously abnormal: {', '.join(issues)}. "
            "Immediate ICU escalation.",
            vitals)
    return None


_ALERT_CHECKERS = [
    _check_critical_combination,
    _check_hypotension,
    _check_hypertensive_crisis,
    _check_hypoxia,
    _check_respiratory_distress,
    _check_tachycardia,
    _check_fever,
    _check_hrv_collapse,
]

SEVERITY_ORDER = {"CRITICAL": 0, "HIGH": 1, "WARNING": 2, "INFO": 3}

_trackers:          dict = {}
_alert_last_fired:  dict = {}
_consecutive_count: dict = {}


def _is_on_cooldown(patient_id, event):
    key  = (patient_id, event)
    last = _alert_last_fired.get(key)
    if last is None:
        return False
    return (time.monotonic() - last) < _COOLDOWN_SECS


def _increment_consecutive(patient_id, event):
    key = (patient_id, event)
    _consecutive_count[key] = _consecutive_count.get(key, 0) + 1
    return _consecutive_count[key]


def _mark_fired(patient_id, event):
    _alert_last_fired[(patient_id, event)] = time.monotonic()
    _consecutive_count.pop((patient_id, event), None)


def evaluate_alerts(patient_id, vitals: dict) -> list:
    if patient_id not in _trackers:
        _trackers[patient_id] = PatientTrendTracker(window=6)

    tracker = _trackers[patient_id]
    tracker.push(vitals)

    candidate_alerts = []
    for checker in _ALERT_CHECKERS:
        try:
            alert = checker(patient_id, tracker, vitals)
            if alert is not None:
                candidate_alerts.append(alert)
        except Exception:
            pass

    filtered     = []
    fired_events = set()

    for a in sorted(candidate_alerts, key=lambda x: SEVERITY_ORDER.get(x.severity, 99)):
        if a.event in fired_events:
            continue
        if _is_on_cooldown(patient_id, a.event):
            continue
        count       = _increment_consecutive(patient_id, a.event)
        min_readings = 1 if a.severity == "CRITICAL" else SUSTAINED_READINGS_REQUIRED
        if count >= min_readings:
            filtered.append(a)
            fired_events.add(a.event)
            _mark_fired(patient_id, a.event)

    all_fired_events = {a.event for a in candidate_alerts}
    for key in list(_consecutive_count.keys()):
        kid, kevent = key
        if kid == patient_id and kevent not in all_fired_events:
            _consecutive_count.pop(key, None)

    return filtered
