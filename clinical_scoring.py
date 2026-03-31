"""
clinical_scoring.py — NEWS2 and MEWS Clinical Early Warning Score Engine
"""

from dataclasses import dataclass
from typing import Literal

RiskLevel = Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]

RISK_COLOR = {
    "LOW":      "#00c853",
    "MEDIUM":   "#ffab00",
    "HIGH":     "#ff6d00",
    "CRITICAL": "#d50000",
}

RISK_EMOJI = {
    "LOW":      "🟢",
    "MEDIUM":   "🟡",
    "HIGH":     "🟠",
    "CRITICAL": "🔴",
}

RISK_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}


@dataclass
class ClinicalScore:
    news2:      int
    mews:       int
    composite:  float
    risk_level: RiskLevel
    breakdown:  dict


def _news2_rr(rr):
    if rr <= 8:   return 3
    if rr <= 11:  return 1
    if rr <= 20:  return 0
    if rr <= 24:  return 2
    return 3

def _news2_spo2(spo2):
    if spo2 >= 96: return 0
    if spo2 >= 94: return 1
    if spo2 >= 92: return 2
    return 3

def _news2_bp(systolic):
    if systolic <= 90:  return 3
    if systolic <= 100: return 2
    if systolic <= 110: return 1
    if systolic <= 219: return 0
    return 3

def _news2_hr(hr):
    if hr <= 40:  return 3
    if hr <= 50:  return 1
    if hr <= 90:  return 0
    if hr <= 110: return 1
    if hr <= 130: return 2
    return 3

def _news2_temp(temp):
    if temp <= 35.0: return 3
    if temp <= 36.0: return 1
    if temp <= 38.0: return 0
    if temp <= 39.0: return 1
    return 2

def _news2_consciousness(avpu="A"):
    return 0 if avpu == "A" else 3

def _mews_rr(rr):
    if rr < 9:    return 2
    if rr <= 14:  return 0
    if rr <= 20:  return 1
    if rr <= 29:  return 2
    return 3

def _mews_hr(hr):
    if hr < 40:   return 2
    if hr <= 50:  return 1
    if hr <= 100: return 0
    if hr <= 110: return 1
    if hr <= 129: return 2
    return 3

def _mews_bp(systolic):
    if systolic < 70:   return 3
    if systolic < 80:   return 2
    if systolic < 100:  return 1
    if systolic <= 199: return 0
    return 2

def _mews_temp(temp):
    if temp < 35.0:  return 2
    if temp <= 38.4: return 0
    return 2

def _hrv_penalty(hrv):
    if hrv < 0.015: return 3
    if hrv < 0.025: return 2
    if hrv < 0.035: return 1
    return 0

def _news2_to_risk(score):
    if score == 0:  return "LOW"
    if score <= 4:  return "LOW"
    if score <= 6:  return "MEDIUM"
    if score <= 8:  return "HIGH"
    return "CRITICAL"

def _composite_to_risk(composite):
    if composite < 20: return "LOW"
    if composite < 45: return "MEDIUM"
    if composite < 70: return "HIGH"
    return "CRITICAL"


def compute_clinical_score(
    hr: float,
    rr: float,
    spo2: float,
    temp: float,
    systolic: float,
    hrv: float = 0.07,
    avpu: str = "A",
) -> ClinicalScore:
    n_rr   = _news2_rr(rr)
    n_spo2 = _news2_spo2(spo2)
    n_bp   = _news2_bp(systolic)
    n_hr   = _news2_hr(hr)
    n_temp = _news2_temp(temp)
    n_avpu = _news2_consciousness(avpu)
    news2  = n_rr + n_spo2 + n_bp + n_hr + n_temp + n_avpu

    m_rr   = _mews_rr(rr)
    m_hr   = _mews_hr(hr)
    m_bp   = _mews_bp(systolic)
    m_temp = _mews_temp(temp)
    mews   = m_rr + m_hr + m_bp + m_temp

    hrv_pen = _hrv_penalty(hrv)

    composite = (
        (news2 / 20.0) * 60.0 +
        (mews  / 14.0) * 30.0 +
        (hrv_pen / 3.0) * 10.0
    )
    composite = min(composite, 100.0)

    r_news2     = _news2_to_risk(news2)
    r_composite = _composite_to_risk(composite)
    risk_level  = r_news2 if RISK_ORDER[r_news2] <= RISK_ORDER[r_composite] else r_composite

    breakdown = {
        "news2_rr":    n_rr,
        "news2_spo2":  n_spo2,
        "news2_bp":    n_bp,
        "news2_hr":    n_hr,
        "news2_temp":  n_temp,
        "mews":        mews,
        "hrv_penalty": hrv_pen,
    }

    return ClinicalScore(
        news2=news2,
        mews=mews,
        composite=round(composite, 2),
        risk_level=risk_level,
        breakdown=breakdown,
    )
