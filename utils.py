"""
utils.py — Shared utilities for Swasthya Sense
"""

import os
import logging
import numpy as np

from config import PHYSIO_LIMITS, PHYSIO_DELTA, PHYSIO_NOISE, VITAL_FEATURES


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s",
                              datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def compute_pulse_pressure(systolic: float, diastolic: float) -> float:
    return systolic - diastolic


def compute_map(systolic: float, diastolic: float) -> float:
    return (systolic + 2 * diastolic) / 3.0


def compute_bmi(weight_kg: float, height_m: float) -> float:
    if height_m <= 0:
        return 0.0
    return weight_kg / (height_m ** 2)


def apply_physio_constraints(prev_vitals: dict, raw_next: dict) -> dict:
    constrained = {}
    for feat in VITAL_FEATURES:
        prev  = prev_vitals[feat]
        raw   = raw_next[feat]
        delta = PHYSIO_DELTA[feat]
        lo, hi = PHYSIO_LIMITS[feat]
        clamped = float(np.clip(raw, prev - delta, prev + delta))
        noise   = float(np.random.normal(0, PHYSIO_NOISE[feat]))
        val     = clamped + noise
        constrained[feat] = float(np.clip(val, lo, hi))
    return constrained


def score_to_severity(score: float) -> str:
    from config import THRESH_LOW, THRESH_MED
    if score < THRESH_LOW:
        return "LOW"
    elif score < THRESH_MED:
        return "MEDIUM"
    return "HIGH"


def ensure_model_dir() -> None:
    from config import MODEL_DIR
    os.makedirs(MODEL_DIR, exist_ok=True)
