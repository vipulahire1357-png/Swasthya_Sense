"""
config.py — Swasthya Sense Central Configuration
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── Database ──────────────────────────────────────────────────────────────────
DATABASE_URL: str = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:postgresql%40123@localhost:5432/vitals_db"
)

# ── Kafka ─────────────────────────────────────────────────────────────────────
KAFKA_BOOTSTRAP: list = ["localhost:9092"]
KAFKA_TOPIC: str = "vitals"

# ── Model paths ───────────────────────────────────────────────────────────────
MODEL_DIR: str = "models"
VITAL_FORECASTER_PATH: str = f"{MODEL_DIR}/vital_forecaster.keras"
ANOMALY_DETECTOR_PATH: str = f"{MODEL_DIR}/anomaly_detector.pkl"
RISK_PREDICTOR_PATH: str   = f"{MODEL_DIR}/risk_predictor.pkl"
SCALER_VITALS_PATH: str    = f"{MODEL_DIR}/scaler_vitals.pkl"
SCALER_FULL_PATH: str      = f"{MODEL_DIR}/scaler_full.pkl"

# ── Dataset ───────────────────────────────────────────────────────────────────
DATASET_ZIP: str = "vitals_dataset_2024.zip"
DATASET_CSV: str = "vitals_dataset_2024.csv"

# ── LSTM training ─────────────────────────────────────────────────────────────
SEQ_LEN: int      = 10
LSTM_EPOCHS: int  = 30
LSTM_BATCH: int   = 64
LSTM_PATIENCE: int = 5

# ── Vital features ────────────────────────────────────────────────────────────
VITAL_FEATURES: list = [
    "heart_rate",
    "respiratory_rate",
    "body_temperature",
    "oxygen_saturation",
    "systolic_bp",
    "diastolic_bp",
    "hrv",
]

RISK_FEATURES: list = VITAL_FEATURES + ["age", "bmi"]

# ── Severity thresholds ───────────────────────────────────────────────────────
THRESH_LOW: float  = 0.30
THRESH_MED: float  = 0.65

# ── Producer ──────────────────────────────────────────────────────────────────
PRODUCER_INTERVAL_SEC: float = 6.0
N_PATIENTS: int = 5

# ── Physiological hard limits ─────────────────────────────────────────────────
PHYSIO_LIMITS: dict = {
    "heart_rate":        (40,   180),
    "respiratory_rate":  (8,    35),
    "body_temperature":  (35.0, 41.5),
    "oxygen_saturation": (84,   100),
    "systolic_bp":       (80,   200),
    "diastolic_bp":      (50,   130),
    "hrv":               (0.01, 0.45),
}

PHYSIO_DELTA: dict = {
    "heart_rate":        3.0,
    "respiratory_rate":  1.0,
    "body_temperature":  0.15,
    "oxygen_saturation": 0.8,
    "systolic_bp":       3.0,
    "diastolic_bp":      2.0,
    "hrv":               0.01,
}

PHYSIO_NOISE: dict = {
    "heart_rate":        0.4,
    "respiratory_rate":  0.2,
    "body_temperature":  0.03,
    "oxygen_saturation": 0.2,
    "systolic_bp":       0.5,
    "diastolic_bp":      0.4,
    "hrv":               0.002,
}

# ── Email alerts ──────────────────────────────────────────────────────────────
EMAIL_USER: str = os.environ.get("EMAIL_USER", "")
EMAIL_PASS: str = os.environ.get("EMAIL_PASS", "")

# ── Alert settings ────────────────────────────────────────────────────────────
ALERT_SUSTAINED_READINGS: int = 3
ALERT_COOLDOWN_MIN: int = 10
