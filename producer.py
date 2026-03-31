"""
producer.py — Swasthya Sense Vital Signs Producer
==================================================
Generates physiologically realistic patient vitals using
Ornstein-Uhlenbeck process + state machine + LSTM guidance.

Modes:
    python producer.py            # Kafka (default)
    python producer.py --mode console
    python producer.py --mode rest
"""

import os
import json
import time
import random
import argparse
import pickle
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from datetime import datetime, timezone

from config import (
    KAFKA_BOOTSTRAP, KAFKA_TOPIC,
    VITAL_FORECASTER_PATH, SCALER_VITALS_PATH,
    VITAL_FEATURES, PHYSIO_LIMITS,
    PRODUCER_INTERVAL_SEC, N_PATIENTS, SEQ_LEN,
)
from utils import compute_pulse_pressure, compute_map, get_logger

log = get_logger("producer")

# ── OU parameters: (theta, sigma) ────────────────────────────────────────────
OU_PARAMS = {
    "heart_rate":        (0.12, 0.6),
    "respiratory_rate":  (0.15, 0.3),
    "body_temperature":  (0.08, 0.02),
    "oxygen_saturation": (0.18, 0.15),
    "systolic_bp":       (0.10, 0.7),
    "diastolic_bp":      (0.10, 0.5),
    "hrv":               (0.12, 0.003),
}

BASELINE_VITALS = {
    "heart_rate":        72.0,
    "respiratory_rate":  15.0,
    "body_temperature":  36.8,
    "oxygen_saturation": 98.0,
    "systolic_bp":       118.0,
    "diastolic_bp":      76.0,
    "hrv":               0.070,
}

STATE_OFFSETS = {
    "resting":       {"heart_rate": 0, "respiratory_rate": 0, "body_temperature": 0,
                      "oxygen_saturation": 0, "systolic_bp": 0, "diastolic_bp": 0, "hrv": 0},
    "sleeping":      {"heart_rate": -14, "respiratory_rate": -4, "body_temperature": -0.3,
                      "oxygen_saturation": -0.5, "systolic_bp": -12, "diastolic_bp": -8, "hrv": +0.025},
    "exercise":      {"heart_rate": +40, "respiratory_rate": +12, "body_temperature": +0.5,
                      "oxygen_saturation": -1.0, "systolic_bp": +25, "diastolic_bp": +10, "hrv": -0.025},
    "stress":        {"heart_rate": +22, "respiratory_rate": +5, "body_temperature": +0.2,
                      "oxygen_saturation": -0.5, "systolic_bp": +18, "diastolic_bp": +10, "hrv": -0.020},
    "deteriorating": {"heart_rate": +18, "respiratory_rate": +8, "body_temperature": +0.8,
                      "oxygen_saturation": -5.0, "systolic_bp": -10, "diastolic_bp": -5, "hrv": -0.030},
    "recovering":    {"heart_rate": +5, "respiratory_rate": +2, "body_temperature": +0.2,
                      "oxygen_saturation": -1.5, "systolic_bp": -3, "diastolic_bp": -2, "hrv": -0.010},
}

STATE_TRANSITION_PROBS = {
    "sleeping": 0.08, "exercise": 0.06,
    "stress": 0.05, "deteriorating": 0.03,
}

STATE_DWELL = {
    "resting": (10, 40), "sleeping": (40, 120), "exercise": (10, 35),
    "stress": (8, 25), "deteriorating": (20, 80), "recovering": (15, 50),
}


def ou_step(current, mean, theta, sigma, dt=1.0):
    drift = theta * (mean - current) * dt
    noise = sigma * np.sqrt(dt) * np.random.normal()
    return current + drift + noise


class PatientSimulator:
    def __init__(self, patient_id, initial_vitals, static, lstm_model, scaler):
        self.pid    = patient_id
        self.vitals = initial_vitals.copy()
        self.static = static
        self.model  = lstm_model
        self.scaler = scaler

        init_vec     = np.array([[initial_vitals[f] for f in VITAL_FEATURES]])
        init_scaled  = scaler.transform(init_vec)[0]
        self.history = np.tile(init_scaled, (SEQ_LEN, 1))

        self.personal_baseline  = initial_vitals.copy()
        self.state              = "resting"
        self.state_steps_left   = random.randint(*STATE_DWELL["resting"])
        self.anomaly_mode       = "normal"
        self.anomaly_steps_left = 0

    def _tick_state(self):
        self.state_steps_left -= 1
        if self.state_steps_left > 0:
            return
        if self.state == "deteriorating":
            self.state = "recovering"
        elif self.state == "recovering":
            self.state = "resting"
        else:
            roll, cumulative, new_state = random.random(), 0.0, "resting"
            for s, p in STATE_TRANSITION_PROBS.items():
                cumulative += p
                if roll < cumulative:
                    new_state = s
                    break
            if new_state != self.state:
                log.info(f"P{self.pid:02d}: {self.state} → {new_state}")
            self.state = new_state
        self.state_steps_left = random.randint(*STATE_DWELL.get(self.state, (10, 30)))

    def _tick_anomaly(self):
        if self.anomaly_steps_left > 0:
            self.anomaly_steps_left -= 1
        elif random.random() < 0.008:
            self.anomaly_mode       = random.choice(["hypoxia", "tachycardia", "hypertension"])
            self.anomaly_steps_left = random.randint(5, 18)
            log.warning(f"P{self.pid:02d}: anomaly → {self.anomaly_mode}")
        else:
            self.anomaly_mode = "normal"

    def step(self):
        self._tick_state()
        self._tick_anomaly()

        offsets    = STATE_OFFSETS.get(self.state, STATE_OFFSETS["resting"])
        target     = {f: self.personal_baseline[f] + offsets.get(f, 0) for f in VITAL_FEATURES}
        new_vitals = {}

        for feat in VITAL_FEATURES:
            theta, sigma = OU_PARAMS[feat]
            new_vitals[feat] = ou_step(self.vitals[feat], target[feat], theta, sigma)

        # Cross-signal coupling
        spo2_drop = max(0.0, 98.0 - self.vitals["oxygen_saturation"])
        new_vitals["heart_rate"] += 0.30 * spo2_drop
        new_vitals["heart_rate"] += 8.0 * max(0.0, self.vitals["body_temperature"] - 37.0)
        new_vitals["respiratory_rate"] += 0.20 * spo2_drop
        new_vitals["hrv"] = max(0.005, new_vitals["hrv"])

        # Anomaly injection
        anomaly_deltas = {
            "hypoxia":      {"oxygen_saturation": -7.0, "respiratory_rate": +6.0, "heart_rate": +15.0},
            "tachycardia":  {"heart_rate": +45.0, "respiratory_rate": +4.0},
            "hypertension": {"systolic_bp": +40.0, "diastolic_bp": +22.0},
        }
        if self.anomaly_mode in anomaly_deltas:
            for feat, delta in anomaly_deltas[self.anomaly_mode].items():
                new_vitals[feat] = new_vitals.get(feat, self.vitals[feat]) + delta

        # LSTM blending
        try:
            seq_input   = self.history.reshape(1, SEQ_LEN, len(VITAL_FEATURES))
            pred_scaled = self.model.predict(seq_input, verbose=0)[0]
            raw_pred    = self.scaler.inverse_transform(pred_scaled.reshape(1, -1))[0]
            for i, feat in enumerate(VITAL_FEATURES):
                new_vitals[feat] = 0.90 * new_vitals[feat] + 0.10 * raw_pred[i]
        except Exception:
            pass

        # Hard limits
        for feat in VITAL_FEATURES:
            lo, hi = PHYSIO_LIMITS[feat]
            new_vitals[feat] = float(np.clip(new_vitals[feat], lo, hi))

        # Update history
        new_vec          = np.array([[new_vitals[f] for f in VITAL_FEATURES]])
        new_scaled       = self.scaler.transform(new_vec)[0]
        self.history     = np.roll(self.history, -1, axis=0)
        self.history[-1] = new_scaled
        self.vitals      = new_vitals

        sys_bp = new_vitals["systolic_bp"]
        dia_bp = new_vitals["diastolic_bp"]

        return {
            "patient_id":        f"PAT_{self.pid:08X}",
            "timestamp":         datetime.now(timezone.utc).isoformat(),
            "heart_rate":        round(new_vitals["heart_rate"], 1),
            "respiratory_rate":  round(new_vitals["respiratory_rate"], 1),
            "body_temperature":  round(new_vitals["body_temperature"], 2),
            "oxygen_saturation": round(new_vitals["oxygen_saturation"], 1),
            "systolic_bp":       round(sys_bp, 1),
            "diastolic_bp":      round(dia_bp, 1),
            "hrv":               round(new_vitals["hrv"], 4),
            "pulse_pressure":    round(compute_pulse_pressure(sys_bp, dia_bp), 1),
            "map":               round(compute_map(sys_bp, dia_bp), 1),
            "bmi":               round(self.static.get("bmi", 22.0), 2),
            "age":               self.static.get("age", 40),
            "activity":          self.state,
            "anomaly_injected":  self.anomaly_mode,
            "drift_mode":        self.state,
            "hr":                round(new_vitals["heart_rate"], 1),
            "bp":                round(sys_bp, 1),
            "temp":              round(new_vitals["body_temperature"], 2),
            "spo2":              round(new_vitals["oxygen_saturation"], 1),
        }


def load_seed_patients(n):
    seeds = []
    for _ in range(n):
        seeds.append({
            "vitals": {
                "heart_rate":        random.uniform(64, 88),
                "respiratory_rate":  random.uniform(12, 18),
                "body_temperature":  random.uniform(36.3, 37.1),
                "oxygen_saturation": random.uniform(96, 99),
                "systolic_bp":       random.uniform(112, 130),
                "diastolic_bp":      random.uniform(70, 84),
                "hrv":               random.uniform(0.04, 0.12),
            },
            "static": {
                "age": random.randint(25, 72),
                "bmi": round(random.uniform(18.5, 30.0), 1),
            },
        })
    return seeds


def run_producer(mode: str = "kafka"):
    if not os.path.exists(VITAL_FORECASTER_PATH) or not os.path.exists(SCALER_VITALS_PATH):
        log.error("Models not found. Run  python models.py  first.")
        return

    log.info("Loading LSTM forecaster and scaler...")
    from tensorflow.keras.models import load_model
    lstm_model = load_model(VITAL_FORECASTER_PATH)
    with open(SCALER_VITALS_PATH, "rb") as f:
        scaler = pickle.load(f)

    seeds    = load_seed_patients(N_PATIENTS)
    patients = [
        PatientSimulator(
            patient_id     = pid + 1,
            initial_vitals = seeds[pid]["vitals"],
            static         = seeds[pid]["static"],
            lstm_model     = lstm_model,
            scaler         = scaler,
        )
        for pid in range(N_PATIENTS)
    ]

    kafka_producer = None
    if mode == "kafka":
        try:
            from kafka import KafkaProducer
            kafka_producer = KafkaProducer(
                bootstrap_servers=KAFKA_BOOTSTRAP,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            )
            log.info("Connected to Kafka.")
        except Exception as e:
            log.error(f"Kafka failed: {e} — switching to console mode.")
            mode = "console"

    log.info(f"Producer running | mode={mode} | {N_PATIENTS} patients | {PRODUCER_INTERVAL_SEC}s")

    try:
        while True:
            for patient in patients:
                event = patient.step()

                if mode == "kafka" and kafka_producer:
                    kafka_producer.send(KAFKA_TOPIC, event)
                elif mode == "rest":
                    import requests
                    try:
                        requests.post("http://localhost:5000/api/ingest", json=event, timeout=2)
                    except Exception:
                        pass

                log.info(
                    f"{event['patient_id']} [{event['activity']:<12}] "
                    f"HR:{event['heart_rate']:5.1f} "
                    f"BP:{event['systolic_bp']:5.1f}/{event['diastolic_bp']:4.1f} "
                    f"SpO2:{event['oxygen_saturation']:5.1f} "
                    f"Temp:{event['body_temperature']:5.2f}"
                )

            if kafka_producer:
                kafka_producer.flush()
            time.sleep(PRODUCER_INTERVAL_SEC)

    except KeyboardInterrupt:
        log.info("Producer stopped.")
    finally:
        if kafka_producer:
            kafka_producer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["kafka", "console", "rest"], default="kafka")
    args = parser.parse_args()
    run_producer(mode=args.mode)
