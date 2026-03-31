"""
consumer.py — Swasthya Sense Kafka Consumer + ML Scoring
=========================================================
Reads from Kafka, runs anomaly detection + clinical scoring,
stores results to PostgreSQL.
"""

import os
import json
import pickle
import smtplib
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from email.message import EmailMessage
from datetime import datetime
from kafka import KafkaConsumer
from sqlalchemy import (
    create_engine, Column, Integer, Float, String, DateTime, Text
)
from sqlalchemy.orm import declarative_base, sessionmaker
from dotenv import load_dotenv

from config import (
    DATABASE_URL, KAFKA_BOOTSTRAP, KAFKA_TOPIC,
    VITAL_FORECASTER_PATH, ANOMALY_DETECTOR_PATH, RISK_PREDICTOR_PATH,
    SCALER_VITALS_PATH,
    VITAL_FEATURES, SEQ_LEN,
    THRESH_LOW, THRESH_MED,
    EMAIL_USER, EMAIL_PASS,
)
from utils import score_to_severity, get_logger
from clinical_scoring import compute_clinical_score

load_dotenv()
log = get_logger("consumer")

Base = declarative_base()


class PatientVital(Base):
    __tablename__ = "patient_vitals"
    id               = Column(Integer, primary_key=True)
    timestamp        = Column(DateTime, default=datetime.utcnow)
    patient_id       = Column(String(20), index=True)
    heart_rate       = Column(Float)
    blood_pressure   = Column(Float)
    diastolic_bp     = Column(Float)
    respiratory_rate = Column(Float)
    temp             = Column(Float)
    spo2             = Column(Float)
    hrv              = Column(Float)
    pulse_pressure   = Column(Float)
    map_value        = Column(Float)
    anomaly_score    = Column(Float)
    iso_score        = Column(Float)
    lstm_risk        = Column(Float)
    risk_prediction  = Column(String)
    severity         = Column(String)
    news2_score      = Column(Integer)
    mews_score       = Column(Integer)
    clinical_risk    = Column(String)
    composite_score  = Column(Float)
    activity         = Column(String)
    drift_mode       = Column(String)
    active_alerts    = Column(Text)


def get_db_session():
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def send_alert(patient_token: str, event: dict, final_score: float, clinical_risk: str):
    if not EMAIL_USER or not EMAIL_PASS:
        log.info("Alert email skipped — EMAIL_USER/EMAIL_PASS not set.")
        return

    body = f"""🚨 ALERT — {patient_token}
Risk Score: {final_score:.3f}  |  Clinical Risk: {clinical_risk}
HR: {event.get('heart_rate','?')} bpm
BP: {event.get('systolic_bp','?')}/{event.get('diastolic_bp','?')} mmHg
RR: {event.get('respiratory_rate','?')} /min
Temp: {event.get('body_temperature', event.get('temp','?'))} °C
SpO₂: {event.get('oxygen_saturation', event.get('spo2','?'))} %
HRV: {event.get('hrv','?')}
Immediate clinical review recommended."""

    msg = EmailMessage()
    msg.set_content(body)
    msg["Subject"] = f"🚨 {clinical_risk} Alert — {patient_token}"
    msg["From"]    = EMAIL_USER
    msg["To"]      = EMAIL_USER

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
            s.login(EMAIL_USER, EMAIL_PASS)
            s.send_message(msg)
        log.info(f"Alert sent for {patient_token}")
    except Exception as e:
        log.warning(f"Email failed: {e}")


def run_consumer():
    from alert_engine import evaluate_alerts
    session = get_db_session()

    log.info("Loading ML models...")
    with open(SCALER_VITALS_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(ANOMALY_DETECTOR_PATH, "rb") as f:
        iso_forest = pickle.load(f)

    lstm_model = None
    try:
        from tensorflow.keras.models import load_model
        lstm_model = load_model(VITAL_FORECASTER_PATH)
        log.info("LSTM loaded.")
    except Exception as e:
        log.warning(f"LSTM load failed: {e}")

    risk_artifact = None
    try:
        with open(RISK_PREDICTOR_PATH, "rb") as f:
            risk_artifact = pickle.load(f)
        log.info("Risk predictor loaded.")
    except Exception as e:
        log.warning(f"Risk predictor load failed: {e}")

    patient_histories: dict = {}

    consumer = KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP,
        value_deserializer=lambda x: json.loads(x.decode("utf-8")),
    )
    log.info(f"Listening on '{KAFKA_TOPIC}' …")

    for msg in consumer:
        v = msg.value

        raw_pid = v.get("patient_id", 0)
        patient_token = (
            raw_pid if isinstance(raw_pid, str) and raw_pid.startswith("PAT_")
            else f"PAT_{int(raw_pid):08X}"
        )

        vital_vec = np.array([[
            v.get("heart_rate",        v.get("hr",   75)),
            v.get("respiratory_rate",  16),
            v.get("body_temperature",  v.get("temp", 37)),
            v.get("oxygen_saturation", v.get("spo2", 98)),
            v.get("systolic_bp",       v.get("bp",   120)),
            v.get("diastolic_bp",      80),
            v.get("hrv",               0.07),
        ]])
        f_scaled = scaler.transform(vital_vec)

        iso_score = float(max(0.0, -iso_forest.decision_function(f_scaled)[0]))

        lstm_risk = None
        if lstm_model is not None:
            if patient_token not in patient_histories:
                patient_histories[patient_token] = np.tile(f_scaled[0], (SEQ_LEN, 1))
            else:
                hist = patient_histories[patient_token]
                patient_histories[patient_token] = np.roll(hist, -1, axis=0)
                patient_histories[patient_token][-1] = f_scaled[0]

            seq_input   = patient_histories[patient_token].reshape(1, SEQ_LEN, len(VITAL_FEATURES))
            pred_scaled = lstm_model.predict(seq_input, verbose=0)[0]
            lstm_err    = float(np.mean(np.square(f_scaled[0] - pred_scaled)))
            lstm_risk   = min(lstm_err * 10.0, 1.0)

        if lstm_risk is not None:
            final_score = 0.4 * min(iso_score, 1.0) + 0.6 * lstm_risk
        else:
            final_score = min(iso_score, 1.0)

        severity = score_to_severity(final_score)

        risk_label = "Unknown"
        if risk_artifact is not None:
            try:
                age      = float(v.get("age", 40))
                bmi      = float(v.get("bmi", 22.0))
                full_vec = np.hstack([vital_vec[0], [age, bmi]]).reshape(1, -1)
                full_sc  = risk_artifact["scaler"].transform(full_vec)
                pred_cls = risk_artifact["model"].predict(full_sc)[0]
                risk_label = risk_artifact["label_encoder"].inverse_transform([pred_cls])[0]
            except Exception as ex:
                log.debug(f"Risk predictor error: {ex}")

        clinical = compute_clinical_score(
            hr       = vital_vec[0][0],
            rr       = vital_vec[0][1],
            spo2     = vital_vec[0][3],
            temp     = vital_vec[0][2],
            systolic = vital_vec[0][4],
            hrv      = vital_vec[0][6],
        )

        alerts      = evaluate_alerts(patient_token, v)
        alerts_json = json.dumps([a.to_dict() for a in alerts])

        log.info(
            f"{patient_token} | "
            f"HR:{vital_vec[0][0]:5.1f} BP:{vital_vec[0][4]:5.1f}/{vital_vec[0][5]:4.1f} "
            f"Temp:{vital_vec[0][2]:5.2f} SpO2:{vital_vec[0][3]:5.1f} "
            f"| ISO:{iso_score:.3f} Final:{final_score:.3f} [{severity}] "
            f"NEWS2:{clinical.news2} [{clinical.risk_level}]"
            + (f" ⚠ {len(alerts)} alerts" if alerts else "")
        )

        if clinical.risk_level in ("HIGH", "CRITICAL") or severity == "HIGH":
            send_alert(patient_token, v, final_score, clinical.risk_level)

        record = PatientVital(
            patient_id       = patient_token,
            heart_rate       = float(vital_vec[0][0]),
            blood_pressure   = float(vital_vec[0][4]),
            diastolic_bp     = float(vital_vec[0][5]),
            respiratory_rate = float(vital_vec[0][1]),
            temp             = float(vital_vec[0][2]),
            spo2             = float(vital_vec[0][3]),
            hrv              = float(vital_vec[0][6]),
            pulse_pressure   = float(v.get("pulse_pressure", 0)),
            map_value        = float(v.get("map", 0)),
            anomaly_score    = float(final_score),
            iso_score        = float(iso_score),
            lstm_risk        = float(lstm_risk) if lstm_risk is not None else None,
            risk_prediction  = risk_label,
            severity         = severity,
            news2_score      = clinical.news2,
            mews_score       = clinical.mews,
            clinical_risk    = clinical.risk_level,
            composite_score  = clinical.composite,
            activity         = v.get("activity", "resting"),
            drift_mode       = v.get("drift_mode", "normal"),
            active_alerts    = alerts_json,
        )
        session.add(record)
        session.commit()


if __name__ == "__main__":
    run_consumer()
