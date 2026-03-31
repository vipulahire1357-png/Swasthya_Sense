"""
dashboard.py — Swasthya Sense Doctor Dashboard & AI Copilot
"""

import os
import json
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from datetime import datetime
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Text
from sqlalchemy.orm import declarative_base, sessionmaker

load_dotenv()
from config import DATABASE_URL
from clinical_scoring import RISK_COLOR, RISK_EMOJI, RISK_ORDER, compute_clinical_score
from alert_engine import evaluate_alerts

Base = declarative_base()

SIMULATED_TOKENS = [f"PAT_{i:08X}" for i in range(1, 6)]


class PatientVital(Base):
    __tablename__ = "patient_vitals"
    id               = Column(Integer, primary_key=True)
    timestamp        = Column(DateTime, default=datetime.utcnow)
    patient_id       = Column(String(20))
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


def _get_session():
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def _pill(label: str, color: str, size: str = "0.78rem") -> str:
    return (
        f'<span style="background:{color};color:#fff;padding:3px 10px;'
        f'border-radius:12px;font-size:{size};font-weight:700;">{label}</span>'
    )


def _get_clinical_risk(row) -> str:
    if row.clinical_risk:
        return row.clinical_risk
    cs = compute_clinical_score(
        hr=row.heart_rate or 72,
        rr=row.respiratory_rate or 15,
        spo2=row.spo2 or 98,
        temp=row.temp or 37,
        systolic=row.blood_pressure or 120,
        hrv=row.hrv or 0.07,
    )
    return cs.risk_level


def _render_priority_panel(patient_rows: list):
    st.markdown("## 🚨 Patient Priority Queue")
    if not patient_rows:
        st.info("No patient data available.")
        return

    sorted_rows = sorted(
        patient_rows,
        key=lambda x: RISK_ORDER.get(_get_clinical_risk(x[1]), 99)
        if x[1] is not None else 99,
    )

    for rank, (token, row) in enumerate(sorted_rows, 1):
        if row is None:
            continue
        risk  = _get_clinical_risk(row)
        color = RISK_COLOR.get(risk, "#888888")
        emoji = RISK_EMOJI.get(risk, "❓")
        news2 = row.news2_score if row.news2_score is not None else "—"
        score = row.anomaly_score or 0.0
        label = token[-8:]

        st.markdown(
            f"""<div style="border-left:6px solid {color};background:#1a1a2e;
                    padding:10px 16px;border-radius:6px;margin-bottom:6px;">
                <b style="color:{color};font-size:1.1rem;">#{rank}</b>&nbsp;&nbsp;
                <b style="color:#fff;">Patient {label}</b>&nbsp;&nbsp;
                {_pill(f"{emoji} {risk}", color)}
                <span style="color:#aaa;font-size:0.78rem;margin-left:8px;">
                    NEWS2:{news2} | Score:{score:.3f} | {row.activity or '—'}
                </span>
            </div>""",
            unsafe_allow_html=True,
        )


def _render_alert_feed(patient_rows: list):
    st.markdown("## ⚠️ Active Alert Feed")
    SEVERITY_COLOR = {
        "CRITICAL": "#d50000", "HIGH": "#ff6d00", "WARNING": "#ffab00", "INFO": "#0288d1",
    }
    SORDER = {"CRITICAL": 0, "HIGH": 1, "WARNING": 2, "INFO": 3}
    all_alerts = []

    for token, row in patient_rows:
        if row is None:
            continue
        if row.active_alerts:
            try:
                stored = json.loads(row.active_alerts)
                for a in stored:
                    a["patient_id"] = token
                    all_alerts.append(a)
                continue
            except Exception:
                pass
        vitals_dict = {
            "heart_rate": row.heart_rate or 72,
            "respiratory_rate": row.respiratory_rate or 15,
            "oxygen_saturation": row.spo2 or 98,
            "body_temperature": row.temp or 37,
            "systolic_bp": row.blood_pressure or 120,
            "diastolic_bp": row.diastolic_bp or 80,
            "hrv": row.hrv or 0.07,
        }
        live_alerts = evaluate_alerts(token, vitals_dict)
        for a in live_alerts:
            d = a.to_dict(); d["patient_id"] = token
            all_alerts.append(d)

    all_alerts.sort(key=lambda x: SORDER.get(x.get("severity", "INFO"), 99))

    if "dismissed_alerts" not in st.session_state:
        st.session_state.dismissed_alerts = set()

    def _key(a):
        return f"{a.get('patient_id','?')}_{a.get('severity','')}_{a.get('event','')}"

    active = [a for a in all_alerts if _key(a) not in st.session_state.dismissed_alerts]
    n_dismissed = len(all_alerts) - len(active)

    _, right = st.columns([3, 1])
    with right:
        if n_dismissed and st.button(f"↺ Reset ({n_dismissed})", key="reset_alerts", use_container_width=True):
            st.session_state.dismissed_alerts = set()
            st.rerun()

    if not active:
        st.success("✅ No active alerts.")
        return

    for alert in active:
        sev   = alert.get("severity", "INFO")
        color = SEVERITY_COLOR.get(sev, "#888")
        label = str(alert.get("patient_id", "?"))[-8:]
        event = alert.get("event", "Alert")
        desc  = alert.get("description", "")
        key   = _key(alert)

        col_alert, col_btn = st.columns([11, 1])
        with col_alert:
            st.markdown(
                f"""<div style="border-left:5px solid {color};background:#12122a;
                        padding:10px 16px;border-radius:6px;margin-bottom:4px;">
                    {_pill(sev, color)}&nbsp;
                    <b style="color:#fff;">Patient {label}</b>&nbsp;
                    <span style="color:{color};font-weight:700;">{event}</span>
                    <br><span style="color:#bbb;font-size:0.82rem;">{desc}</span>
                </div>""",
                unsafe_allow_html=True,
            )
        with col_btn:
            st.markdown("<div style='margin-top:10px'></div>", unsafe_allow_html=True)
            if st.button("✅", key=f"dismiss_{key}", help="Dismiss", use_container_width=True):
                st.session_state.dismissed_alerts.add(key)
                st.rerun()


def _render_patient_cards(session) -> list:
    st.markdown("## 🏥 Live Patient Monitor")
    patient_rows = []
    cols = st.columns(5)

    for idx, token in enumerate(SIMULATED_TOKENS):
        row = (
            session.query(PatientVital)
            .filter_by(patient_id=token)
            .order_by(PatientVital.timestamp.desc())
            .first()
        )
        patient_rows.append((token, row))
        label = token[-8:]

        with cols[idx]:
            if row is None:
                st.markdown(
                    f"""<div style="border:2px solid #555;border-radius:12px;
                            padding:16px;background:#1a1a2e;text-align:center;">
                        <h4 style="color:#aaa;">Patient {label}</h4>
                        <p style="color:#666;font-size:0.8rem;">Waiting for data…</p>
                    </div>""",
                    unsafe_allow_html=True,
                )
                st.progress(0)
                continue

            risk  = _get_clinical_risk(row)
            color = RISK_COLOR.get(risk, "#888888")
            news2 = row.news2_score if row.news2_score is not None else "—"
            mews  = row.mews_score  if row.mews_score  is not None else "—"
            score = row.anomaly_score or 0.0
            rr    = f"{row.respiratory_rate:.0f}" if row.respiratory_rate else "—"
            hrv   = f"{row.hrv:.3f}" if row.hrv else "—"

            st.markdown(
                f"""<div style="border:2px solid {color};border-radius:12px;
                        padding:14px;background:#1a1a2e;text-align:center;
                        box-shadow:0 0 12px {color}44;">
                    <h4 style="color:#fff;margin:0 0 6px 0;">Patient {label}</h4>
                    <div style="margin-bottom:8px;">{_pill(f"{RISK_EMOJI[risk]} {risk}", color, "0.85rem")}</div>
                    <table style="width:100%;color:#ccc;font-size:0.76rem;border-collapse:collapse;">
                        <tr><td>❤️ HR</td><td><b style="color:#fff;">{row.heart_rate:.1f} bpm</b></td></tr>
                        <tr><td>🩸 BP</td><td><b style="color:#fff;">{row.blood_pressure:.1f}/{row.diastolic_bp or 0:.1f}</b></td></tr>
                        <tr><td>🌬 RR</td><td><b style="color:#fff;">{rr} /min</b></td></tr>
                        <tr><td>🌡️ Temp</td><td><b style="color:#fff;">{row.temp:.2f} °C</b></td></tr>
                        <tr><td>💨 SpO₂</td><td><b style="color:#fff;">{row.spo2:.1f} %</b></td></tr>
                        <tr><td>📉 HRV</td><td><b style="color:#fff;">{hrv}</b></td></tr>
                        <tr style="border-top:1px solid #333;">
                            <td>📊 NEWS2</td><td><b style="color:{color};">{news2}</b></td>
                        </tr>
                        <tr><td>📊 MEWS</td><td><b style="color:{color};">{mews}</b></td></tr>
                        <tr><td>🔢 Score</td><td><b style="color:{color};">{score:.3f}</b></td></tr>
                        <tr><td>🏃 State</td><td><b style="color:#ccc;">{row.activity or '—'}</b></td></tr>
                    </table>
                </div>""",
                unsafe_allow_html=True,
            )
            st.progress(min(int(score * 100), 100))

    return patient_rows


def _render_timeline(session):
    st.markdown("---")
    st.markdown("## 📈 Patient Timeline")
    token = st.selectbox(
        "Select Patient",
        SIMULATED_TOKENS,
        format_func=lambda t: f"Patient {t[-8:]}",
    )

    records = (
        session.query(PatientVital)
        .filter(PatientVital.patient_id == token)
        .order_by(PatientVital.timestamp.asc())
        .limit(200)
        .all()
    )

    if not records:
        st.info(f"No records for Patient {token[-8:]} yet.")
        return

    df = pd.DataFrame([{
        "timestamp":        r.timestamp,
        "heart_rate":       r.heart_rate,
        "respiratory_rate": r.respiratory_rate,
        "spo2":             r.spo2,
        "hrv":              r.hrv,
        "temp":             r.temp,
        "anomaly_score":    r.anomaly_score,
        "news2_score":      r.news2_score,
        "composite_score":  r.composite_score,
    } for r in records])

    _dark = dict(
        plot_bgcolor="#0d0d1a", paper_bgcolor="#0d0d1a",
        font_color="#ccc", margin=dict(t=40, b=20, l=20, r=20),
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        fig = px.line(df, x="timestamp", y="heart_rate", title="Heart Rate (bpm)",
                      color_discrete_sequence=["#ff6b6b"])
        fig.update_layout(**_dark); st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.line(df, x="timestamp", y="spo2", title="SpO₂ (%)",
                      color_discrete_sequence=["#4ecdc4"])
        fig.update_layout(**_dark); st.plotly_chart(fig, use_container_width=True)
    with c3:
        fig = px.line(df, x="timestamp", y="respiratory_rate", title="Respiratory Rate",
                      color_discrete_sequence=["#a29bfe"])
        fig.update_layout(**_dark); st.plotly_chart(fig, use_container_width=True)

    c4, c5, c6 = st.columns(3)
    with c4:
        fig = px.line(df, x="timestamp", y="temp", title="Temperature (°C)",
                      color_discrete_sequence=["#fdcb6e"])
        fig.update_layout(**_dark); st.plotly_chart(fig, use_container_width=True)
    with c5:
        fig = px.line(df, x="timestamp", y="hrv", title="HRV",
                      color_discrete_sequence=["#fd79a8"])
        fig.update_layout(**_dark); st.plotly_chart(fig, use_container_width=True)
    with c6:
        fig = px.line(df, x="timestamp", y="anomaly_score", title="Anomaly Score",
                      color_discrete_sequence=["#ffd93d"])
        fig.update_layout(**_dark); st.plotly_chart(fig, use_container_width=True)

    if df["news2_score"].notna().any():
        c7, c8 = st.columns(2)
        with c7:
            fig = px.line(df, x="timestamp", y="news2_score", title="NEWS2 Score",
                          color_discrete_sequence=["#e17055"])
            fig.add_hline(y=5, line_dash="dot", line_color="#ffab00", annotation_text="Medium ≥5")
            fig.add_hline(y=7, line_dash="dot", line_color="#d50000", annotation_text="High ≥7")
            fig.update_layout(**_dark); st.plotly_chart(fig, use_container_width=True)
        with c8:
            if df["composite_score"].notna().any():
                fig = px.line(df, x="timestamp", y="composite_score",
                              title="Composite Risk (0–100)",
                              color_discrete_sequence=["#6c5ce7"])
                fig.update_layout(**_dark); st.plotly_chart(fig, use_container_width=True)


def show_dashboard():
    st.title("🏥 Swasthya Sense — ICU Remote Patient Monitor")
    st.caption("Real-time vital monitoring with ML anomaly detection & clinical scoring")

    with st.sidebar:
        st.markdown("### ⏱ Auto-Refresh")
        refresh_label = st.select_slider(
            "Interval", options=["10s", "30s", "1 min", "5 min", "Off"], value="30s"
        )
        refresh_map = {"10s": 10, "30s": 30, "1 min": 60, "5 min": 300, "Off": 0}
        refresh_sec = refresh_map[refresh_label]

    try:
        session      = _get_session()
        patient_rows = _render_patient_cards(session)
        st.caption(f"Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
        st.markdown("---")

        col_left, col_right = st.columns([1, 2])
        with col_left:
            _render_priority_panel(patient_rows)
        with col_right:
            _render_alert_feed(patient_rows)

        _render_timeline(session)
        session.close()

    except Exception as e:
        st.error(f"Database error: {e}")
        st.info("Make sure PostgreSQL is running and DATABASE_URL is correct in .env")
        return

    if refresh_sec > 0:
        time.sleep(refresh_sec)
        st.rerun()


def show_copilot():
    st.title("🤖 AI Health Copilot")
    st.markdown("Ask a clinical question about any patient's vitals.")
    st.markdown("---")

    question = st.text_input(
        "Clinical question",
        placeholder='e.g. "Why is this patient at HIGH risk?"'
    )
    token = st.selectbox(
        "Patient",
        SIMULATED_TOKENS,
        format_func=lambda t: f"Patient {t[-8:]}",
        key="copilot_token",
    )

    if st.button("🧠 Ask Copilot", use_container_width=True):
        if not question.strip():
            st.warning("Please enter a question.")
            return

        try:
            session = _get_session()
            records = (
                session.query(PatientVital)
                .filter_by(patient_id=token)
                .order_by(PatientVital.timestamp.desc())
                .limit(10)
                .all()
            )
            session.close()
        except Exception as e:
            st.error(f"Database error: {e}"); return

        if not records:
            st.error(f"No records for Patient {token[-8:]}."); return

        table = "\n".join([
            f"{i+1}. HR:{r.heart_rate:.1f} "
            f"BP:{r.blood_pressure:.1f}/{r.diastolic_bp or 0:.1f} "
            f"RR:{r.respiratory_rate or 0:.1f} "
            f"Temp:{r.temp:.2f} SpO2:{r.spo2:.1f} "
            f"HRV:{r.hrv or 0:.4f} "
            f"NEWS2:{r.news2_score or '?'} [{r.clinical_risk or r.severity}]"
            for i, r in enumerate(records)
        ])

        prompt = f"""You are a clinical AI assistant. A doctor asks: "{question}"

Last 10 readings for Patient {token[-8:]} (most recent first):
{table}

Respond with:
- **Key Indicators**: abnormal values, NEWS2 trend, HRV
- **Possible Diagnosis / Risk Factors**
- **Predicted Risk Timeframe**
- **Recommended Action**

Be concise and clinical."""

        groq_key = os.environ.get("GROQ_API_KEY")
        if not groq_key or groq_key.startswith("your_"):
            st.error("GROQ_API_KEY not set in .env"); return

        try:
            from groq import Groq
            client = Groq(api_key=groq_key)
            with st.spinner("Consulting AI Copilot..."):
                response = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="llama-3.3-70b-versatile",
                    temperature=0.2,
                )
            answer = response.choices[0].message.content
        except Exception as e:
            st.error(f"Groq API error: {e}"); return

        st.markdown("### 🏥 Clinical AI Assessment")
        st.markdown(answer)
        st.markdown("---")
        st.warning("⚠️ AI Copilot is a decision-support tool only. Clinical judgment prevails.")
        with st.expander("📊 Raw Vitals Used"):
            st.code(table, language="text")
