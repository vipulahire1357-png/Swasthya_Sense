"""
app.py — Swasthya Sense Unified Streamlit Entry Point

Navigation:
  🩺 Patient Monitoring
    ├── Doctor Dashboard
    └── AI Health Copilot
  💊 MedSafe AI
    └── Drug Interaction Checker
"""

import os
import streamlit as st
from dotenv import load_dotenv

import backend
from medicine_db import init_db
import dashboard

load_dotenv()
init_db()

st.set_page_config(
    page_title="Swasthya Sense",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

with st.sidebar:
    st.title("🏥 Swasthya Sense")
    st.markdown("*AI-Powered Remote Patient Monitor*")
    st.markdown("---")

    st.markdown("### 🫀 Patient Monitoring")
    st.markdown("### 💊 MedSafe AI")

    page = st.radio(
        "Navigate",
        ["Doctor Dashboard", "AI Health Copilot", "Drug Interaction Checker"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.caption("Swasthya Sense v0.3")

if page == "Doctor Dashboard":
    dashboard.show_dashboard()

elif page == "AI Health Copilot":
    dashboard.show_copilot()

elif page == "Drug Interaction Checker":
    st.title("💊 Drug Interaction Checker")
    st.markdown("Check for dangerous drug interactions powered by **Groq LLaMA 3**.")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        med1 = st.text_input("Medicine 1", placeholder="e.g. Aspirin")
        med2 = st.text_input("Medicine 2", placeholder="e.g. Warfarin")
    with col2:
        med3 = st.text_input("Medicine 3 (optional)", placeholder="e.g. Ibuprofen")
        med4 = st.text_input("Medicine 4 (optional)", placeholder="e.g. Metformin")

    if st.button("🔍 Check Interaction", use_container_width=True):
        if med1.strip() and med2.strip():
            extra = [m for m in [med3, med4] if m.strip()]
            with st.spinner("Analyzing drug interactions with AI..."):
                result = backend.check_interaction(med1, med2, *extra)

            severity = result.get("severity", "Unknown")
            details  = result.get("details", "No details returned.")

            color_map = {
                "Safe":      ("#00c853", "✅"),
                "Moderate":  ("#ffab00", "⚠️"),
                "Dangerous": ("#d50000", "🚨"),
            }
            color, icon = color_map.get(severity, ("#888888", "❓"))

            st.markdown(
                f"""<div style="border-left:6px solid {color};background:#1e1e2e;
                            padding:16px 20px;border-radius:8px;margin-top:12px;">
                    <h3 style="color:{color};margin:0;">{icon} Severity: {severity}</h3>
                </div>""",
                unsafe_allow_html=True,
            )
            st.markdown("#### 📋 Details")
            st.markdown(details)
        else:
            st.warning("Please enter at least two medications.")
