"""
setup.py — Swasthya Sense One-Time Setup
=========================================
Creates the PostgreSQL database tables and verifies the environment.

Usage:
    python setup.py
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

print("=" * 60)
print("  Swasthya Sense — Setup")
print("=" * 60)

# ── Check critical env vars ─────────────────────────────────────────
DATABASE_URL = os.environ.get("DATABASE_URL", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

if not DATABASE_URL:
    print("\n❌ DATABASE_URL is not set in .env")
    print("   Example: DATABASE_URL=postgresql://postgres:password@localhost:5432/vitals_db")
    sys.exit(1)

if not GROQ_API_KEY or GROQ_API_KEY.startswith("your_"):
    print("\n⚠️  GROQ_API_KEY is not set — AI features will not work.")
    print("   Get a free key at https://console.groq.com")

# ── Create database tables ──────────────────────────────────────────
print("\n📋 Creating database tables...")

try:
    from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Text
    from sqlalchemy.orm import declarative_base
    from datetime import datetime

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

    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    Base.metadata.create_all(engine)
    print("   ✅ patient_vitals table created (or already exists)")

except Exception as e:
    print(f"   ❌ Database setup failed: {e}")
    print("\n   Common fixes:")
    print("   1. Is PostgreSQL running? Run: pg_isready -h localhost")
    print("   2. Create the database: psql -U postgres -c 'CREATE DATABASE vitals_db;'")
    print("   3. Check DATABASE_URL in .env")
    sys.exit(1)

# ── Init medicine SQLite DB ──────────────────────────────────────────
print("\n💊 Initialising medicine database...")
try:
    from medicine_db import init_db
    init_db()
    print("   ✅ medicine_db.sqlite ready")
except Exception as e:
    print(f"   ⚠️  Medicine DB: {e}")

# ── Create models directory ──────────────────────────────────────────
os.makedirs("models", exist_ok=True)
print("\n📁 models/ directory ready")

print("\n" + "=" * 60)
print("  ✅  Setup Complete!")
print("=" * 60)
print("""
Next steps:
  1. Train ML models (required before starting producer/consumer):
       python models.py

  2. Start services in this order:
       Terminal 1: Start Kafka + Zookeeper (see SETUP.md)
       Terminal 2: python consumer.py
       Terminal 3: python producer.py
       Terminal 4: streamlit run app.py

  3. Open your browser at http://localhost:8501
""")
