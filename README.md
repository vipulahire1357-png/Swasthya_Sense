# 🏥 Swasthya Sense
### AI-Powered Remote Patient Monitoring System

> Real-time vital sign monitoring with ML anomaly detection, clinical early warning scores, and LLM-powered clinical decision support.

---

## 🎯 What It Does

Swasthya Sense simulates an ICU-grade remote patient monitoring platform. It streams physiologically realistic vital signs for 5 patients through Apache Kafka, scores them with ML models and clinical algorithms in real-time, and presents everything through an interactive Streamlit dashboard with an AI clinical copilot.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Data Pipeline                            │
│                                                                 │
│  producer.py  ──Kafka──►  consumer.py  ──SQLAlchemy──► PostgreSQL│
│  (OU process +           (IsoForest +                           │
│   LSTM blending)          LSTM + XGBoost                        │
│                           + NEWS2/MEWS)                         │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼ SQLAlchemy ORM
┌─────────────────────────────────────────────────────────────────┐
│                      Streamlit Dashboard                        │
│                                                                 │
│  ┌─────────────────┐  ┌──────────────────┐  ┌───────────────┐  │
│  │ Doctor Dashboard│  │  AI Health       │  │  Drug         │  │
│  │                 │  │  Copilot         │  │  Interaction  │  │
│  │ • 5 patient     │  │                  │  │  Checker      │  │
│  │   cards (live)  │  │  Groq LLaMA 3    │  │               │  │
│  │ • Priority queue│  │  clinical Q&A    │  │  Groq LLaMA 3 │  │
│  │ • Alert feed    │  │  on patient data │  │  interaction  │  │
│  │ • Timeline      │  │                  │  │  analysis     │  │
│  │   charts        │  │                  │  │               │  │
│  └─────────────────┘  └──────────────────┘  └───────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## ✨ Key Features

### 🫀 Real-Time Patient Monitoring
- **5 simultaneous patients** with physiologically realistic vitals
- **Ornstein-Uhlenbeck process** — mean-reverting stochastic model used in real financial/medical simulation
- **State machine** — patients transition between resting, sleeping, exercise, stress, deteriorating, recovering states
- **Cross-signal physiological coupling** — SpO₂ drops trigger heart rate increases, temperature drives HR changes, etc.
- **Anomaly injection** — rare, random hypoxia / tachycardia / hypertension episodes for realism

### 🧠 3-Layer ML Anomaly Detection
| Layer | Model | Role |
|-------|-------|------|
| 1 | **Isolation Forest** | Instant point anomaly detection |
| 2 | **LSTM Forecaster** | Sequential / temporal risk via reconstruction error |
| 3 | **XGBoost Classifier** | Risk category prediction (Low / High Risk) |

Final score = 40% IsoForest + 60% LSTM (blended)

### 📊 Clinical Early Warning Scores
- **NEWS2** (National Early Warning Score 2) — Royal College of Physicians standard
- **MEWS** (Modified Early Warning Score)
- **HRV Penalty** — Low heart rate variability adds to composite risk
- **Composite Score** (0–100) = NEWS2 (60%) + MEWS (30%) + HRV (10%)
- Risk levels: 🟢 LOW → 🟡 MEDIUM → 🟠 HIGH → 🔴 CRITICAL

### ⚠️ Context-Aware Alert Engine
- Alerts fire only after **3 consecutive abnormal readings** (prevents noise alerts)
- **10-minute cooldown** per alert type per patient (prevents alert fatigue)
- CRITICAL alerts bypass the sustained-reading filter (immediate)
- 8 alert types: Tachycardia, Hypoxia, Respiratory Distress, Fever, Hypotension, HRV Collapse, Hypertensive Crisis, Multi-Organ Deterioration

### 🤖 AI Health Copilot
- Ask plain-English clinical questions about any patient
- Powered by **Groq LLaMA 3.3 70B** (fast inference)
- Receives last 10 readings with NEWS2 scores
- Provides: Key Indicators, Possible Diagnosis, Risk Timeframe, Recommended Action

### 💊 Drug Interaction Checker (MedSafe AI)
- Enter 2–4 medications
- AI classifies interaction as **Safe / Moderate / Dangerous**
- Stores history in local SQLite database

---

## 🔬 Clinical Accuracy

### NEWS2 Scoring (per Royal College of Physicians 2017)
| Parameter | Range | Points |
|-----------|-------|--------|
| Respiratory Rate | ≤8 or >24 | 3 |
| SpO₂ | <92% | 3 |
| Systolic BP | ≤90 mmHg | 3 |
| Heart Rate | ≤40 or >130 | 3 |
| Temperature | ≤35°C | 3 |
| Consciousness | Not Alert | 3 |

**Risk Banding:** Score 0–4 = Low, 5–6 = Medium, 7–8 = High, ≥9 = Critical

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Data Streaming** | Apache Kafka |
| **ML / Deep Learning** | TensorFlow/Keras (LSTM), scikit-learn (IsoForest), XGBoost |
| **Database** | PostgreSQL (vitals) + SQLite (medicines) |
| **ORM** | SQLAlchemy |
| **Dashboard** | Streamlit + Plotly |
| **AI / LLM** | Groq API — LLaMA 3.3 70B Versatile |
| **Language** | Python 3.10+ |

---

## 📁 Project Structure

```
swasthya_sense/
├── app.py                  # Streamlit navigation entry point
├── dashboard.py            # Doctor Dashboard + AI Copilot UI
├── backend.py              # Groq LLM integration
│
├── consumer.py             # Kafka consumer + ML scoring pipeline
├── producer.py             # Physiological vital signs generator
├── models.py               # ML model training (run once)
│
├── clinical_scoring.py     # NEWS2 + MEWS + HRV scoring engine
├── alert_engine.py         # Context-aware alert system
│
├── medicine_db.py          # SQLite drug interaction database
├── config.py               # All constants & config
├── utils.py                # Shared utilities
│
├── setup.py                # One-time database setup
├── requirements.txt
├── .env.example            # Environment variable template
│
├── models/                 # Created by models.py
│   ├── vital_forecaster.keras
│   ├── anomaly_detector.pkl
│   ├── risk_predictor.pkl
│   └── scaler_vitals.pkl
│
├── medicine_db.sqlite      # Auto-created
├── SETUP.md                # Full installation guide
└── README.md               # This file
```

---

## ⚡ Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Edit .env: set DATABASE_URL and GROQ_API_KEY

# 3. Database setup
psql -U postgres -c "CREATE DATABASE vitals_db;"
python setup.py

# 4. Train ML models (needs dataset CSV/ZIP in project root)
python models.py

# 5. Start Kafka (see SETUP.md for full instructions)

# 6. Run all services (4 terminals)
python consumer.py        # Terminal 1
python producer.py        # Terminal 2
streamlit run app.py      # Terminal 3 → opens http://localhost:8501
```

**No Kafka?** Test the producer in console mode:
```bash
python producer.py --mode console
```

---

## 📈 Dashboard Walkthrough

### Doctor Dashboard
- **5 patient cards** — color-coded by risk level (green/yellow/orange/red)
- Each card shows: HR, BP, RR, Temperature, SpO₂, HRV, NEWS2, MEWS, ML score, activity state
- **Priority Queue** — patients sorted by clinical urgency (CRITICAL first)
- **Alert Feed** — live alerts with dismiss buttons, cooldown-filtered
- **Timeline Charts** — 6 charts per patient (HR, SpO₂, RR, Temp, HRV, Anomaly Score) + NEWS2 trend

### AI Health Copilot
- Select a patient → type a clinical question → get structured AI assessment
- Uses last 10 readings, returns: Key Indicators, Diagnosis, Risk Timeframe, Action

### Drug Interaction Checker
- Enter 2–4 drug names → instant Safe/Moderate/Dangerous classification with explanation

---

## 🌊 Physiological Model

The producer uses an **Ornstein-Uhlenbeck (OU) process** — the same mathematical model used in cardiology research and options pricing — to generate mean-reverting vital signs:

```
dX = θ(μ - X)dt + σ√dt · N(0,1)
```

- `θ` = mean-reversion speed (how quickly vitals return to baseline)
- `μ` = state-adjusted target (changes with activity state)
- `σ` = biological noise amplitude

**Cross-signal coupling** (e.g. SpO₂ drop → HR increases) makes the simulation physiologically coherent rather than independent random walks.

---

## 📝 Notes for Judges

- All patient IDs use pseudonymous tokens (PAT_00000001 etc.) — no simulated PHI in the database
- The alert engine is designed for clinical realism: it won't spam alerts, it requires sustained abnormality
- NEWS2 implementation follows the Royal College of Physicians 2017 specification exactly
- The ML pipeline uses 3 independent anomaly signals blended for robustness
- LLM prompts include only de-identified vitals data, never personal information
