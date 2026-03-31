# 🏥 Swasthya Sense — Complete Setup Guide

> **Estimated time:** 20–30 minutes (first run)  
> **Requirements:** Python 3.10+, PostgreSQL 14+, Apache Kafka 3.x, Java 11+

---

## Prerequisites

| Tool | Version | Download |
|------|---------|----------|
| Python | 3.10+ | https://python.org |
| PostgreSQL | 14+ | https://postgresql.org |
| Apache Kafka | 3.x | https://kafka.apache.org/downloads |
| Java (JDK) | 11+ | https://adoptium.net |

---

## Step 1 — Clone / Extract the Project

```bash
# If you have git:
git clone <your-repo-url>
cd swasthya_sense

# OR extract the zip and navigate into the folder
```

---

## Step 2 — Create Python Virtual Environment

```bash
# Create venv
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate
```

---

## Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

> ⏳ Takes 2–5 minutes. TensorFlow is large.

---

## Step 4 — Configure Environment

```bash
# Copy the template
cp .env.example .env
```

Now edit `.env` with your values:

```env
# PostgreSQL — update PASSWORD to match your PostgreSQL setup
DATABASE_URL=postgresql://postgres:YOUR_PASSWORD@localhost:5432/vitals_db

# Get a FREE Groq API key at https://console.groq.com
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxx

# Optional: Gmail alerts (create an App Password in Google Account settings)
EMAIL_USER=your_email@gmail.com
EMAIL_PASS=your_gmail_app_password
```

---

## Step 5 — Create PostgreSQL Database

```bash
# Connect to PostgreSQL
psql -U postgres -h localhost

# Inside psql:
CREATE DATABASE vitals_db;
\q
```

**Windows alternative** using pgAdmin:
1. Open pgAdmin → right-click Databases → Create → Database
2. Name it `vitals_db` → Save

---

## Step 6 — Run Setup Script

```bash
python setup.py
```

Expected output:
```
✅ patient_vitals table created
✅ medicine_db.sqlite ready
✅ models/ directory ready
```

---

## Step 7 — Download Dataset & Train ML Models

The system needs a dataset to train on. Download from Kaggle:

**Dataset:** [Human Vital Signs Dataset](https://www.kaggle.com/datasets/your-dataset-link)  
Place `vitals_dataset_2024.zip` or `vitals_dataset_2024.csv` in the project root.

Then train:

```bash
python models.py
```

> ⏳ Takes 3–10 minutes. Creates:
> - `models/vital_forecaster.keras` — LSTM predictor
> - `models/anomaly_detector.pkl` — Isolation Forest
> - `models/risk_predictor.pkl` — XGBoost classifier
> - `models/scaler_vitals.pkl` — StandardScaler

---

## Step 8 — Start Apache Kafka

### Windows

Open **Terminal 1** (Zookeeper):
```bash
cd C:\kafka
.\bin\windows\zookeeper-server-start.bat .\config\zookeeper.properties
```

Wait for: `INFO binding to port 0.0.0.0/0.0.0.0:2181`

Open **Terminal 2** (Kafka Broker):
```bash
cd C:\kafka
.\bin\windows\kafka-server-start.bat .\config\server.properties
```

Wait for: `INFO [KafkaServer id=0] started`

### Mac/Linux

```bash
# Terminal 1
$KAFKA_HOME/bin/zookeeper-server-start.sh $KAFKA_HOME/config/zookeeper.properties

# Terminal 2
$KAFKA_HOME/bin/kafka-server-start.sh $KAFKA_HOME/config/server.properties
```

---

## Step 9 — Start All Services

Open **4 terminals** in the project directory with venv activated.

### Terminal 3 — Consumer (processes vitals)
```bash
python consumer.py
```
Expected: `Listening on 'vitals' …`

### Terminal 4 — Producer (generates data)
```bash
python producer.py
```
Expected: `PAT_00000001 [resting    ] HR: 72.4 BP:118.2/76.1 ...`

### Terminal 5 — Streamlit Dashboard
```bash
streamlit run app.py
```
Opens browser at **http://localhost:8501**

---

## Step 10 — Verify It's Working

1. Open http://localhost:8501
2. Go to **Doctor Dashboard**
3. Within 30 seconds you should see 5 patient cards filling with data
4. Try **AI Health Copilot** — select a patient, ask "What is the current risk?"
5. Try **Drug Interaction Checker** — enter "Aspirin" and "Warfarin"

---

## Producer Modes

```bash
# Default: send to Kafka (requires Kafka running)
python producer.py

# Console only (no Kafka needed — for testing)
python producer.py --mode console

# REST mode (sends directly to Flask API)
python producer.py --mode rest
```

---

## Troubleshooting

### `psycopg2.OperationalError: could not connect to server`
- PostgreSQL not running. Start it:
  - Windows: Services → PostgreSQL → Start
  - Mac: `brew services start postgresql`
  - Linux: `sudo systemctl start postgresql`

### `kafka.errors.NoBrokersAvailable`
- Kafka not running. Start Zookeeper first, then Kafka (Step 8).
- Or run producer in console mode: `python producer.py --mode console`

### `Error: Models not found. Run python models.py first.`
- Run `python models.py` before starting producer/consumer.
- Check that the `models/` folder contains `.keras` and `.pkl` files.

### Dashboard shows "Waiting for data…"
- Producer and Consumer must both be running.
- Wait ~30 seconds for first readings to arrive.
- Check consumer terminal for error messages.

### `GROQ_API_KEY` errors
- Get a free key at https://console.groq.com
- Paste it into `.env` (no quotes needed).

### `ModuleNotFoundError`
```bash
pip install -r requirements.txt
```

---

## File Structure

```
swasthya_sense/
├── app.py              # Streamlit entry point
├── dashboard.py        # Doctor dashboard & AI Copilot UI
├── backend.py          # Groq LLM integration
├── consumer.py         # Kafka consumer + ML scoring
├── producer.py         # Vital signs data generator
├── models.py           # ML model training pipeline
├── clinical_scoring.py # NEWS2 + MEWS scoring engine
├── alert_engine.py     # Context-aware alert system
├── medicine_db.py      # SQLite drug interaction DB
├── config.py           # Central configuration
├── utils.py            # Shared utilities
├── setup.py            # One-time database setup
├── requirements.txt
├── .env.example
├── models/             # Created by models.py
│   ├── vital_forecaster.keras
│   ├── anomaly_detector.pkl
│   ├── risk_predictor.pkl
│   └── scaler_vitals.pkl
└── medicine_db.sqlite  # Created automatically
```
