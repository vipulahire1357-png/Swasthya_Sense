"""
models.py — AI-HACK-07 Model Training Pipeline
Trains three models from the real human vital-signs dataset:
  1. LSTM vital-sign forecaster  (predicts next timestep)
  2. Autoencoder anomaly detector
  3. XGBoost risk-category classifier

Run once:
    python models.py
"""

import os
import io
import pickle
import zipfile
import logging
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Input, RepeatVector, TimeDistributed
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from config import (
    DATASET_ZIP, DATASET_CSV, MODEL_DIR,
    VITAL_FORECASTER_PATH, ANOMALY_DETECTOR_PATH, RISK_PREDICTOR_PATH,
    SCALER_VITALS_PATH, SCALER_FULL_PATH,
    SEQ_LEN, LSTM_EPOCHS, LSTM_BATCH, LSTM_PATIENCE,
    VITAL_FEATURES, RISK_FEATURES,
)
from utils import ensure_model_dir, get_logger

log = get_logger("models")


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING & PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def load_dataset(max_rows: int = 50_000) -> pd.DataFrame:
    """
    Load the dataset from the zip file (or CSV if already extracted).
    Limits to max_rows to keep training fast on commodity hardware.
    Kaggle column names are mapped to snake_case for consistency.
    """
    log.info("Loading dataset...")

    # Try zip first, then bare CSV
    if os.path.exists(DATASET_ZIP):
        with zipfile.ZipFile(DATASET_ZIP) as zf:
            # The CSV may be inside a subdirectory within the zip
            csv_entries = [n for n in zf.namelist() if n.endswith(".csv")]
            if not csv_entries:
                raise FileNotFoundError("No CSV found inside the zip file.")
            zip_csv_path = csv_entries[0]
            log.info(f"Reading from zip entry: {zip_csv_path}")
            with zf.open(zip_csv_path) as f:
                df = pd.read_csv(f, nrows=max_rows)
    elif os.path.exists(DATASET_CSV):
        df = pd.read_csv(DATASET_CSV, nrows=max_rows)
    else:
        raise FileNotFoundError(
            f"Dataset not found. Place '{DATASET_ZIP}' or '{DATASET_CSV}' "
            "in the project root."
        )

    log.info(f"Loaded {len(df):,} rows, {df.shape[1]} columns.")
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename columns → snake_case, compute derived features,
    drop nulls, sort by timestamp.
    """
    df = df.copy()

    # ── Rename to snake_case ──────────────────────────────────────────────
    rename_map = {
        "Patient ID":               "patient_id",
        "Heart Rate":               "heart_rate",
        "Respiratory Rate":         "respiratory_rate",
        "Timestamp":                "timestamp",
        "Body Temperature":         "body_temperature",
        "Oxygen Saturation":        "oxygen_saturation",
        "Systolic Blood Pressure":  "systolic_bp",
        "Diastolic Blood Pressure": "diastolic_bp",
        "Age":                      "age",
        "Gender":                   "gender",
        "Weight (kg)":              "weight_kg",
        "Height (m)":               "height_m",
        "Derived_HRV":              "hrv",
        "Derived_Pulse_Pressure":   "pulse_pressure",
        "Derived_BMI":              "bmi",
        "Derived_MAP":              "map",
        "Risk Category":            "risk_category",
    }
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns},
              inplace=True)

    # ── Timestamp ─────────────────────────────────────────────────────────
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # ── Derived features (recompute for correctness) ──────────────────────
    df["pulse_pressure"] = df["systolic_bp"] - df["diastolic_bp"]
    df["map"]            = (df["systolic_bp"] + 2 * df["diastolic_bp"]) / 3.0
    df["bmi"]            = df["weight_kg"] / (df["height_m"] ** 2)

    # ── Drop rows with missing vitals ─────────────────────────────────────
    required = VITAL_FEATURES + ["age", "bmi", "risk_category"]
    df.dropna(subset=required, inplace=True)

    # ── Sort ──────────────────────────────────────────────────────────────
    if "timestamp" in df.columns:
        df.sort_values(["patient_id", "timestamp"], inplace=True)

    log.info(f"After preprocessing: {len(df):,} rows.")
    return df.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# 2. SEQUENCE BUILDER FOR LSTM
# ══════════════════════════════════════════════════════════════════════════════

def build_sequences(
    df: pd.DataFrame,
    scaler: StandardScaler,
    seq_len: int = SEQ_LEN,
    max_sequences: int = 20_000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build overlapping windows of length seq_len.
    X: (N, seq_len, n_features)   — input window
    y: (N, n_features)            — next-step vitals (forecasting target)

    If each patient has only 1 row (as in this dataset), we treat the entire
    DataFrame as one continuous sequence — the temporal order is preserved via
    the timestamp sort done in preprocess().
    """
    log.info("Building LSTM sequences...")
    X_list, y_list = [], []

    # Check whether grouping by patient yields any usable sequences
    max_per_patient = df.groupby("patient_id").size().max()

    if max_per_patient > seq_len:
        # Multiple rows per patient — group by patient (original behaviour)
        for _, grp in df.groupby("patient_id"):
            vals = grp[VITAL_FEATURES].values
            vals_scaled = scaler.transform(vals)
            for i in range(len(vals_scaled) - seq_len):
                X_list.append(vals_scaled[i : i + seq_len])
                y_list.append(vals_scaled[i + seq_len])
            if len(X_list) >= max_sequences:
                break
    else:
        # One row per patient — treat the whole DataFrame as one time-series
        log.info(
            "Each patient has ≤1 row; using global sliding window across all rows."
        )
        vals_scaled = scaler.transform(df[VITAL_FEATURES].values)
        for i in range(len(vals_scaled) - seq_len):
            X_list.append(vals_scaled[i : i + seq_len])
            y_list.append(vals_scaled[i + seq_len])
            if len(X_list) >= max_sequences:
                break

    X = np.array(X_list[:max_sequences])
    y = np.array(y_list[:max_sequences])
    log.info(f"Sequences: X={X.shape}, y={y.shape}")
    return X, y


# ══════════════════════════════════════════════════════════════════════════════
# 3. MODEL BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def build_lstm_forecaster(n_features: int, seq_len: int) -> tf.keras.Model:
    """
    Multi-layer LSTM that predicts the next timestep of all vital features.
    Architecture: Input → LSTM(64) → Dropout → LSTM(32) → Dense(n_features)
    """
    model = Sequential([
        LSTM(64, input_shape=(seq_len, n_features), return_sequences=True),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.1),
        Dense(16, activation="relu"),
        Dense(n_features, activation="linear"),
    ], name="vital_forecaster")
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def build_autoencoder(n_features: int) -> tf.keras.Model:
    """
    LSTM-based sequence autoencoder for anomaly detection.
    Reconstruction error = anomaly score.
    """
    inp = Input(shape=(SEQ_LEN, n_features))
    # Encoder
    encoded = LSTM(16, activation="relu")(inp)
    # Decoder
    decoded = RepeatVector(SEQ_LEN)(encoded)
    decoded = LSTM(16, activation="relu", return_sequences=True)(decoded)
    decoded = TimeDistributed(Dense(n_features))(decoded)

    model = Model(inp, decoded, name="autoencoder")
    model.compile(optimizer="adam", loss="mse")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# 4. TRAINING FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def train_lstm(X_train, y_train, X_val, y_val) -> tf.keras.Model:
    n_features = X_train.shape[2]
    model = build_lstm_forecaster(n_features, SEQ_LEN)
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=LSTM_PATIENCE,
                      restore_best_weights=True, verbose=1),
        ModelCheckpoint(VITAL_FORECASTER_PATH, save_best_only=True, verbose=0),
    ]
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=LSTM_EPOCHS,
        batch_size=LSTM_BATCH,
        callbacks=callbacks,
        verbose=1,
    )
    log.info(f"LSTM saved → {VITAL_FORECASTER_PATH}")
    return model


def train_anomaly_detector(df: pd.DataFrame, scaler: StandardScaler) -> None:
    """
    Dual approach:
      - IsolationForest on raw vitals (fast, interpretable)
      - LSTM Autoencoder on sequences (deep, temporal)
    Both artifacts are saved.
    """
    # ── Isolation Forest ──────────────────────────────────────────────────
    log.info("Training IsolationForest anomaly detector...")
    X_flat = scaler.transform(df[VITAL_FEATURES].values)
    iso = IsolationForest(
        n_estimators=200,
        contamination=0.08,
        random_state=42,
        n_jobs=-1,
    )
    iso.fit(X_flat)
    with open(ANOMALY_DETECTOR_PATH, "wb") as f:
        pickle.dump(iso, f)
    log.info(f"IsolationForest saved → {ANOMALY_DETECTOR_PATH}")


def train_risk_predictor(df: pd.DataFrame) -> None:
    """
    XGBoost risk-category classifier.
    Features: vitals + age + bmi  →  Low Risk / High Risk
    """
    try:
        from xgboost import XGBClassifier
    except ImportError:
        log.warning("xgboost not installed — falling back to RandomForest.")
        from sklearn.ensemble import RandomForestClassifier as XGBClassifier

    log.info("Training Risk Predictor (XGBoost)...")

    # Encode labels: Low Risk → 0, High Risk → 1
    le = LabelEncoder()
    y  = le.fit_transform(df["risk_category"])

    # Scale full feature set
    scaler_full = StandardScaler()
    X = scaler_full.fit_transform(df[RISK_FEATURES].values)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    try:
        model = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
    except TypeError:
        # older xgboost
        model = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            random_state=42,
        )

    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    preds = model.predict(X_test)
    log.info("\n" + classification_report(y_test, preds,
                                          target_names=le.classes_))

    artifact = {"model": model, "scaler": scaler_full, "label_encoder": le}
    with open(RISK_PREDICTOR_PATH, "wb") as f:
        pickle.dump(artifact, f)
    with open(SCALER_FULL_PATH, "wb") as f:
        pickle.dump(scaler_full, f)
    log.info(f"Risk predictor saved → {RISK_PREDICTOR_PATH}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. MASTER TRAINING ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def train_all_models(max_rows: int = 50_000) -> None:
    """
    Complete training pipeline:
      1. Load & preprocess dataset
      2. Fit vitals scaler
      3. Train LSTM forecaster
      4. Train anomaly detector (IsoForest)
      5. Train risk predictor (XGBoost)
    """
    ensure_model_dir()

    # ── Load & preprocess ─────────────────────────────────────────────────
    df_raw  = load_dataset(max_rows=max_rows)
    df      = preprocess(df_raw)

    # ── Fit vitals scaler ─────────────────────────────────────────────────
    log.info("Fitting StandardScaler on vital features...")
    scaler_vitals = StandardScaler()
    scaler_vitals.fit(df[VITAL_FEATURES].values)
    with open(SCALER_VITALS_PATH, "wb") as f:
        pickle.dump(scaler_vitals, f)
    log.info(f"Vitals scaler saved → {SCALER_VITALS_PATH}")

    # ── LSTM sequences ────────────────────────────────────────────────────
    X, y = build_sequences(df, scaler_vitals, seq_len=SEQ_LEN)
    split = int(len(X) * 0.9)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    train_lstm(X_train, y_train, X_val, y_val)

    # ── Anomaly detector ──────────────────────────────────────────────────
    train_anomaly_detector(df, scaler_vitals)

    # ── Risk predictor ────────────────────────────────────────────────────
    train_risk_predictor(df)

    log.info("=" * 60)
    log.info("All models trained and saved to ./models/")
    log.info("=" * 60)


if __name__ == "__main__":
    train_all_models()