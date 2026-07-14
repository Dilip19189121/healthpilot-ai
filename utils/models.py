"""
ML utilities for HealthPilot AI.

Three classic ML topics are demonstrated on the same clinical dataset:
- Classification   -> Risk Classifier (Logistic Regression)
- Regression        -> Metric Regressor (Linear Regression)
- Anomaly detection -> Anomaly Scanner (Isolation Forest)
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, r2_score, mean_absolute_error
from sklearn.ensemble import IsolationForest

DATA_PATH = "data/diabetes.csv"

FEATURE_COLUMNS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]


def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def train_classifier(df: pd.DataFrame) -> dict:
    """Logistic Regression: classifies diabetes risk (Outcome 0/1)."""
    X = df[FEATURE_COLUMNS]
    y = df["Outcome"]

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    Xtr, Xte, ytr, yte = train_test_split(
        Xs, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(Xtr, ytr)

    pred = model.predict(Xte)
    proba = model.predict_proba(Xte)[:, 1]

    return {
        "model": model,
        "scaler": scaler,
        "accuracy": accuracy_score(yte, pred),
        "auc": roc_auc_score(yte, proba),
        "features": FEATURE_COLUMNS,
    }


def train_regressor(df: pd.DataFrame, target: str = "Glucose") -> dict:
    """Linear Regression: predicts a continuous clinical value from the rest of the chart."""
    features = [c for c in FEATURE_COLUMNS if c != target]
    X = df[features]
    y = df[target]

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(Xtr, ytr)

    pred = model.predict(Xte)

    return {
        "model": model,
        "scaler": scaler,
        "r2": r2_score(yte, pred),
        "mae": mean_absolute_error(yte, pred),
        "features": features,
        "target": target,
    }


def train_anomaly_detector(df: pd.DataFrame) -> dict:
    """Isolation Forest: flags vitals that look statistically unusual vs. the population."""
    X = df[FEATURE_COLUMNS]
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=200, contamination=0.08, random_state=42
    )
    model.fit(Xs)

    return {"model": model, "scaler": scaler, "features": FEATURE_COLUMNS}


def predict_risk(clf_bundle: dict, values: dict) -> dict:
    row = np.array([[values[f] for f in clf_bundle["features"]]])
    row_s = clf_bundle["scaler"].transform(row)
    proba = clf_bundle["model"].predict_proba(row_s)[0][1]

    coefs = clf_bundle["model"].coef_[0]
    z = row_s[0]
    contributions = list(zip(clf_bundle["features"], coefs * z))
    contributions.sort(key=lambda t: abs(t[1]), reverse=True)

    return {"probability": float(proba), "contributions": contributions}


def predict_metric(reg_bundle: dict, values: dict) -> float:
    row = np.array([[values[f] for f in reg_bundle["features"]]])
    row_s = reg_bundle["scaler"].transform(row)
    return float(reg_bundle["model"].predict(row_s)[0])


def score_anomaly(anom_bundle: dict, values: dict) -> dict:
    row = np.array([[values[f] for f in anom_bundle["features"]]])
    row_s = anom_bundle["scaler"].transform(row)
    label = anom_bundle["model"].predict(row_s)[0]  # 1 normal, -1 anomaly
    score = anom_bundle["model"].decision_function(row_s)[0]  # higher = more normal
    return {"is_anomaly": label == -1, "score": float(score)}
