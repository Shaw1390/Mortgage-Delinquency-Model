import os
import json
import joblib
import numpy as np
import pandas as pd

from config import (
    GOLD_DIR, MODEL_DIR, REPORTS_DIR,
    PSI_STABLE_MAX, PSI_FLAG_MAX, AUC_DROP_ESCALATE, TARGET_COL,
)
from sklearn.metrics import roc_auc_score, precision_score, recall_score

DRIFT_TRACKED_FEATURES = [
    "credit_score", "original_dti", "original_ltv",
    "original_interest_rate", "original_upb",
]


def population_stability_index(expected, actual, bins=10):
    """
    Standard PSI calculation: bin the EXPECTED (training) distribution into
    deciles, then compare what share of ACTUAL (monitor) values fall into
    those same bins.
    PSI < 0.10  -> no significant shift
    PSI 0.10-0.25 -> moderate shift, worth watching
    PSI > 0.25  -> significant shift
    """
    expected = pd.Series(expected).dropna()
    actual = pd.Series(actual).dropna()

    quantiles = np.linspace(0, 1, bins + 1)
    bin_edges = np.unique(expected.quantile(quantiles).values)
    if len(bin_edges) < 3:
        return 0.0  # not enough spread to compute meaningfully

    expected_counts, _ = np.histogram(expected, bins=bin_edges)
    actual_counts, _ = np.histogram(actual, bins=bin_edges)

    expected_pct = np.where(expected_counts == 0, 1e-4, expected_counts / expected_counts.sum())
    actual_pct = np.where(actual_counts == 0, 1e-4, actual_counts / actual_counts.sum())

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return round(float(psi), 4)


def psi_status(psi_value):
    if psi_value < PSI_STABLE_MAX:
        return "Stable"
    if psi_value < PSI_FLAG_MAX:
        return "Flag for monitoring"
    return "Trigger off-cycle review"


if __name__ == "__main__":
    os.makedirs(REPORTS_DIR, exist_ok=True)

    print("Loading train/monitor gold sets and trained model...")
    train_df = pd.read_csv(os.path.join(GOLD_DIR, "gold_train.csv"))
    monitor_df = pd.read_csv(os.path.join(GOLD_DIR, "gold_monitor.csv"))
    with open(os.path.join(GOLD_DIR, "feature_columns.json")) as f:
        feature_cols = json.load(f)
    with open(os.path.join(MODEL_DIR, "metrics.json")) as f:
        train_metrics = json.load(f)
    model = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))

    # --- 1. Data drift on key raw features ---
    print("\nComputing feature drift (PSI)...")
    feature_drift = {}
    for feat in DRIFT_TRACKED_FEATURES:
        psi = population_stability_index(train_df[feat], monitor_df[feat])
        feature_drift[feat] = {"psi": psi, "status": psi_status(psi)}
        print(f"  {feat:28s} PSI={psi:.4f}  -> {psi_status(psi)}")

    # --- 2. Prediction score drift ---
    X_train = train_df[feature_cols].fillna(0)
    X_monitor = monitor_df[feature_cols].fillna(0)
    y_monitor = monitor_df[TARGET_COL]

    train_scores = model.predict_proba(X_train)[:, 1]
    monitor_scores = model.predict_proba(X_monitor)[:, 1]
    score_psi = population_stability_index(train_scores, monitor_scores)
    print(f"\nPrediction score PSI: {score_psi:.4f} -> {psi_status(score_psi)}")

    # --- 3. Performance decay on monitor set (real outcomes, since these loans have played out) ---
    monitor_preds = (monitor_scores >= 0.5).astype(int)
    monitor_auc = round(roc_auc_score(y_monitor, monitor_scores), 4)
    monitor_precision = round(precision_score(y_monitor, monitor_preds, zero_division=0), 4)
    monitor_recall = round(recall_score(y_monitor, monitor_preds, zero_division=0), 4)

    validation_auc = train_metrics["logistic_regression"]["auc"]
    auc_drop = round(validation_auc - monitor_auc, 4)
    auc_status = "Escalate to model risk owner" if auc_drop > AUC_DROP_ESCALATE else "Within tolerance"

    print(f"\nMonitor set performance (2011-2012, real outcomes):")
    print(f"  AUC: {monitor_auc}  (validation was {validation_auc}, drop = {auc_drop})")
    print(f"  Precision: {monitor_precision}  Recall: {monitor_recall}")
    print(f"  Status: {auc_status}")

    # --- Assemble drift report ---
    drift_report = {
        "feature_drift": feature_drift,
        "prediction_score_psi": {"psi": score_psi, "status": psi_status(score_psi)},
        "monitor_performance": {
            "auc": monitor_auc,
            "precision": monitor_precision,
            "recall": monitor_recall,
            "validation_auc": validation_auc,
            "auc_drop": auc_drop,
            "status": auc_status,
        },
        "monitor_default_rate": round(float(y_monitor.mean()), 4),
        "train_default_rate": train_metrics["train_default_rate"],
    }
    with open(os.path.join(REPORTS_DIR, "drift_report.json"), "w") as f:
        json.dump(drift_report, f, indent=2)
    print(f"\nSaved {os.path.join(REPORTS_DIR, 'drift_report.json')}")

    # --- Governance status table -- the signal -> threshold -> action mapping ---
    rows = []
    for feat, d in feature_drift.items():
        rows.append({
            "monitoring_signal": f"PSI: {feat}",
            "value": d["psi"],
            "threshold": f"< {PSI_STABLE_MAX} stable / < {PSI_FLAG_MAX} flag / >= {PSI_FLAG_MAX} trigger",
            "governance_action": d["status"],
        })
    rows.append({
        "monitoring_signal": "PSI: prediction score",
        "value": score_psi,
        "threshold": f"< {PSI_STABLE_MAX} stable / < {PSI_FLAG_MAX} flag / >= {PSI_FLAG_MAX} trigger",
        "governance_action": psi_status(score_psi),
    })
    rows.append({
        "monitoring_signal": "AUC drop vs. validation",
        "value": auc_drop,
        "threshold": f">= {AUC_DROP_ESCALATE} escalate",
        "governance_action": auc_status,
    })
    governance_df = pd.DataFrame(rows)
    governance_path = os.path.join(REPORTS_DIR, "governance_status.csv")
    governance_df.to_csv(governance_path, index=False)
    print(f"Saved {governance_path}")
