"""
04_train_baseline_model.py

Trains on gold_train.csv (2005-2007 vintages) ONLY. The 2011-2012 monitor
set stays completely untouched here -- phase 6 is where it gets scored,
and it needs to be genuinely "unseen" for the drift story to mean anything.

Trains two models for comparison:
  - Logistic Regression: the model we actually document and monitor going
    forward. Chosen deliberately over the more complex model -- in a
    regulated lending context, a coefficient you can explain to an auditor
    is often worth more than a percentage point of AUC.
  - HistGradientBoostingClassifier: a stronger benchmark, kept only as a
    reference point in the Model Card ("we chose interpretability over the
    ~X point AUC the boosted model would have bought us").

Outputs:
  model.pkl              -- the trained logistic regression pipeline
  metrics.json           -- validation metrics for both models
  feature_importance.csv -- logistic regression coefficients, sorted
  shap_summary.png       -- SHAP feature importance plot

Run after phase 3:  python 04_train_baseline_model.py
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, precision_score, recall_score, confusion_matrix

from config import GOLD_DIR, MODEL_DIR, TARGET_COL

RANDOM_STATE = 42


def load_gold_train():
    df = pd.read_csv(os.path.join(GOLD_DIR, "gold_train.csv"))
    with open(os.path.join(GOLD_DIR, "feature_columns.json")) as f:
        feature_cols = json.load(f)
    X = df[feature_cols].fillna(0)
    y = df[TARGET_COL]
    return X, y, feature_cols


def evaluate(model, X_val, y_val):
    proba = model.predict_proba(X_val)[:, 1]
    preds = (proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_val, preds).ravel()
    return {
        "auc": round(roc_auc_score(y_val, proba), 4),
        "precision": round(precision_score(y_val, preds, zero_division=0), 4),
        "recall": round(recall_score(y_val, preds, zero_division=0), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


if __name__ == "__main__":
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Loading gold_train.csv...")
    X, y, feature_cols = load_gold_train()
    print(f"  {len(X):,} loans, {len(feature_cols)} features, default rate {y.mean():.4f}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # --- Logistic Regression: the documented, monitored model ---
    print("\nTraining Logistic Regression (primary model)...")
    logreg_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ])
    logreg_pipeline.fit(X_train, y_train)
    logreg_metrics = evaluate(logreg_pipeline, X_val, y_val)
    print(f"  Validation AUC: {logreg_metrics['auc']}")

    # --- HistGradientBoosting: benchmark only, not deployed ---
    print("\nTraining HistGradientBoostingClassifier (benchmark only)...")
    gbm = HistGradientBoostingClassifier(random_state=RANDOM_STATE)
    gbm.fit(X_train, y_train)
    gbm_metrics = evaluate(gbm, X_val, y_val)
    print(f"  Validation AUC: {gbm_metrics['auc']}")

    # --- Save the chosen model ---
    model_path = os.path.join(MODEL_DIR, "model.pkl")
    joblib.dump(logreg_pipeline, model_path)
    print(f"\nSaved {model_path}")

    metrics = {
        "primary_model": "Logistic Regression",
        "logistic_regression": logreg_metrics,
        "hist_gradient_boosting_benchmark": gbm_metrics,
        "n_train": len(X_train),
        "n_validation": len(X_val),
        "train_default_rate": round(float(y_train.mean()), 4),
    }
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics.json")

    # --- Feature importance (logistic regression coefficients) ---
    coefs = logreg_pipeline.named_steps["clf"].coef_[0]
    importance_df = pd.DataFrame({"feature": feature_cols, "coefficient": coefs})
    importance_df["abs_coefficient"] = importance_df["coefficient"].abs()
    importance_df = importance_df.sort_values("abs_coefficient", ascending=False).drop(columns="abs_coefficient")
    importance_df.to_csv(os.path.join(MODEL_DIR, "feature_importance.csv"), index=False)
    print("Saved feature_importance.csv")

    # --- SHAP summary plot ---
    print("\nComputing SHAP values (this can take a minute)...")
    import shap
    X_val_scaled = logreg_pipeline.named_steps["scaler"].transform(X_val)
    explainer = shap.LinearExplainer(logreg_pipeline.named_steps["clf"], X_val_scaled)
    shap_values = explainer.shap_values(X_val_scaled)

    plt.figure()
    shap.summary_plot(shap_values, X_val, feature_names=feature_cols, show=False, max_display=15)
    plt.tight_layout()
    shap_path = os.path.join(MODEL_DIR, "shap_summary.png")
    plt.savefig(shap_path, dpi=120)
    plt.close()
    print(f"Saved {shap_path}")

    print("\nTop 10 features by coefficient magnitude:")
    print(importance_df.head(10).to_string(index=False))
