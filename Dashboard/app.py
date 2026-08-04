"""
dashboard/app.py

Streamlit dashboard for the model governance & monitoring project. Reads
the outputs of phases 3-6 -- doesn't retrain or recompute anything itself,
it's purely a visualization layer on top of files that already exist.

Run:
    streamlit run app.py
(run this FROM INSIDE the dashboard/ folder, or adjust the paths below)
"""

import os
import sys
import json

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import GOLD_DIR, MODEL_DIR, REPORTS_DIR, TRAIN_YEARS, MONITOR_YEARS

st.set_page_config(page_title="RESL Model Governance & Monitoring", layout="wide")

st.title("Mortgage Delinquency Model \u2014 Governance & Monitoring Dashboard")
st.caption(
    f"Trained on {', '.join(str(y) for y in TRAIN_YEARS)} vintages \u00b7 "
    f"Monitored against {', '.join(str(y) for y in MONITOR_YEARS)} vintages"
)


@st.cache_data
def load_all():
    with open(os.path.join(MODEL_DIR, "metrics.json")) as f:
        metrics = json.load(f)
    drift_path = os.path.join(REPORTS_DIR, "drift_report.json")
    governance_path = os.path.join(REPORTS_DIR, "governance_status.csv")
    drift = json.load(open(drift_path)) if os.path.exists(drift_path) else None
    governance_df = pd.read_csv(governance_path) if os.path.exists(governance_path) else None
    importance_df = pd.read_csv(os.path.join(MODEL_DIR, "feature_importance.csv"))
    train_df = pd.read_csv(os.path.join(GOLD_DIR, "gold_train.csv"))
    monitor_df = pd.read_csv(os.path.join(GOLD_DIR, "gold_monitor.csv"))
    return metrics, drift, governance_df, importance_df, train_df, monitor_df


try:
    metrics, drift, governance_df, importance_df, train_df, monitor_df = load_all()
except FileNotFoundError as e:
    st.error(
        f"Missing an expected file: {e}\n\n"
        "Run phases 3-6 first (03_feature_engineering.py through "
        "06_monitoring_drift.py) so this dashboard has something to read."
    )
    st.stop()

# --- Row 1: headline metrics ---
col1, col2, col3, col4 = st.columns(4)
lr = metrics["logistic_regression"]
col1.metric("Validation AUC", lr["auc"])
col2.metric("Validation Precision", lr["precision"])
col3.metric("Validation Recall", lr["recall"])
if drift:
    mp = drift["monitor_performance"]
    col4.metric("Monitor Set AUC", mp["auc"], delta=round(mp["auc"] - lr["auc"], 4))

st.divider()

# --- Row 2: governance status table ---
st.subheader("Governance Status")
if governance_df is not None:
    def highlight_status(val):
        if val == "Stable" or val == "Within tolerance":
            return "background-color: #d4edda"
        if "Flag" in str(val):
            return "background-color: #fff3cd"
        if "Trigger" in str(val) or "Escalate" in str(val):
            return "background-color: #f8d7da"
        return ""

    try:
        styled = governance_df.style.map(highlight_status, subset=["governance_action"])
    except AttributeError:
        styled = governance_df.style.applymap(highlight_status, subset=["governance_action"])

    st.dataframe(styled, use_container_width=True)
else:
    st.info("Run 06_monitoring_drift.py to populate governance status.")

st.divider()

# --- Row 3: feature drift + default rate by vintage ---
left, right = st.columns(2)

with left:
    st.subheader("Feature Drift (PSI)")
    if drift:
        drift_features = drift["feature_drift"]
        drift_df = pd.DataFrame(
            [{"feature": k, "psi": v["psi"], "status": v["status"]} for k, v in drift_features.items()]
        ).sort_values("psi", ascending=True)

        fig, ax = plt.subplots(figsize=(6, 4))
        colors = drift_df["psi"].apply(
            lambda v: "#d4edda" if v < 0.10 else ("#fff3cd" if v < 0.25 else "#f8d7da")
        )
        ax.barh(drift_df["feature"], drift_df["psi"], color=colors, edgecolor="#333")
        ax.axvline(0.10, color="gray", linestyle="--", linewidth=1)
        ax.axvline(0.25, color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel("PSI (train vs. monitor)")
        st.pyplot(fig)
    else:
        st.info("Run 06_monitoring_drift.py to populate drift metrics.")

with right:
    st.subheader("Default Rate by Vintage")
    combined = pd.concat([
        train_df.assign(period="Train"),
        monitor_df.assign(period="Monitor"),
    ])
    rate_by_year = combined.groupby("vintage_year")["ever_90dpd"].mean().reset_index()
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    bar_colors = ["#1F3864" if y in TRAIN_YEARS else "#8FAADC" for y in rate_by_year["vintage_year"]]
    ax2.bar(rate_by_year["vintage_year"].astype(str), rate_by_year["ever_90dpd"], color=bar_colors)
    ax2.set_ylabel("Ever 90+ DPD rate")
    ax2.set_xlabel("Vintage year")
    st.pyplot(fig2)

st.divider()

# --- Row 4: prediction score distribution, train vs monitor ---
st.subheader("Prediction Score Distribution: Train vs. Monitor")
import joblib
with open(os.path.join(GOLD_DIR, "feature_columns.json")) as f:
    feature_cols = json.load(f)
model = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
train_scores = model.predict_proba(train_df[feature_cols].fillna(0))[:, 1]
monitor_scores = model.predict_proba(monitor_df[feature_cols].fillna(0))[:, 1]

fig3, ax3 = plt.subplots(figsize=(10, 3.5))
ax3.hist(train_scores, bins=30, alpha=0.6, label="Train (2005-2007)", color="#1F3864")
ax3.hist(monitor_scores, bins=30, alpha=0.6, label="Monitor (2011-2012)", color="#8FAADC")
ax3.set_xlabel("Predicted default probability")
ax3.legend()
st.pyplot(fig3)

st.divider()

# --- Row 5: feature importance ---
st.subheader("Feature Importance (Logistic Regression Coefficients)")
top_features = importance_df.head(15)
fig4, ax4 = plt.subplots(figsize=(10, 5))
colors4 = top_features["coefficient"].apply(lambda v: "#c0392b" if v > 0 else "#1F3864")
ax4.barh(top_features["feature"][::-1], top_features["coefficient"][::-1], color=colors4[::-1])
ax4.set_xlabel("Coefficient (positive = higher default risk)")
st.pyplot(fig4)
