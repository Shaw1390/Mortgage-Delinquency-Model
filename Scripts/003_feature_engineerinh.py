"""
03_feature_engineering.py  (Gold layer)

Takes loan_level_labeled.csv (output of phases 1-2) and builds the
model-ready "Gold" table:
  - drops columns that are IDs, near-empty, or would leak information
  - buckets a few continuous features (credit score, DTI) into the
    categories underwriters actually think in
  - one-hot encodes categorical fields
  - splits into gold_train.csv (2005-2007) and gold_monitor.csv (2011-2012)
    using the same encoded column set, so the monitor set can be scored
    by a model trained only on gold_train.csv

Run after phase 2:  python 03_feature_engineering.py
"""

import os
import json
import pandas as pd

from config import CLEANED_DIR, GOLD_DIR, TRAIN_YEARS, MONITOR_YEARS, TARGET_COL

# Columns we deliberately do NOT feed to the model:
#   - loan_id: identifier, not a feature
#   - seller_name: almost entirely "OTHER" in the public dataset, no signal
#   - first_payment_date / maturity_date: redundant with vintage_year + loan term
#   - msa / postal_code: high-cardinality / mostly blank in the sample data
#   - pre_harp_loan_id / special_eligibility_program / harp_indicator: not
#     relevant for 2005-2012 originations, HARP didn't exist yet
#   - property_valuation_method: valuation method, not a risk driver at origination
#   - vantage_score_4_0: almost entirely "Not Available" for this era, see
#     the credit_score sanity check from phase 1
#   - super_conforming_flag: near-constant "N" for this sample -- no signal
DROP_COLS = [
    "loan_id", "seller_name", "first_payment_date", "maturity_date",
    "msa", "postal_code", "pre_harp_loan_id", "special_eligibility_program",
    "harp_indicator", "property_valuation_method", "vantage_score_4_0",
    "super_conforming_flag",
]

CATEGORICAL_COLS = [
    "first_time_homebuyer_flag", "occupancy_status", "channel", "ppm_flag",
    "amortization_type", "property_region", "property_type", "loan_purpose",
    "io_indicator", "credit_score_bucket", "dti_bucket",
]

# Standard US Census Bureau 4-region grouping -- keeps property_state's
# information without a 50-state-wide one-hot explosion.
STATE_TO_REGION = {
    "CT": "Northeast", "ME": "Northeast", "MA": "Northeast", "NH": "Northeast",
    "RI": "Northeast", "VT": "Northeast", "NJ": "Northeast", "NY": "Northeast", "PA": "Northeast",
    "IL": "Midwest", "IN": "Midwest", "MI": "Midwest", "OH": "Midwest", "WI": "Midwest",
    "IA": "Midwest", "KS": "Midwest", "MN": "Midwest", "MO": "Midwest", "NE": "Midwest",
    "ND": "Midwest", "SD": "Midwest",
    "DE": "South", "FL": "South", "GA": "South", "MD": "South", "NC": "South",
    "SC": "South", "VA": "South", "DC": "South", "WV": "South", "AL": "South",
    "KY": "South", "MS": "South", "TN": "South", "AR": "South", "LA": "South",
    "OK": "South", "TX": "South",
    "AZ": "West", "CO": "West", "ID": "West", "MT": "West", "NV": "West",
    "NM": "West", "UT": "West", "WY": "West", "AK": "West", "CA": "West",
    "HI": "West", "OR": "West", "WA": "West",
}


def bucket_credit_score(score):
    if pd.isna(score):
        return "Unknown"
    if score < 620:
        return "Subprime (<620)"
    if score < 680:
        return "Near-prime (620-679)"
    if score < 740:
        return "Prime (680-739)"
    if score < 800:
        return "Super-prime (740-799)"
    return "Excellent (800+)"


def bucket_dti(dti):
    if pd.isna(dti):
        return "Unknown"
    if dti < 30:
        return "Low (<30)"
    if dti < 43:
        return "Moderate (30-42)"
    return "High (43+)"


def engineer_features(df):
    df = df.copy()

    df["credit_score_bucket"] = df["credit_score"].apply(bucket_credit_score)
    df["dti_bucket"] = df["original_dti"].apply(bucket_dti)
    df["high_ltv_flag"] = (df["original_ltv"] >= 90).astype(int)
    df["property_region"] = df["property_state"].map(STATE_TO_REGION).fillna("Other")

    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    df = df.drop(columns=["property_state"])  # replaced by property_region

    df = pd.get_dummies(df, columns=[c for c in CATEGORICAL_COLS if c in df.columns], drop_first=False)

    return df


if __name__ == "__main__":
    os.makedirs(GOLD_DIR, exist_ok=True)

    in_path = os.path.join(CLEANED_DIR, "loan_level_labeled.csv")
    df = pd.read_csv(in_path)
    print(f"Loaded {len(df):,} labeled loans")

    gold_df = engineer_features(df)

    feature_cols = [c for c in gold_df.columns if c not in (TARGET_COL, "vintage_year")]
    print(f"Engineered feature set: {len(feature_cols)} columns")

    train_df = gold_df[gold_df["vintage_year"].isin(TRAIN_YEARS)].reset_index(drop=True)
    monitor_df = gold_df[gold_df["vintage_year"].isin(MONITOR_YEARS)].reset_index(drop=True)

    train_out = os.path.join(GOLD_DIR, "gold_train.csv")
    monitor_out = os.path.join(GOLD_DIR, "gold_monitor.csv")
    train_df.to_csv(train_out, index=False)
    monitor_df.to_csv(monitor_out, index=False)

    with open(os.path.join(GOLD_DIR, "feature_columns.json"), "w") as f:
        json.dump(feature_cols, f, indent=2)

    print(f"\nSaved {train_out}  ({len(train_df):,} loans, vintages {TRAIN_YEARS})")
    print(f"Saved {monitor_out}  ({len(monitor_df):,} loans, vintages {MONITOR_YEARS})")
    print(f"Saved feature_columns.json ({len(feature_cols)} features)")
