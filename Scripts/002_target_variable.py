"""
02_build_target_label.py

Builds the target variable: "ever_90dpd" -- did this loan reach 90+ days
past due, or REO, at ANY point in its performance history. This collapses
the many-rows-per-loan performance file down to one label per loan, which
is what lets you join it onto the one-row-per-loan origination file.

current_delinquency_status is coded as months past due:
    "0"      = current
    "1"      = 30 days late
    "2"      = 60 days late
    "3"      = 90 days late
    "4","5"...= further past due
    "R"      = REO (bank-owned, post-foreclosure)

We treat 3+ months past due OR "R" as "seriously delinquent."

HOW TO USE:
    1. Make sure 01_load_and_clean.py has already run and produced
       origination_clean.csv and performance_clean.csv.
    2. Edit CLEANED_DIR below if needed.
    3. Run directly: python 02_build_target_label.py
"""

import os
import pandas as pd

# ---------------------------------------------------------------------------
# EDIT THIS IF NEEDED -- should match OUTPUT_DIR from the previous script
# ---------------------------------------------------------------------------
CLEANED_DIR = r"C:\Users\shawa\Downloads\cleaned"
# ---------------------------------------------------------------------------

SERIOUS_DELINQUENCY_THRESHOLD = 3  # months past due (i.e. 90+ days)


def delinquency_severity(status):
    """
    Converts current_delinquency_status into a comparable severity number.
    'R' (REO) is treated as more severe than any months-past-due count.
    Anything unparseable comes back as -1 (i.e. "unknown", not counted
    as delinquent -- we don't want a data quality issue to inflate the
    default rate).
    """
    if pd.isna(status):
        return -1
    status = str(status).strip()
    if status == "R":
        return 999
    try:
        return int(status)
    except ValueError:
        return -1


def build_target_label(perf_df):
    perf_df = perf_df.copy()
    perf_df["_severity"] = perf_df["current_delinquency_status"].apply(delinquency_severity)

    loan_max_severity = perf_df.groupby("loan_id")["_severity"].max().reset_index()
    loan_max_severity["ever_90dpd"] = (
        loan_max_severity["_severity"] >= SERIOUS_DELINQUENCY_THRESHOLD
    ).astype(int)

    return loan_max_severity[["loan_id", "ever_90dpd"]]


if __name__ == "__main__":
    orig_path = os.path.join(CLEANED_DIR, "origination_clean.csv")
    perf_path = os.path.join(CLEANED_DIR, "performance_clean.csv")

    print("Loading cleaned files...")
    orig_df = pd.read_csv(orig_path)
    perf_df = pd.read_csv(perf_path, dtype={"current_delinquency_status": str})

    print(f"  origination_clean.csv: {len(orig_df):,} loans")
    print(f"  performance_clean.csv: {len(perf_df):,} monthly records")

    print("\nBuilding ever_90dpd label from performance history...")
    labels_df = build_target_label(perf_df)

    loan_level_df = orig_df.merge(labels_df, on="loan_id", how="left")

    # Any loan with no performance records at all has an unknown outcome --
    # flag it rather than silently assuming it's "good."
    missing_label = loan_level_df["ever_90dpd"].isna().sum()
    if missing_label > 0:
        print(f"\n  Note: {missing_label} loans had no matching performance records -- dropping them.")
    loan_level_df = loan_level_df.dropna(subset=["ever_90dpd"])
    loan_level_df["ever_90dpd"] = loan_level_df["ever_90dpd"].astype(int)

    out_path = os.path.join(CLEANED_DIR, "loan_level_labeled.csv")
    loan_level_df.to_csv(out_path, index=False)

    print(f"\nSaved {out_path}")
    print(f"Total loans: {len(loan_level_df):,}")

    print("\nOverall default rate:")
    print(loan_level_df["ever_90dpd"].value_counts(normalize=True).rename("share").to_string())

    if "vintage_year" in loan_level_df.columns:
        print("\nDefault rate by vintage year (this is your monitoring-dashboard preview --")
        print("watch how much higher 2007/2008 originations run vs 2011/2012):")
        print(
            loan_level_df.groupby("vintage_year")["ever_90dpd"]
            .mean()
            .rename("default_rate")
            .round(4)
            .to_string()
        )