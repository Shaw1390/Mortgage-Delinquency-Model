

import os
import pandas as pd



CLEANED_DIR = r"C:\Users\shawa\Downloads\cleaned"


SERIOUS_DELINQUENCY_THRESHOLD = 3  


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
