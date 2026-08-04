"""
01_load_and_clean.py

Loads the raw pipe-delimited Freddie Mac origination + performance files,
assigns the real column names (confirmed against the official July 2026
file layout), and converts Freddie Mac's documented placeholder/"Not
Available" codes into real nulls so they don't get treated as actual
values downstream (e.g. a credit score of 9999, an MI% of 999).

HOW TO USE:
    1. Edit BASE_DIR and YEARS below.
    2. This expects a subfolder per year, e.g.:
           <BASE_DIR>/sample_2005/sample_orig_2005.txt
           <BASE_DIR>/sample_2005/sample_perf_2005.txt
       If your folders are named differently, adjust FOLDER_PATTERN.
    3. Run directly:
           python 01_load_and_clean.py
"""

import os
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# EDIT THESE
# ---------------------------------------------------------------------------
BASE_DIR = r"C:\Users\shawa\Downloads"
YEARS = [2005, 2007, 2006, 2011, 2012]
FOLDER_PATTERN = "sample_{year}"          # -> sample_2005, sample_2007, ...
OUTPUT_DIR = r"C:\Users\shawa\Downloads\cleaned"
# ---------------------------------------------------------------------------

ORIG_COLUMNS = [
    "credit_score", "first_payment_date", "first_time_homebuyer_flag", "maturity_date",
    "msa", "mi_pct", "num_units", "occupancy_status", "original_cltv", "original_dti",
    "original_upb", "original_ltv", "original_interest_rate", "channel", "ppm_flag",
    "amortization_type", "property_state", "property_type", "postal_code", "loan_id",
    "loan_purpose", "original_loan_term", "num_borrowers", "seller_name",
    "super_conforming_flag", "pre_harp_loan_id", "special_eligibility_program",
    "harp_indicator", "property_valuation_method", "io_indicator", "vantage_score_4_0",
]

PERF_COLUMNS = [
    "loan_id", "reporting_period", "current_actual_upb", "current_delinquency_status",
    "loan_age", "remaining_months_to_maturity", "defect_settlement_date", "modification_flag",
    "zero_balance_code", "zero_balance_effective_date", "current_interest_rate",
    "current_non_interest_bearing_upb", "ddlpi", "mi_recoveries", "net_sales_proceeds",
    "non_mi_recoveries", "total_expenses", "legal_costs", "maintenance_preservation_costs",
    "taxes_and_insurance", "misc_expenses", "actual_loss", "cumulative_modification_costs",
    "interest_rate_step_indicator", "payment_deferral_flag", "eltv", "zero_balance_removal_upb",
    "delinquent_accrued_interest", "delinquency_due_to_disaster", "borrower_assistance_plan",
    "current_period_modification_costs", "current_interest_bearing_upb",
    "mi_cancellation_indicator", "servicer_name", "bankruptcy_cramdown_costs",
]

# Freddie Mac's documented "Not Available" sentinel values, per field.
# Anything listed here gets swapped for a real null before we do any modeling.
ORIG_NA_VALUES = {
    "credit_score": ["9999"],
    "mi_pct": ["999"],
    "num_units": ["99"],
    "original_cltv": ["999"],
    "original_dti": ["999"],
    "original_ltv": ["999"],
    "num_borrowers": ["99"],
    "vantage_score_4_0": ["9999"],
    "postal_code": [""],
    "msa": [""],
}

PERF_NA_VALUES = {
    "eltv": ["999", "9999"],
    "current_delinquency_status": ["XX", ""],
}

ORIG_NUMERIC_COLS = [
    "credit_score", "mi_pct", "num_units", "original_cltv", "original_dti",
    "original_upb", "original_ltv", "original_interest_rate", "original_loan_term",
    "num_borrowers", "vantage_score_4_0",
]

PERF_NUMERIC_COLS = [
    "current_actual_upb", "loan_age", "remaining_months_to_maturity",
    "current_interest_rate", "current_non_interest_bearing_upb", "eltv",
    "current_interest_bearing_upb",
]


def load_raw(filepath, column_names):
    df = pd.read_csv(
        filepath,
        sep="|",
        header=None,
        names=column_names,
        dtype=str,           # read everything as string first -- safest for cleaning sentinels
        na_filter=False,     # keep blanks as "" rather than NaN so our NA-value maps can catch them
        encoding="latin-1",
    )
    return df


def clean_na_values(df, na_map):
    for col, sentinels in na_map.items():
        if col in df.columns:
            df[col] = df[col].replace(sentinels, np.nan)
    # Any remaining blank strings across the whole frame become real nulls too
    df = df.replace("", np.nan)
    return df


def cast_numeric(df, numeric_cols):
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_year(base_dir, year, folder_pattern):
    folder = os.path.join(base_dir, folder_pattern.format(year=year))
    orig_path = os.path.join(folder, f"sample_orig_{year}.txt")
    perf_path = os.path.join(folder, f"sample_perf_{year}.txt")

    if not os.path.exists(orig_path) or not os.path.exists(perf_path):
        print(f"  [{year}] SKIPPED -- expected files not found in {folder}")
        return None, None

    orig_df = load_raw(orig_path, ORIG_COLUMNS)
    orig_df = clean_na_values(orig_df, ORIG_NA_VALUES)
    orig_df = cast_numeric(orig_df, ORIG_NUMERIC_COLS)
    orig_df["vintage_year"] = year

    perf_df = load_raw(perf_path, PERF_COLUMNS)
    perf_df = clean_na_values(perf_df, PERF_NA_VALUES)
    perf_df = cast_numeric(perf_df, PERF_NUMERIC_COLS)
    perf_df["vintage_year"] = year

    print(f"  [{year}] orig: {len(orig_df):,} loans | perf: {len(perf_df):,} monthly records")
    return orig_df, perf_df


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_orig, all_perf = [], []

    print("Loading and cleaning each vintage year...")
    for year in YEARS:
        orig_df, perf_df = load_year(BASE_DIR, year, FOLDER_PATTERN)
        if orig_df is not None:
            all_orig.append(orig_df)
            all_perf.append(perf_df)

    if not all_orig:
        print("\nNothing loaded -- check BASE_DIR / FOLDER_PATTERN at the top of this file.")
    else:
        orig_combined = pd.concat(all_orig, ignore_index=True)
        perf_combined = pd.concat(all_perf, ignore_index=True)

        orig_out = os.path.join(OUTPUT_DIR, "origination_clean.csv")
        perf_out = os.path.join(OUTPUT_DIR, "performance_clean.csv")
        orig_combined.to_csv(orig_out, index=False)
        perf_combined.to_csv(perf_out, index=False)

        print(f"\nSaved {orig_out}  ({len(orig_combined):,} loans total)")
        print(f"Saved {perf_out}  ({len(perf_combined):,} monthly records total)")

        print("\nQuick sanity check -- credit score stats after cleaning:")
        print(orig_combined["credit_score"].describe())