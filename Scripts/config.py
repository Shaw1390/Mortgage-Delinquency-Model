"""
config.py
Shared paths and constants for phases 3-7. Phases 1-2 (01_load_and_clean.py,
02_build_target_label.py) already produced loan_level_labeled.csv in
CLEANED_DIR -- everything downstream starts from that file.
"""

# ---------------------------------------------------------------------------
# EDIT THESE IF YOUR FOLDERS ARE DIFFERENT
# ---------------------------------------------------------------------------
CLEANED_DIR = r"C:\Users\shawa\Downloads\cleaned"    # output of phases 1-2
GOLD_DIR = r"C:\Users\shawa\Downloads\gold"          # output of phase 3
MODEL_DIR = r"C:\Users\shawa\Downloads\model"        # output of phase 4
REPORTS_DIR = r"C:\Users\shawa\Downloads\reports"    # output of phases 5-6
# ---------------------------------------------------------------------------

# Which vintages train the model vs. which ones we monitor it against.
# This is the pre-crisis vs. post-crisis split that drives the whole
# monitoring/drift story.
TRAIN_YEARS = [2005, 2006, 2007]
MONITOR_YEARS = [2011, 2012]

TARGET_COL = "ever_90dpd"

# Governance thresholds -- used by both 06_monitoring_drift.py and the
# dashboard. Keeping them here means the code and the Model Card reference
# the exact same numbers.
PSI_STABLE_MAX = 0.10       # below this: no action
PSI_FLAG_MAX = 0.25         # between STABLE and this: flag for monitoring
                             # above this: trigger an off-cycle review
AUC_DROP_ESCALATE = 0.05    # absolute AUC drop vs. validation that triggers escalation
