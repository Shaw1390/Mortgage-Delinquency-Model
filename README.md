# Mortgage Delinquency Model

**A mortgage delinquency model, wrapped in the governance and monitoring layer a real model risk team would require — built on 250,000 real Freddie Mac loans spanning the 2008 financial crisis.**

`Python` `scikit-learn` `SHAP` `Streamlit` `python-docx`

---

Most credit risk portfolio projects stop at "trained a model, got a good AUC." This one goes further: it trains a logistic regression on pre-crisis mortgage originations (2005-2007), scores it against post-crisis loans (2011-2012) it's never seen, and wraps the whole thing in the artifacts a real governance review would expect — a formal Model Card & Validation Report, a drift-monitoring dashboard, and a threshold table that ties monitoring signals directly to governance actions.

## Key Findings

**The regime shift shows up before any model gets trained.** Just from labeling outcomes across 250,000 loans, default rate by origination year tells the story on its own:

| Vintage | Ever 90+ Days Delinquent |
|---|---|
| 2005 | 11.0% |
| 2006 | 14.3% |
| 2007 | 16.5% |
| 2011 | 3.1% |
| 2012 | 3.4% |

Loans originated in 2007 defaulted at **over 5x** the rate of loans originated in 2012 — same country, same asset class, only the lending environment changed.

**Model performance (Logistic Regression, trained on 2005-2007 only):**

| Metric | Validation | Monitor Set (2011-2012) |
|---|---|---|
| AUC | 0.777 | 0.777 |
| Precision | — | 0.185 |
| Recall | — | 0.081 |

Top drivers: credit score and number of borrowers pull risk down; combined LTV, DTI, loan term, and interest rate push it up — all in the expected direction.

**The real finding is in the gap between AUC and precision/recall.** The model's ability to *rank* loans by risk barely moved — but precision and recall both collapsed on the monitor set. The model was calibrated to a ~14% default environment and got handed a ~3% one. It still knows which loans are relatively riskier; it's just wrong about how risky "risky" actually is anymore. That distinction — discrimination holding up while calibration breaks — is exactly what a monitoring system needs to catch, and it's the kind of failure an AUC-only check would miss entirely.

**Feature drift (PSI, train vs. monitor):**

| Feature | PSI | Status |
|---|---|---|
| Credit score | 0.46 | Trigger review |
| DTI | 0.88 | Trigger review |
| Interest rate | 8.15 | Trigger review |
| LTV | 0.04 | Stable |
| Loan amount (UPB) | 0.06 | Stable |
| Prediction score | 1.10 | Trigger review |

Interest rate drift is the most dramatic, for an obvious reason once you think about it: 2005-2007 mortgages were priced in a ~6%+ rate environment, and 2011-2012 loans were priced near historic lows post-crisis. Credit score and DTI drift reflect the underwriting tightening that followed — tighter QM-era standards meant materially different borrower profiles.

## What's in this repo

| Phase | What it does |
|---|---|
| 1-2 | Load and clean raw Freddie Mac data; build the target label |
| 3 | Feature engineering (Gold layer); train/monitor vintage split |
| 4 | Train and validate the model; SHAP interpretability |
| 5 | Generate the Model Card & Validation Report (`.docx`) |
| 6 | Score the monitor set; compute drift (PSI) and governance status |
| 7 | Interactive Streamlit monitoring dashboard |

Full write-up: **[link to blog post]**

## Setup

\`\`\`bash
pip install -r requirements.txt
\`\`\`

## Run order

Edit `config.py` first — it holds every file path and governance threshold used from phase 3 onward.

\`\`\`bash
python 01_load_and_clean.py
python 02_build_target_label.py
python 03_feature_engineering.py
python 04_train_baseline_model.py
python 05_generate_model_card.py      # run once now, once more after phase 6
python 06_monitoring_drift.py
python 05_generate_model_card.py      # re-run to add the monitoring section
cd dashboard && streamlit run app.py
\`\`\`

## Data source

Freddie Mac Single-Family Loan-Level Dataset (public data). This is U.S. loan-leve
