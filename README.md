# Credit Card Fraud Detection on Real Data (ULB Machine Learning Group)

## Overview
This project detects fraudulent credit card transactions in a severely
imbalanced, real-world dataset — a classic problem where naive accuracy is
meaningless and the entire methodology (metrics, splitting, resampling) has
to be built around the imbalance from the start.

## Dataset
- **Source:** [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) —
  published by the Machine Learning Group at Université Libre de Bruxelles
  (ULB), in collaboration with Worldline.
- **284,807 real transactions** by European cardholders over 2 days in
  September 2013, with **492 confirmed frauds (0.172%)**.
- Features `V1`-`V28` are anonymized via PCA for confidentiality; `Time`
  and `Amount` are the only original, unmodified fields.

## Data Cleaning
- **1,081 exact duplicate rows found and removed.** With 28 continuous PCA
  features, an exact match across all of them is effectively impossible by
  chance — these are almost certainly the same transaction captured twice
  during data collection. Left in place, they would let the same
  transaction leak between train and test splits.
- **Feature scaling:** `Time` and `Amount` were scaled with `RobustScaler`
  (less sensitive to `Amount`'s extreme outliers than a standard z-score
  scaler); `V1`-`V28` were left as-is since they're already PCA components.
- Post-cleaning: 283,726 rows, 473 frauds (0.167%).

## Methodology
1. **EDA** — class imbalance visualization, transaction amount by class
   (revealing a more nuanced pattern than "fraud = lower amount": fraud has
   a *lower median* but *higher mean and variance* than legitimate
   transactions, suggesting a mix of small "card testing" fraud and larger
   fraudulent charges).
2. **Train/validation/test split** (64/16/20, stratified) — used instead of
   heavy k-fold cross-validation, since Random Forest alone takes over a
   minute per fit at this data scale; candidates are compared once on
   validation, and the winner is retrained on train+validation before a
   single, final evaluation on the untouched test set.
3. **Three modeling approaches compared** by Average Precision (the metric
   this dataset's own documentation recommends over accuracy or plain
   ROC-AUC, given the ~600:1 imbalance):
   - Logistic Regression (`class_weight="balanced"`)
   - Random Forest (`class_weight="balanced_subsample"`)
   - Logistic Regression on **SMOTE-oversampled training data only** — SMOTE
     is fit strictly on the training fold, never on validation/test, to
     avoid leaking synthetic near-duplicates of test fraud cases into
     evaluation.
4. **Explainability with SHAP** on the winning model.

## Results
| Model | Validation Average Precision |
|---|---|
| Logistic Regression | 0.814 |
| **Random Forest** | **0.856** |
| Logistic Regression + SMOTE | 0.825 |

**Final model (Random Forest) on the held-out test set (95 real frauds, 56,651 real legitimate transactions):**
| Metric | Value |
|---|---|
| Average Precision | 0.751 |
| ROC-AUC | 0.966 |
| Precision (Fraud) | 0.886 |
| Recall (Fraud) | 0.737 |

**Business impact:** the model catches **70 of 95 frauds (73.7%)** in the
test period while flagging only **9 legitimate transactions out of 56,651**
for review (a 0.016% false-alarm rate) — a workable trade-off for a real
fraud review team.

**Top fraud indicators (SHAP):** `V14`, `V12`, `V4`, and `V10` — these match
the features most frequently cited as important in published analyses of
this exact dataset, a strong sanity check that the pipeline is correct.

## Tech Stack
- **Python**, **Pandas / NumPy** — data cleaning and processing
- **scikit-learn** — Logistic Regression, Random Forest, evaluation metrics
- **imbalanced-learn** — SMOTE (applied correctly, train-fold only)
- **SHAP** — model explainability
- **Matplotlib / Seaborn** — EDA and evaluation visualizations

## Files
- `Credit_Card_Fraud_Detection.ipynb` — full notebook, runs top to bottom
- `creditcard.csv` — the real dataset (284,807 rows)
- `eda_overview.png` — class distribution and amount-by-class
- `transactions_over_time.png` — fraud vs. legitimate over the 2-day window
- `model_comparison.png` — the three candidate approaches on validation
- `model_evaluation.png` — confusion matrix, ROC curve, precision-recall curve
- `shap_summary.png` — global feature importance

## Limitations
`V1`-`V28` are anonymized PCA components, so SHAP explains predictions in
terms of `V14`, `V4`, etc. rather than human-readable business concepts — a
real deployment would need the issuing bank's original, non-anonymized
feature set to turn this into plain-language fraud reasons for an analyst.

## Possible Next Steps
- Try XGBoost/LightGBM with `scale_pos_weight` for potentially stronger
  performance at similar training cost.
- Add a cost-sensitive threshold tuned to the bank's actual cost of a missed
  fraud vs. a false alarm, rather than the default 0.5 probability cutoff.
- Build a real-time scoring API and a simple review dashboard for flagged
  transactions.
