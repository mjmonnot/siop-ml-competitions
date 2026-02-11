"""PROCRUSTINATION-STYLE REFERENCE SOLUTION (FULLY ANNOTATED)

This script is intentionally written in plain English.

What it does (high level):
1) Loads training data and drops rows with missing values (a contest simplification).
2) One-hot encodes many categorical assessment items so ML can use them.
3) Trains several XGBoost models to predict different outcomes:
   - High performer (classification, two variants)
   - Overall rating (regression)
   - A special proxy target: (Protected_Group == 1 AND Retained == 1)
4) Standardizes predictions (z-scores) so they can be combined safely.
5) Builds a weighted ensemble score and hires the top half (median cutoff).
6) Writes a submission file: UNIQUE_ID, Hire

Why this matters:
- It’s a strong example of "metric-aware engineering" for a competition.
- It’s also a good contrast case for a standards-aligned workflow.

NOTE:
This script uses a proxy that depends on Protected_Group in training.
That is included for teaching contrast, not as a recommended applied practice.
"""
import argparse
import numpy as np
import pandas as pd
from scipy import stats
from xgboost import XGBClassifier
import xgboost as xgb

# These are the categorical columns the winning-style solution one-hot encodes.
# One-hot encoding turns category labels into 0/1 indicator columns.
CAT_COLUMNS = [
    'SJ_Most_1', 'SJ_Least_1', 'SJ_Most_2', 'SJ_Least_2',
    'SJ_Most_3', 'SJ_Least_3', 'SJ_Most_4', 'SJ_Least_4',  'SJ_Most_5',
    'SJ_Least_5',  'SJ_Most_6', 'SJ_Least_6',  'SJ_Most_7', 'SJ_Least_7',
    'SJ_Most_8', 'SJ_Least_8', 'SJ_Most_9', 'SJ_Least_9',
    'Scenario1_1', 'Scenario1_2', 'Scenario1_3', 'Scenario1_4', 'Scenario1_5',
    'Scenario1_6', 'Scenario1_7', 'Scenario1_8',
    'Scenario2_1', 'Scenario2_2', 'Scenario2_3', 'Scenario2_4', 'Scenario2_5',
    'Scenario2_6', 'Scenario2_7', 'Scenario2_8',
    'Biodata_01', 'Biodata_02', 'Biodata_03', 'Biodata_04', 'Biodata_05',
    'Biodata_06', 'Biodata_07', 'Biodata_08', 'Biodata_09', 'Biodata_10',
    'Biodata_11', 'Biodata_12', 'Biodata_13', 'Biodata_14', 'Biodata_15',
    'Biodata_16', 'Biodata_17', 'Biodata_18', 'Biodata_19', 'Biodata_20'
]

def load_data(train_path: str, test_path: str):
    """Load train/test, create consistent one-hot features."""
    # Read training data and remove 'split' column if present.
    cols = list(pd.read_csv(train_path, nrows=1))
    usecols = [c for c in cols if c != 'split']
    data = pd.read_csv(train_path, usecols=usecols, na_values=[' ', '.'])

    # Contest simplification: keep only complete cases.
    df_train0 = data.dropna()

    # Contest convention: use columns after index 9 as predictors.
    df_train = df_train0.iloc[:, 9:].copy()

    # Read test data and exclude ID and split.
    cols1 = list(pd.read_csv(test_path, nrows=1))
    usecols_test = [c for c in cols1 if c not in ['UNIQUE_ID', 'split']]
    df_test = pd.read_csv(test_path, usecols=usecols_test, na_values=[' ', '.'])

    # Combine so one-hot encoding produces the same columns in train and test.
    all_data = pd.concat([df_train, df_test], axis=0)

    # Only one-hot encode categorical columns that actually exist in this dataset.
    cat_present = [c for c in CAT_COLUMNS if c in all_data.columns]
    all_data = pd.get_dummies(all_data, prefix_sep="__", columns=cat_present)

    X_train = all_data.iloc[:len(df_train), :].copy()
    X_test = all_data.iloc[len(df_train):, :].copy()

    # Keep test IDs for submission file.
    ID_test = pd.read_csv(test_path, usecols=['UNIQUE_ID'])
    return df_train0, X_train, X_test, ID_test

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="Path to train.csv")
    ap.add_argument("--test", required=True, help="Path to participant_test.csv")
    ap.add_argument("--out", default="final_submission.csv", help="Output submission CSV")
    args = ap.parse_args()

    df_train0, X_train, X_test, ID_test = load_data(args.train, args.test)

    # --- Model 1: High_Performer (simple) ---
    # scale_pos_weight helps when the positive class (1) is rarer than (0).
    y_train = df_train0['High_Performer']
    model1 = XGBClassifier(scale_pos_weight=1.5)
    model1.fit(X_train, y_train)
    y_pred1 = stats.zscore(model1.predict_proba(X_test)[:, 1])

    # --- Model 2: Overall_Rating (regression) ---
    y_train = df_train0['Overall_Rating']
    reg = xgb.XGBRegressor(
        subsample=0.6, n_estimators=200, learning_rate=0.05,
        colsample_bytree=0.6, max_depth=2, min_child_weight=4
    )
    reg.fit(X_train, y_train)
    y_pred2 = stats.zscore(reg.predict(X_test))

    # --- Model 3: High_Performer (tuned) ---
    # The team reports Bayesian optimization; parameters are copied as provided.
    y_train = df_train0['High_Performer']
    model3 = xgb.XGBClassifier(
        learning_rate=0.005411872947900535,
        n_estimators=1789,
        max_depth=3,
        min_child_weight=5.818676232053935,
        gamma=0.05591980172280099,
        subsample=0.5744551033482959,
        colsample_bytree=0.5226781217635239,
        seed=42
    )
    model3.fit(X_train, y_train)
    y_pred3 = stats.zscore(model3.predict_proba(X_test)[:, 1])

    # --- Model 4: Protected AND Retained proxy (contest fairness hack) ---
    # This creates a new label that depends on Protected_Group, which is why this is best viewed as a contrast case.
    def protected_retained(row):
        return 1 if (row['Protected_Group'] == 1 and row['Retained'] == 1) else 0

    y_train = df_train0.apply(protected_retained, axis=1)
    model4 = XGBClassifier(scale_pos_weight=3)
    model4.fit(X_train, y_train)
    y_pred4 = stats.zscore(model4.predict_proba(X_test)[:, 1])

    # --- Final ensemble ---
    # 90% weight goes to performance/ratings signals, 10% to the protected+retained correction.
    score = (y_pred1 * 0.5 + y_pred2 * 0.2 + y_pred3 * 0.3) * 0.9 + y_pred4 * 0.1

    # Hire top half (median cutoff)
    cutoff = np.median(score)
    hire = (score > cutoff).astype(int)

    sub = pd.DataFrame({"UNIQUE_ID": ID_test["UNIQUE_ID"], "Hire": hire})
    sub.to_csv(args.out, index=False)
    print(f"Wrote submission to: {args.out}")

if __name__ == "__main__":
    main()
