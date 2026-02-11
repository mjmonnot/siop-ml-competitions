"""JOB SUCCESS MODEL (FULLY ANNOTATED)

This script builds a prediction score intended to represent **job success**.

We define a "Job Success Index" (JSI) that mirrors the business weights in the competition metric:
JSI = 0.25*(Top Performer) + 0.25*(Retained) + 0.50*(Top AND Retained)

Key teaching point:
- We do NOT include Protected_Group as a predictor.
- Protected_Group is used later ONLY for fairness evaluation and governance.

Outputs:
- scored_test.csv with columns: UNIQUE_ID, score
"""
from __future__ import annotations
import argparse
import pandas as pd
from xgboost import XGBRegressor

# Optional: categorical fields you may want one-hot encoded (kept parallel to contest baseline)
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

def build_job_success_index(df: pd.DataFrame) -> pd.Series:
    top = df["High_Performer"].astype(int)
    ret = df["Retained"].astype(int)
    both = ((top == 1) & (ret == 1)).astype(int)
    return 0.25 * top + 0.25 * ret + 0.50 * both

def preprocess(train: pd.DataFrame, test: pd.DataFrame):
    """One-hot encode categorical columns consistently across train/test."""
    # Remove split if present
    for d in (train, test):
        if "split" in d.columns:
            d.drop(columns=["split"], inplace=True)

    # Identify label + protected columns to exclude from predictors
    label_cols = [c for c in ["High_Performer", "Overall_Rating", "Retained", "Protected_Group"] if c in train.columns]
    feature_cols = [c for c in train.columns if c not in (label_cols + ["UNIQUE_ID"])]

    all_data = pd.concat([train[feature_cols], test[feature_cols]], axis=0)

    cat_present = [c for c in CAT_COLUMNS if c in all_data.columns]
    all_data = pd.get_dummies(all_data, prefix_sep="__", columns=cat_present)

    X_train = all_data.iloc[:len(train), :].copy()
    X_test = all_data.iloc[len(train):, :].copy()

    # Explicit missing handling for teaching clarity
    X_train = X_train.apply(lambda col: col.fillna(col.median()) if col.dtype.kind in "biufc" else col.fillna(0))
    X_test = X_test.apply(lambda col: col.fillna(col.median()) if col.dtype.kind in "biufc" else col.fillna(0))

    return X_train, X_test

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", default="scored_test.csv")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    train = pd.read_csv(args.train, na_values=[' ', '.'])
    test = pd.read_csv(args.test, na_values=[' ', '.'])

    test_ids = test[["UNIQUE_ID"]].copy()

    y = build_job_success_index(train)
    X_train, X_test = preprocess(train, test)

    # A simple gradient-boosted regressor to predict the JSI score.
    model = XGBRegressor(
        n_estimators=800,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=args.seed
    )
    model.fit(X_train, y.to_numpy())

    score = model.predict(X_test)

    out = test_ids.copy()
    out["score"] = score
    out.to_csv(args.out, index=False)
    print(f"Wrote scored test to: {args.out}")

if __name__ == "__main__":
    main()
