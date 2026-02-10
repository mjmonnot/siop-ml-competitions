from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBRegressor

# Categorical columns (same as contest baseline where relevant)
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
    """Job Success Index aligned to the business weights in the competition metric.

    JSI = 0.25*Top + 0.25*Retained + 0.50*(Top & Retained)
    """
    top = df["High_Performer"].astype(int)
    ret = df["Retained"].astype(int)
    both = ((top == 1) & (ret == 1)).astype(int)
    return 0.25 * top + 0.25 * ret + 0.50 * both

def preprocess(train_path: str, test_path: str):
    train = pd.read_csv(train_path, na_values=[' ', '.'])
    test = pd.read_csv(test_path, na_values=[' ', '.'])

    # Keep ID separately
    test_ids = test[["UNIQUE_ID"]].copy()

    # Drop split if present
    for df in (train, test):
        if "split" in df.columns:
            df.drop(columns=["split"], inplace=True)

    # Define features: by default, drop label columns if present
    label_cols = [c for c in ["High_Performer", "Overall_Rating", "Retained", "Protected_Group"] if c in train.columns]
    feature_cols = [c for c in train.columns if c not in (label_cols + ["UNIQUE_ID"])]

    # Combine for consistent dummies
    all_data = pd.concat([train[feature_cols], test[feature_cols]], axis=0)

    cat_present = [c for c in CAT_COLUMNS if c in all_data.columns]
    all_data = pd.get_dummies(all_data, prefix_sep="__", columns=cat_present)

    X_train = all_data.iloc[:len(train), :].copy()
    X_test = all_data.iloc[len(train):, :].copy()

    # Simple missing handling: median impute numeric, mode for dummies already 0/1
    # (XGBoost can handle NaNs, but we keep this explicit for teaching clarity.)
    X_train = X_train.apply(lambda col: col.fillna(col.median()) if col.dtype.kind in "biufc" else col.fillna(0))
    X_test = X_test.apply(lambda col: col.fillna(col.median()) if col.dtype.kind in "biufc" else col.fillna(0))

    return train, test, test_ids, X_train, X_test

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="Path to train.csv")
    ap.add_argument("--test", required=True, help="Path to participant_test.csv")
    ap.add_argument("--out", default="scored_test.csv", help="Output scored test CSV (includes UNIQUE_ID + score)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    train, test, test_ids, X_train, X_test = preprocess(args.train, args.test)

    # Build criterion index (does NOT use Protected_Group in scoring)
    y = build_job_success_index(train).to_numpy()

    # Fit a simple regressor to predict Job Success Index
    model = XGBRegressor(
        n_estimators=800,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=args.seed
    )
    model.fit(X_train, y)

    score = model.predict(X_test)
    out = test_ids.copy()
    out["score"] = score
    out.to_csv(args.out, index=False)
    print(f"Wrote scored test to: {args.out}")

if __name__ == "__main__":
    main()
