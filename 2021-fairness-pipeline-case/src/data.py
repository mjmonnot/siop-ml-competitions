"""Data loading and freeze-protocol splits for the 2021 post-hoc case."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

LABEL_COLS = ["High_Performer", "Overall_Rating", "Retained", "Protected_Group"]
REQUIRED_SCORE_COLS = ["High_Performer", "Retained", "Protected_Group"]

CAT_COLUMNS = [
    "SJ_Most_1", "SJ_Least_1", "SJ_Most_2", "SJ_Least_2",
    "SJ_Most_3", "SJ_Least_3", "SJ_Most_4", "SJ_Least_4", "SJ_Most_5",
    "SJ_Least_5", "SJ_Most_6", "SJ_Least_6", "SJ_Most_7", "SJ_Least_7",
    "SJ_Most_8", "SJ_Least_8", "SJ_Most_9", "SJ_Least_9",
    "Scenario1_1", "Scenario1_2", "Scenario1_3", "Scenario1_4", "Scenario1_5",
    "Scenario1_6", "Scenario1_7", "Scenario1_8",
    "Scenario2_1", "Scenario2_2", "Scenario2_3", "Scenario2_4", "Scenario2_5",
    "Scenario2_6", "Scenario2_7", "Scenario2_8",
    "Biodata_01", "Biodata_02", "Biodata_03", "Biodata_04", "Biodata_05",
    "Biodata_06", "Biodata_07", "Biodata_08", "Biodata_09", "Biodata_10",
    "Biodata_11", "Biodata_12", "Biodata_13", "Biodata_14", "Biodata_15",
    "Biodata_16", "Biodata_17", "Biodata_18", "Biodata_19", "Biodata_20",
]

# Published private-test leaderboard (organizer deck / winners README).
PUBLISHED_LEADERBOARD = [
    {"place": 1, "team": "Team Procrustination", "final_score": 62.53},
    {"place": 2, "team": "Axiom Consulting Partners", "final_score": 62.50},
    {"place": 3, "team": "RHDS", "final_score": 61.09},
    {"place": 4, "team": "Go Ahead, Make My Data", "final_score": 60.72},
]


def year_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_train_path() -> Path:
    return year_root() / "00_data" / "train.csv"


def load_train(path: str | Path | None = None) -> pd.DataFrame:
    path = Path(path) if path else default_train_path()
    df = pd.read_csv(path, na_values=[" ", "."])
    if "split" in df.columns:
        df = df.drop(columns=["split"])
    return df


def labeled_subset(train: pd.DataFrame) -> pd.DataFrame:
    """Rows with the labels required by the competition scoring function."""
    out = train.dropna(subset=REQUIRED_SCORE_COLS).copy()
    out["High_Performer"] = out["High_Performer"].astype(int)
    out["Retained"] = out["Retained"].astype(int)
    out["Protected_Group"] = out["Protected_Group"].astype(int)
    return out.reset_index(drop=True)


def freeze_split(
    labeled: pd.DataFrame,
    *,
    test_size: float = 0.20,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Single frozen holdout for honest post-hoc measurement.

    Stratify on Protected_Group × High_Performer so AIR and accuracy
    components stay roughly balanced across fit/holdout.
    """
    strata = (
        labeled["Protected_Group"].astype(str)
        + "_"
        + labeled["High_Performer"].astype(str)
        + "_"
        + labeled["Retained"].astype(str)
    )
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    fit_idx, hold_idx = next(splitter.split(np.zeros(len(labeled)), strata))
    fit = labeled.iloc[fit_idx].reset_index(drop=True)
    hold = labeled.iloc[hold_idx].reset_index(drop=True)
    return fit, hold


def feature_matrix(
    train_like: pd.DataFrame,
    test_like: pd.DataFrame,
    *,
    drop_protected: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One-hot encode categorical assessment items consistently across splits."""
    exclude = set(LABEL_COLS + ["UNIQUE_ID"])
    if drop_protected:
        exclude.add("Protected_Group")

    feature_cols = [c for c in train_like.columns if c not in exclude]
    # Keep only columns that exist in both (participant files omit labels).
    feature_cols = [c for c in feature_cols if c in test_like.columns]

    all_data = pd.concat(
        [train_like[feature_cols], test_like[feature_cols]],
        axis=0,
        ignore_index=True,
    )
    cat_present = [c for c in CAT_COLUMNS if c in all_data.columns]
    all_data = pd.get_dummies(all_data, prefix_sep="__", columns=cat_present)

    n = len(train_like)
    X_train = all_data.iloc[:n].copy()
    X_test = all_data.iloc[n:].copy()

    for frame in (X_train, X_test):
        for col in frame.columns:
            if frame[col].dtype.kind in "biufc":
                frame[col] = frame[col].fillna(frame[col].median())
            else:
                frame[col] = frame[col].fillna(0)
    return X_train, X_test
