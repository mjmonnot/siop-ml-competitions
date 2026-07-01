"""Data loading and shared constants for the 2019 SIOP ML personality task.

Honest-evaluation protocol (see handoff brief Section 5):
- Fit only on Train.
- Select on Dev (or nested CV on Train).
- Touch Test exactly once with the frozen model.

All preprocessing that learns parameters (vectorizers, scalers, embedders'
downstream heads) must be fit on Train (or fold-internal) only.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
DEFAULT_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "2019_siop_ml_comp_data.csv")

TEXT_COLS = [f"open_ended_{i}" for i in range(1, 6)]

# Verified against organizers' full_data/README.md:
#   open_ended_1 -> Agreeableness
#   open_ended_2 -> Conscientiousness
#   open_ended_3 -> Extraversion
#   open_ended_4 -> Neuroticism
#   open_ended_5 -> Openness
TRAITS = ["A", "C", "E", "N", "O"]
TARGETS = ["A_Scale_score", "C_Scale_score", "E_Scale_score", "N_Scale_score", "O_Scale_score"]
TRAIT_TO_TARGET = {t: f"{t}_Scale_score" for t in TRAITS}
# The single open-ended prompt designed to elicit each trait.
TRAIT_TO_PROMPT = {
    "A": "open_ended_1",
    "C": "open_ended_2",
    "E": "open_ended_3",
    "N": "open_ended_4",
    "O": "open_ended_5",
}

SEED = 42


def load_data(data_path: str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load the full labeled dataset and add a concatenated all-text column."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Could not find data file at {data_path}. "
            "Download it from izk8/2019_SIOP_Machine_Learning_Winners/full_data/."
        )
    df = pd.read_csv(data_path)
    for c in TEXT_COLS:
        df[c] = df[c].fillna("").astype(str)
    df["all_text"] = df[TEXT_COLS].agg(" ".join, axis=1)
    return df


def split_data(df: pd.DataFrame):
    """Return (train, dev, test) DataFrames, index reset."""
    train = df[df["Dataset"] == "Train"].copy().reset_index(drop=True)
    dev = df[df["Dataset"] == "Dev"].copy().reset_index(drop=True)
    test = df[df["Dataset"] == "Test"].copy().reset_index(drop=True)
    return train, dev, test


def get_targets(frame: pd.DataFrame) -> np.ndarray:
    """Return an (n, 5) array of trait targets in TRAITS order."""
    return frame[TARGETS].to_numpy(dtype=float)
