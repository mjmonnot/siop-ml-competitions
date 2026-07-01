"""Frozen evaluation entry point.

Default behavior evaluates Dev and Test. Pass --dev-only to validate the frozen
configuration on Train-CV + Dev without touching Test.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from .data import PROJECT_ROOT, SEED, TARGETS, TRAITS, load_data, split_data
from .experiment import build_bases
from .metrics import report
from .stack import run_stack

E5 = "intfloat/e5-large-v2"
HAIKU = "claude-haiku-4-5-20251001"
SONNET = "claude-sonnet-4-6"

FROZEN_SPECS = [
    ("llm:" + HAIKU, {"variant": "general"}),
    ("llm:" + HAIKU, {"variant": "evidence"}),
    ("llm:" + HAIKU, {"variant": "ranked"}),
    ("llm:" + HAIKU, {"variant": "trait_focus"}),
    ("llm:" + SONNET, {"variant": "general"}),
    ("llmfeat:" + HAIKU, {"alpha": 5.0}),
    ("llmq:" + HAIKU, {}),
    ("embpsvr:" + E5, {"C": 4.0, "gamma": "scale"}),
    ("embsvr:" + E5, {"C": 4.0, "gamma": "scale"}),
    ("tfidf_char_all", {"alpha": 8.0}),
    ("tfidf_word1_all", {"alpha": 8.0}),
    ("engineered", {"alpha": 5.0}),
    ("engineered_gbm", {}),
]
FROZEN_META_ALPHA = 4.0
FROZEN_CROSS_TRAIT = False
FROZEN_SEED = SEED
FROZEN_N_SPLITS = 5


def main(dev_only: bool = False):
    df = load_data()
    train, dev, test = split_data(df)
    Y = train[TARGETS].to_numpy(dtype=float)
    Ydev = dev[TARGETS].to_numpy(dtype=float)
    eval_frames = {"dev": dev} if dev_only else {"dev": dev, "test": test}

    bases = build_bases(FROZEN_SPECS)
    res = run_stack(
        bases, train, eval_frames, Y,
        n_splits=FROZEN_N_SPLITS, seed=FROZEN_SEED,
        meta_alpha=FROZEN_META_ALPHA, cross_trait=FROZEN_CROSS_TRAIT,
    )

    title = "DEV-ONLY VALIDATION" if dev_only else "TEST EVALUATION"
    print(f"\n================ STAGE 5 FROZEN MODEL (+q-direct) -- {title} ================")
    print("config:", json.dumps(FROZEN_SPECS), "meta_alpha=", FROZEN_META_ALPHA)
    oof = report("Train-CV (OOF)", Y, res["oof"])
    devr = report("Dev (public)", Ydev, res["eval"]["dev"])

    if not dev_only:
        Ytest = test[TARGETS].to_numpy(dtype=float)
        testr = report("Test (PRIVATE)", Ytest, res["eval"]["test"])
        target = 0.26021
        print(f"\n2019 first-place Test mean r to beat: {target}")
        print(f"Our Test mean r: {testr['MEAN']:.5f}  -> "
              f"{'BEATS' if testr['MEAN'] > target else 'does NOT beat'} the leaderboard.")

    sub_dir = os.path.join(PROJECT_ROOT, "results", "submissions")
    cv_dir = os.path.join(PROJECT_ROOT, "results", "cv")
    os.makedirs(sub_dir, exist_ok=True)
    os.makedirs(cv_dir, exist_ok=True)

    def write_sub(frame, pred, tag):
        out = pd.DataFrame({"Respondent_ID": frame["Respondent_ID"].values})
        for i, t in enumerate(TRAITS):
            out[f"{t}_Pred"] = pred[:, i]
        path = os.path.join(sub_dir, f"submission_{tag}_frozen.csv")
        out.to_csv(path, index=False)
        print("wrote", path)

    write_sub(dev, res["eval"]["dev"], "dev")
    if not dev_only:
        write_sub(test, res["eval"]["test"], "test")

    summary = {"Train_CV": oof, "Dev": devr}
    if not dev_only:
        summary["Test"] = testr
    summary = pd.DataFrame(summary)
    name = "stage4_dev_summary.csv" if dev_only else "frozen_summary.csv"
    path = os.path.join(cv_dir, name)
    summary.to_csv(path)
    print("wrote", path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-only", action="store_true")
    args = parser.parse_args()
    main(dev_only=args.dev_only)
