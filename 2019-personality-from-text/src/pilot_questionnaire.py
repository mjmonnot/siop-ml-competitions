"""Pilot: validate the role-play + questionnaire extractor on a small sample.

Diagnostic only. Scores the frozen (no-label) extractor on N Train rows and
reports the direct reverse-scored aggregate's correlation with the labels, plus
a Ridge learned-head OOF estimate. Labels are used ONLY to measure signal, never
to fit the extractor. No Test access.
"""
from __future__ import annotations

import argparse

import numpy as np

from .data import TARGETS, TRAITS, load_data, split_data
from .llm_extract import (aggregate_questionnaire, score_questionnaire,
                          score_questionnaire_summary)
from .metrics import per_trait_r


def main(n: int = 60, summary: bool = False):
    df = load_data()
    train, _dev, _test = split_data(df)
    sub = train.iloc[:n].copy()
    Y = sub[TARGETS].to_numpy(dtype=float)

    if summary:
        print(f"Scoring TWO-STAGE (persona summary + questionnaire) on {n} Train rows...")
        items = score_questionnaire_summary(sub)
    else:
        print(f"Scoring questionnaire on {n} Train rows...")
        items = score_questionnaire(sub)
    agg = aggregate_questionnaire(items)

    print("\nDirect reverse-scored aggregate vs labels (diagnostic):")
    r = per_trait_r(Y, agg)
    for t in TRAITS:
        print(f"  {t}: {r[t]:+.3f}")
    print(f"  MEAN: {np.mean([r[t] for t in TRAITS]):+.3f}")
    print("\nNon-degenerate check (item response std per item):")
    print(np.round(items.std(axis=0), 2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=60)
    p.add_argument("--summary", action="store_true")
    args = p.parse_args()
    main(n=args.n, summary=args.summary)
