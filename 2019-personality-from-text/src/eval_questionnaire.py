"""Dev-only evaluation of the role-play + questionnaire extractor.

Compares the current best (Stage 4) stack against variants that add:
  - llmq     : direct reverse-scored questionnaire aggregate (no fit)
  - llmqfeat : questionnaire item responses -> learned Ridge head

Train-CV (OOF) + Dev only. Test is NEVER touched here.
"""
from __future__ import annotations

import numpy as np

from .data import TARGETS, TRAITS, load_data, split_data
from .experiment import build_bases
from .metrics import mean_r, report
from .stack import run_stack

E5 = "intfloat/e5-large-v2"
HAIKU = "claude-haiku-4-5-20251001"
SONNET = "claude-sonnet-4-6"

STAGE4 = [
    ("llm:" + HAIKU, {"variant": "general"}),
    ("llm:" + HAIKU, {"variant": "evidence"}),
    ("llm:" + HAIKU, {"variant": "ranked"}),
    ("llm:" + HAIKU, {"variant": "trait_focus"}),
    ("llm:" + SONNET, {"variant": "general"}),
    ("llmfeat:" + HAIKU, {"alpha": 5.0}),
    ("embpsvr:" + E5, {"C": 4.0, "gamma": "scale"}),
    ("embsvr:" + E5, {"C": 4.0, "gamma": "scale"}),
    ("tfidf_char_all", {"alpha": 8.0}),
    ("tfidf_word1_all", {"alpha": 8.0}),
    ("engineered", {"alpha": 5.0}),
    ("engineered_gbm", {}),
]

QDIRECT = [("llmq:" + HAIKU, {})]
QFEAT = [("llmqfeat:" + HAIKU, {"alpha": 5.0})]
QSDIRECT = [("llmqs:" + HAIKU, {})]
QSFEAT = [("llmqsfeat:" + HAIKU, {"alpha": 5.0})]

CANDIDATES = {
    "stage4 (current best)": STAGE4,
    "stage4 + q-direct": STAGE4 + QDIRECT,
    "stage4 + q-feat": STAGE4 + QFEAT,
    "stage4 + q-direct + q-feat": STAGE4 + QDIRECT + QFEAT,
    "stage4 + qsummary-direct": STAGE4 + QSDIRECT,
    "stage4 + qsummary-feat": STAGE4 + QSFEAT,
    "stage4 + q + qsummary (all)": STAGE4 + QDIRECT + QFEAT + QSDIRECT + QSFEAT,
    "questionnaire-only stack": QDIRECT + QFEAT + [
        ("embpsvr:" + E5, {"C": 4.0, "gamma": "scale"}),
    ],
}

META_ALPHA = 4.0


def run_one(label, specs):
    df = load_data()
    train, dev, _test = split_data(df)
    Y = train[TARGETS].to_numpy(dtype=float)
    Ydev = dev[TARGETS].to_numpy(dtype=float)
    bases = build_bases(specs)
    res = run_stack(bases, train, {"dev": dev}, Y, n_splits=5, seed=42,
                    meta_alpha=META_ALPHA, cross_trait=False)
    oof = mean_r(Y, res["oof"])
    devm = mean_r(Ydev, res["eval"]["dev"])
    print(f"\n### {label}")
    print(f"    Train-CV mean_r = {oof:.4f}   Dev mean_r = {devm:.4f}")
    return oof, devm, res


def main():
    results = {}
    for label, specs in CANDIDATES.items():
        oof, devm, res = run_one(label, specs)
        results[label] = (oof, devm)
    print("\n================ SUMMARY (Dev-only) ================")
    print(f"{'config':<32} {'Train-CV':>10} {'Dev':>10}")
    for label, (oof, devm) in results.items():
        print(f"{label:<32} {oof:>10.4f} {devm:>10.4f}")


if __name__ == "__main__":
    main()
