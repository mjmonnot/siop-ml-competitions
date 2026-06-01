"""Per-task metrics for the 2024 SIOP ML Competition.

From the official competition deck:
- Empathy:   accuracy on binary classification (0/1)
- Interview: average cosine similarity between generated and reference text
- Clarity:   Pearson correlation between predicted and true mean ratings
- Fairness:  accuracy on binary classification ("first"/"second")

Final composite score = 0.25 * each. The final scores published by the
organizers are reported on this composite scale (so PAID's .666 means
their four task metrics averaged to .666 / .25 = 2.664, distributed
across tasks per the breakdown in WINNERS_SYNTHESIS.md).

The interview metric needs an embedding model. The 2024 competition used
SentenceTransformer all-MiniLM-L6-v2 (per the competition portal docs).
That's what we replicate here, so the score we compute is directly
comparable to the leaderboard.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def accuracy(predictions: Sequence[str], truths: Sequence[str]) -> float:
    """Plain accuracy on string-equal comparisons.

    Used for empathy (0/1 strings) and fairness ('first'/'second').
    Both predictions and truths must be strings; the adapters are
    responsible for normalizing their outputs before scoring.
    """
    if len(predictions) != len(truths):
        raise ValueError(
            f"length mismatch: {len(predictions)} predictions vs {len(truths)} truths"
        )
    if not predictions:
        return 0.0
    n_correct = sum(1 for p, t in zip(predictions, truths) if str(p).strip() == str(t).strip())
    return n_correct / len(predictions)


def pearson_r(predictions: Sequence[float], truths: Sequence[float]) -> float:
    """Pearson correlation for the clarity task.

    NaN safety: if either array is constant, np.corrcoef returns NaN.
    We return 0.0 instead so the composite score doesn't blow up. This
    is consistent with how the leaderboard handled degenerate
    submissions (per the organizers' note in the competition portal).
    """
    if len(predictions) != len(truths):
        raise ValueError(
            f"length mismatch: {len(predictions)} predictions vs {len(truths)} truths"
        )
    if len(predictions) < 2:
        return 0.0
    p = np.asarray(predictions, dtype=float)
    t = np.asarray(truths, dtype=float)
    if np.std(p) == 0 or np.std(t) == 0:
        return 0.0
    r = float(np.corrcoef(p, t)[0, 1])
    return 0.0 if np.isnan(r) else r


def avg_cosine_similarity(
    predictions: Sequence[str],
    truths: Sequence[str],
    embedder=None,
) -> float:
    """Average cosine similarity between generated and reference text.

    Used for the interview task. The embedder argument lets callers
    inject the model; if not provided, we lazy-load all-MiniLM-L6-v2,
    which is what the competition used.
    """
    if len(predictions) != len(truths):
        raise ValueError(
            f"length mismatch: {len(predictions)} predictions vs {len(truths)} truths"
        )
    if not predictions:
        return 0.0
    if embedder is None:
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    pred_emb = embedder.encode(list(predictions), normalize_embeddings=True, show_progress_bar=False)
    truth_emb = embedder.encode(list(truths), normalize_embeddings=True, show_progress_bar=False)

    # Both arrays are L2-normalized, so cosine = dot product, row-wise.
    sims = (pred_emb * truth_emb).sum(axis=1)
    return float(np.mean(sims))


def composite(
    empathy_acc: float,
    interview_cos: float,
    clarity_r: float,
    fairness_acc: float,
) -> float:
    """0.25-weighted sum, matching the official scoring formula.

    Returns the composite final score on the same scale as the
    published leaderboard. PAID was .666; the upper bound is 1.0.
    """
    return 0.25 * (empathy_acc + interview_cos + clarity_r + fairness_acc)


def _selftest() -> int:
    # accuracy
    assert accuracy(["1", "0", "1"], ["1", "0", "0"]) == 2 / 3
    assert accuracy([], []) == 0.0
    assert accuracy(["first", "second"], ["first", "second"]) == 1.0

    # pearson with hand-checkable inputs
    r = pearson_r([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0])
    assert abs(r - 1.0) < 1e-9, f"expected r=1, got {r}"
    r = pearson_r([1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0])
    assert abs(r - (-1.0)) < 1e-9, f"expected r=-1, got {r}"
    # degenerate
    assert pearson_r([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) == 0.0

    # composite
    c = composite(0.5, 0.5, 0.5, 0.5)
    assert abs(c - 0.5) < 1e-9
    c = composite(0.58, 0.44, 0.816, 0.828)  # reconstructed PAID Team task scores
    # 0.25 * (0.58 + 0.44 + 0.816 + 0.828) = 0.25 * 2.664 = 0.666
    assert abs(c - 0.666) < 1e-3, f"PAID reconstruction off: {c}"

    print("scoring selftest OK")
    return 0


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        raise SystemExit(_selftest())
    print("Use --selftest to run sanity checks (no embedding model needed for the binary checks).")
