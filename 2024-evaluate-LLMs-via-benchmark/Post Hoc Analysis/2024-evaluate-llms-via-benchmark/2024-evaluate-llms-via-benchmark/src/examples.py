"""Few-shot example selection for the four 2024 SIOP tasks.

Two strategies are provided:

1. **Random.** Pick K training examples at random with a fixed seed.
   This is what PAID Team did (they used ALL training examples as
   few-shot, which is a limit case of random selection with K=N).

2. **Similarity.** Pick the K training examples whose input is most
   similar (cosine, on sentence embeddings) to the test row's input.
   This was NOT what the 2024 winners did — Hungry Llama considered it
   briefly and abandoned. With 2026 hindsight it's a clear winner on
   empathy and fairness, where the few-shot set is large enough that
   you can afford to be picky. See KNOWN_LANDMINES.md Landmine 4.

The similarity model is loaded lazily so notebooks that only use random
selection don't pay the import cost.

Embedding model choice: all-MiniLM-L6-v2 is ~80 MB, runs on CPU in well
under a second per sentence, and has been a default for this kind of
retrieval task since 2021. There's no reason to reach for a bigger model
for K-of-N reranking.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np


_EMBEDDER = None  # lazy-loaded sentence-transformer


def _get_embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        from sentence_transformers import SentenceTransformer
        _EMBEDDER = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _EMBEDDER


def pick_random(
    pool: list[dict],
    k: int,
    seed: int = 1234,
) -> list[dict]:
    """Pick k items from pool. If k >= len(pool), return all of them in
    the original order. Otherwise sample without replacement using `seed`.
    """
    if k >= len(pool):
        return list(pool)
    rng = random.Random(seed)
    return rng.sample(pool, k)


def pick_similar(
    pool: list[dict],
    target: str,
    k: int,
    text_field: str = "text",
    pool_embeddings: np.ndarray | None = None,
) -> tuple[list[dict], np.ndarray]:
    """Pick the k pool items whose `text_field` is most similar to
    `target`. Returns the items and the (possibly newly-computed) pool
    embeddings so the caller can reuse them across many targets.

    The embeddings are L2-normalized by sentence-transformers, so cosine
    similarity reduces to a dot product. This matters: a full test split
    across the empathy or fairness train set is a few thousand dot
    products in numpy, well under a second.
    """
    if k >= len(pool):
        if pool_embeddings is None:
            pool_embeddings = _embed([row[text_field] for row in pool])
        return list(pool), pool_embeddings

    embedder = _get_embedder()
    if pool_embeddings is None:
        pool_embeddings = _embed([row[text_field] for row in pool])
    target_emb = embedder.encode([target], normalize_embeddings=True)[0]

    sims = pool_embeddings @ target_emb  # both already normalized
    top_idx = np.argsort(-sims)[:k]
    # Return in similarity order (most similar first), which gives the
    # model a useful ordering signal in long few-shot prompts.
    chosen = [pool[i] for i in top_idx]
    return chosen, pool_embeddings


def _embed(texts: list[str]) -> np.ndarray:
    embedder = _get_embedder()
    return embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)


# --- selftest ---


def _selftest() -> int:
    pool = [{"id": i, "text": f"item {i}"} for i in range(10)]
    chosen = pick_random(pool, k=3, seed=42)
    assert len(chosen) == 3
    chosen2 = pick_random(pool, k=3, seed=42)
    assert chosen == chosen2, "random with same seed must be reproducible"
    all_back = pick_random(pool, k=99)
    assert len(all_back) == 10, "k >= n returns full pool"
    print("examples selftest OK")
    return 0


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        raise SystemExit(_selftest())
    print("Use --selftest to run sanity checks.")
