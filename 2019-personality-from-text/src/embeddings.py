"""Dense sentence embeddings with on-disk caching.

Pretrained embedders are frozen (not trained on this data), so encoding every
row up front introduces no leakage. We cache by (model, column, row-hash) so
repeated experiments are cheap.
"""
from __future__ import annotations

import hashlib
import os

import numpy as np

from .data import PROJECT_ROOT

CACHE_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "emb_cache")

# E5-family models need "query: " / "passage: " prefixes; we use a single
# prefix for symmetric encoding of documents.
PREFIX = {
    "intfloat/e5-base-v2": "query: ",
    "intfloat/e5-large-v2": "query: ",
    "intfloat/multilingual-e5-large": "query: ",
}

_MODEL_CACHE: dict = {}


def _key(model_name: str, texts) -> str:
    h = hashlib.sha1()
    h.update(model_name.encode("utf-8"))
    for t in texts:
        h.update(b"\x00")
        h.update(t.encode("utf-8", "replace"))
    return h.hexdigest()[:16]


def _get_model(model_name: str):
    if model_name not in _MODEL_CACHE:
        from sentence_transformers import SentenceTransformer
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _MODEL_CACHE[model_name] = SentenceTransformer(model_name, device=device)
    return _MODEL_CACHE[model_name]


def encode(texts, model_name: str, batch_size: int = 64, normalize: bool = True) -> np.ndarray:
    """Encode a list of texts; cache the resulting matrix to disk."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    safe = model_name.replace("/", "__")
    key = _key(model_name, texts)
    path = os.path.join(CACHE_DIR, f"{safe}__{key}.npy")
    if os.path.exists(path):
        return np.load(path)

    prefix = PREFIX.get(model_name, "")
    prepared = [prefix + t for t in texts] if prefix else list(texts)
    model = _get_model(model_name)
    emb = model.encode(
        prepared,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=normalize,
        convert_to_numpy=True,
    )
    np.save(path, emb)
    return emb
