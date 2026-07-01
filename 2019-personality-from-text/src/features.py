"""Feature extractors.

Three families of representation:
1. Sparse TF-IDF (word + char n-grams) -- must be fit fold-internally because
   the vocabulary/idf are learned. Exposed as sklearn Pipeline factories.
2. Engineered psycholinguistic / stylometric features -- deterministic per row,
   so they can be precomputed once for all splits without leakage.
3. Dense sentence embeddings (see embeddings.py) -- frozen pretrained models,
   also leakage-free to precompute.
"""
from __future__ import annotations

import re
import string

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# ---------------------------------------------------------------------------
# TF-IDF vectorizer factories (return fresh, unfitted vectorizers)
# ---------------------------------------------------------------------------

def word_vectorizer(ngram_range=(1, 2), min_df=2) -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="word",
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=0.9,
        sublinear_tf=True,
        strip_accents="unicode",
        token_pattern=r"(?u)\b\w+\b",
    )


def word1_vectorizer() -> TfidfVectorizer:
    # Unigram words: better Dev than (1,2) on this short text (bigrams overfit).
    return word_vectorizer(ngram_range=(1, 1), min_df=2)


def char_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 6),
        min_df=2,
        sublinear_tf=True,
    )


# ---------------------------------------------------------------------------
# Engineered features (deterministic per row)
# ---------------------------------------------------------------------------
_FIRST_PERSON = {"i", "me", "my", "mine", "myself", "we", "us", "our", "ours"}
_NEGATIONS = {"no", "not", "never", "none", "nobody", "nothing", "neither", "nor",
              "cannot", "can't", "won't", "don't", "doesn't", "didn't", "isn't",
              "aren't", "wasn't", "weren't", "shouldn't", "wouldn't", "couldn't"}
_HEDGES = {"maybe", "perhaps", "possibly", "probably", "might", "could", "would",
           "guess", "suppose", "seem", "seems", "likely", "sometimes", "somewhat"}
_CERTAINTY = {"always", "definitely", "certainly", "absolutely", "never", "sure",
              "clearly", "obviously", "must", "completely", "totally"}
_SOCIAL = {"friend", "friends", "colleague", "colleagues", "team", "people",
           "coworker", "coworkers", "client", "clients", "boss", "manager",
           "everyone", "others", "together", "group"}

_word_re = re.compile(r"[a-zA-Z']+")


def _basic_stats(text: str) -> dict:
    text = text or ""
    words = _word_re.findall(text.lower())
    n_words = len(words)
    n_chars = len(text)
    uniq = len(set(words))
    sentences = re.split(r"[.!?]+", text)
    sentences = [s for s in sentences if s.strip()]
    n_sent = max(len(sentences), 1)
    word_lens = [len(w) for w in words] or [0]
    counts = {
        "n_chars": n_chars,
        "n_words": n_words,
        "n_sent": n_sent,
        "words_per_sent": n_words / n_sent,
        "avg_word_len": float(np.mean(word_lens)),
        "ttr": uniq / n_words if n_words else 0.0,
        "exclaim": text.count("!"),
        "question": text.count("?"),
        "comma": text.count(","),
        "ellipsis": text.count("..."),
        "upper_ratio": (sum(1 for c in text if c.isupper()) / n_chars) if n_chars else 0.0,
        "punct_ratio": (sum(1 for c in text if c in string.punctuation) / n_chars) if n_chars else 0.0,
    }
    if n_words:
        wl = words
        counts["first_person"] = sum(1 for w in wl if w in _FIRST_PERSON) / n_words
        counts["negation"] = sum(1 for w in wl if w in _NEGATIONS) / n_words
        counts["hedge"] = sum(1 for w in wl if w in _HEDGES) / n_words
        counts["certainty"] = sum(1 for w in wl if w in _CERTAINTY) / n_words
        counts["social"] = sum(1 for w in wl if w in _SOCIAL) / n_words
    else:
        counts.update({k: 0.0 for k in ["first_person", "negation", "hedge", "certainty", "social"]})
    return counts


def _readability(text: str) -> dict:
    import textstat
    text = text if text.strip() else "x."
    try:
        return {
            "flesch": textstat.flesch_reading_ease(text),
            "fkgrade": textstat.flesch_kincaid_grade(text),
            "gunning": textstat.gunning_fog(text),
            "dchall": textstat.dale_chall_readability_score(text),
            "diff_words": textstat.difficult_words(text),
        }
    except Exception:
        return {"flesch": 0.0, "fkgrade": 0.0, "gunning": 0.0, "dchall": 0.0, "diff_words": 0.0}


def _sentiment(text: str) -> dict:
    from textblob import TextBlob
    try:
        b = TextBlob(text)
        return {"polarity": b.sentiment.polarity, "subjectivity": b.sentiment.subjectivity}
    except Exception:
        return {"polarity": 0.0, "subjectivity": 0.0}


_POS_GROUPS = {
    "noun": ("NN", "NNS", "NNP", "NNPS"),
    "propn": ("NNP", "NNPS"),
    "verb": ("VB", "VBD", "VBG", "VBN", "VBP", "VBZ"),
    "adj": ("JJ", "JJR", "JJS"),
    "adv": ("RB", "RBR", "RBS"),
    "pron": ("PRP", "PRP$", "WP", "WP$"),
    "modal": ("MD",),
    "conj": ("CC",),
    "prep": ("IN",),
    "det": ("DT",),
}


def _pos_features(text: str) -> dict:
    from nltk import pos_tag, word_tokenize
    keys = {f"pos_{k}": 0.0 for k in _POS_GROUPS}
    if not text.strip():
        return keys
    try:
        tags = [t for _, t in pos_tag(word_tokenize(text))]
    except Exception:
        return keys
    n = len(tags) or 1
    for k, grp in _POS_GROUPS.items():
        keys[f"pos_{k}"] = sum(1 for t in tags if t in grp) / n
    return keys


def engineered_features(df: pd.DataFrame, text_cols, use_pos: bool = True) -> pd.DataFrame:
    """Compute engineered features for each given text column.

    Returns a DataFrame (aligned to df.index) with prefixed columns.
    Deterministic per row -> safe to compute once across all splits.
    """
    frames = []
    for col in text_cols:
        rows = []
        for text in df[col].tolist():
            feat = {}
            feat.update(_basic_stats(text))
            feat.update(_readability(text))
            feat.update(_sentiment(text))
            if use_pos:
                feat.update(_pos_features(text))
            rows.append(feat)
        sub = pd.DataFrame(rows, index=df.index).add_prefix(f"{col}__")
        frames.append(sub)
    out = pd.concat(frames, axis=1)
    out = out.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return out
