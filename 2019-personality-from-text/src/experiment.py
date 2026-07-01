"""Experiment driver: build base learners, run the stack, report Train-CV + Dev.

Test is NEVER read here. Use freeze_and_test.py for the single Test evaluation.
"""
from __future__ import annotations

import argparse

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import embeddings
from .data import SEED, TARGETS, TRAITS, load_data, split_data
from .features import char_vectorizer, engineered_features, word1_vectorizer, word_vectorizer
from .metrics import mean_r, report
from .stack import (
    LLMBase,
    LLMFeatureBase,
    LLMQBase,
    LLMQFeatureBase,
    MatchedPromptBase,
    StackBase,
    make_splits,
    run_stack,
)

# ---------------------------------------------------------------------------
# Cached extract helpers
# ---------------------------------------------------------------------------
_ENG_CACHE: dict = {}


def _engineered(df, cols):
    ids = tuple(df["Respondent_ID"].tolist())
    key = (ids, tuple(cols))
    if key not in _ENG_CACHE:
        _ENG_CACHE[key] = engineered_features(df, cols).to_numpy(dtype=float)
    return _ENG_CACHE[key]


def text_all(df):
    return df["all_text"].to_numpy()


def emb_extract(model_name):
    def fn(df):
        return embeddings.encode(df["all_text"].tolist(), model_name)
    return fn


def emb_concat_prompts_extract(model_name):
    """Encode each prompt separately and concatenate -> richer per-trait signal."""
    from .data import TEXT_COLS

    def fn(df):
        mats = [embeddings.encode(df[c].tolist(), model_name) for c in TEXT_COLS]
        return np.concatenate(mats, axis=1)
    return fn


# ---------------------------------------------------------------------------
# Base learner registry
# ---------------------------------------------------------------------------
def ridge_pipe(alpha):
    return lambda: Pipeline([("sc", StandardScaler(with_mean=True)), ("rdg", Ridge(alpha=alpha))])


def svr_pipe(C, gamma="scale", epsilon=0.1, pca=None):
    from sklearn.decomposition import PCA
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.svm import SVR

    def make():
        steps = [("sc", StandardScaler(with_mean=True))]
        if pca:
            steps.append(("pca", PCA(n_components=pca, random_state=SEED)))
        steps.append(("svr", MultiOutputRegressor(
            SVR(C=C, gamma=gamma, epsilon=epsilon, kernel="rbf"))))
        return Pipeline(steps)
    return make


def gbm_pipe(**gkw):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.multioutput import MultiOutputRegressor
    params = dict(max_depth=3, learning_rate=0.05, max_iter=300,
                  l2_regularization=1.0, min_samples_leaf=20, random_state=SEED)
    params.update(gkw)
    return lambda: MultiOutputRegressor(HistGradientBoostingRegressor(**params))


def sparse_pipe(vec_factory, alpha):
    return lambda: Pipeline([("vec", vec_factory()), ("rdg", Ridge(alpha=alpha))])


def _eng_cols(kw):
    from .data import TEXT_COLS
    return kw.get("cols", ["all_text"] + TEXT_COLS)


def make_base(name, **kw):
    if name == "tfidf_word_all":
        return StackBase("tfidf_word_all", text_all, sparse_pipe(word_vectorizer, kw.get("alpha", 8.0)))
    if name == "tfidf_word1_all":
        return StackBase("tfidf_word1_all", text_all, sparse_pipe(word1_vectorizer, kw.get("alpha", 8.0)))
    if name == "tfidf_char_all":
        return StackBase("tfidf_char_all", text_all, sparse_pipe(char_vectorizer, kw.get("alpha", 8.0)))
    if name == "matched_word":
        return MatchedPromptBase("matched_word", sparse_pipe(word_vectorizer, kw.get("alpha", 8.0)))
    if name == "matched_char":
        return MatchedPromptBase("matched_char", sparse_pipe(char_vectorizer, kw.get("alpha", 8.0)))
    if name == "engineered":
        return StackBase("engineered", lambda df: _engineered(df, _eng_cols(kw)), ridge_pipe(kw.get("alpha", 5.0)))
    if name == "engineered_gbm":
        return StackBase("engineered_gbm", lambda df: _engineered(df, _eng_cols(kw)), gbm_pipe(**kw.get("gbm", {})))
    if name.startswith("emb:"):
        model = name.split("emb:", 1)[1]
        return StackBase(name, emb_extract(model), ridge_pipe(kw.get("alpha", 30.0)))
    if name.startswith("embp:"):
        model = name.split("embp:", 1)[1]
        return StackBase(name, emb_concat_prompts_extract(model), ridge_pipe(kw.get("alpha", 30.0)))
    if name.startswith("embsvr:"):
        model = name.split("embsvr:", 1)[1]
        return StackBase(name, emb_extract(model), svr_pipe(kw.get("C", 4.0), kw.get("gamma", "scale"), kw.get("epsilon", 0.1), kw.get("pca")))
    if name.startswith("embpsvr:"):
        model = name.split("embpsvr:", 1)[1]
        return StackBase(name, emb_concat_prompts_extract(model), svr_pipe(kw.get("C", 4.0), kw.get("gamma", "scale"), kw.get("epsilon", 0.1), kw.get("pca")))
    if name.startswith("llm:"):
        # llm:claude-3-5-haiku-20241022  or  llm:gemini-2.0-flash@gemini
        spec = name.split("llm:", 1)[1]
        if "@" in spec:
            model, provider = spec.rsplit("@", 1)
        else:
            model, provider = spec, "anthropic"
        variant = kw.get("variant", "general")
        base_name = f"{name}#{variant}"
        return LLMBase(base_name, model, provider=provider, variant=variant)
    if name.startswith("llmfeat:"):
        spec = name.split("llmfeat:", 1)[1]
        if "@" in spec:
            model, provider = spec.rsplit("@", 1)
        else:
            model, provider = spec, "anthropic"
        return LLMFeatureBase(name, model, ridge_pipe(kw.get("alpha", 5.0)), provider=provider)
    if name.startswith("llmq:"):
        spec = name.split("llmq:", 1)[1]
        if "@" in spec:
            model, provider = spec.rsplit("@", 1)
        else:
            model, provider = spec, "anthropic"
        return LLMQBase(name, model, provider=provider)
    if name.startswith("llmqfeat:"):
        spec = name.split("llmqfeat:", 1)[1]
        if "@" in spec:
            model, provider = spec.rsplit("@", 1)
        else:
            model, provider = spec, "anthropic"
        return LLMQFeatureBase(name, model, ridge_pipe(kw.get("alpha", 5.0)), provider=provider)
    if name.startswith("llmqs:"):
        spec = name.split("llmqs:", 1)[1]
        if "@" in spec:
            model, provider = spec.rsplit("@", 1)
        else:
            model, provider = spec, "anthropic"
        return LLMQBase(name, model, provider=provider, use_summary=True)
    if name.startswith("llmqsfeat:"):
        spec = name.split("llmqsfeat:", 1)[1]
        if "@" in spec:
            model, provider = spec.rsplit("@", 1)
        else:
            model, provider = spec, "anthropic"
        return LLMQFeatureBase(name, model, ridge_pipe(kw.get("alpha", 5.0)),
                               provider=provider, use_summary=True)
    raise ValueError(f"unknown base {name}")


def build_bases(specs):
    """specs: list of (name, kwargs)."""
    return [make_base(n, **kw) for n, kw in specs]


# ---------------------------------------------------------------------------
# Evaluation entry points
# ---------------------------------------------------------------------------
def evaluate(specs, meta_alpha=1.0, cross_trait=False, n_splits=5, seed=SEED,
             label="", show_bases=True):
    df = load_data()
    train, dev, _test = split_data(df)
    Y = train[TARGETS].to_numpy(dtype=float)
    Ydev = dev[TARGETS].to_numpy(dtype=float)

    bases = build_bases(specs)
    res = run_stack(bases, train, {"dev": dev}, Y,
                    n_splits=n_splits, seed=seed,
                    meta_alpha=meta_alpha, cross_trait=cross_trait)

    print(f"\n=== {label or 'config'} (meta_alpha={meta_alpha}, cross_trait={cross_trait}) ===")
    if show_bases:
        for name in res["base_oof"]:
            report(f"oof:{name}", Y, res["base_oof"][name])
        for name in res["base_eval"]["dev"]:
            report(f"dev:{name}", Ydev, res["base_eval"]["dev"][name])
    oof_scores = report("STACK oof", Y, res["oof"])
    dev_scores = report("STACK dev", Ydev, res["eval"]["dev"])
    return {"oof": oof_scores, "dev": dev_scores, "res": res}


def evaluate_repeated(specs, meta_alpha=1.0, cross_trait=False, n_splits=5,
                      seeds=(13, 42, 101), label=""):
    """Repeated-CV for robust selection: report OOF mean/std + Dev mean/std.

    Base full-fit (Dev) predictions are seed-independent; only the meta-learner
    (fit on OOF) varies with the split seed, so Dev variance here is small and
    reflects meta sensitivity.
    """
    df = load_data()
    train, dev, _test = split_data(df)
    Y = train[TARGETS].to_numpy(dtype=float)
    Ydev = dev[TARGETS].to_numpy(dtype=float)

    oof_means, dev_means = [], []
    for s in seeds:
        bases = build_bases(specs)
        res = run_stack(bases, train, {"dev": dev}, Y, n_splits=n_splits,
                        seed=s, meta_alpha=meta_alpha, cross_trait=cross_trait)
        oof_means.append(mean_r(Y, res["oof"]))
        dev_means.append(mean_r(Ydev, res["eval"]["dev"]))
    oof_means, dev_means = np.array(oof_means), np.array(dev_means)
    print(f"\n=== {label or 'config'} [repeated] (meta_alpha={meta_alpha}, "
          f"cross_trait={cross_trait}, seeds={seeds}) ===")
    print(f"  OOF mean_r = {oof_means.mean():.4f} +/- {oof_means.std():.4f}   {np.round(oof_means,4)}")
    print(f"  DEV mean_r = {dev_means.mean():.4f} +/- {dev_means.std():.4f}   {np.round(dev_means,4)}")
    return {"oof_mean": float(oof_means.mean()), "oof_std": float(oof_means.std()),
            "dev_mean": float(dev_means.mean()), "dev_std": float(dev_means.std())}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="baseline")
    args = p.parse_args()

    if args.config == "baseline":
        specs = [("tfidf_word_all", {}), ("tfidf_char_all", {}),
                 ("matched_word", {}), ("matched_char", {})]
        evaluate(specs, label="TF-IDF baseline stack")
