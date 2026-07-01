"""Model selection utilities.

Decouples expensive base-learner computation from cheap meta-layer tuning by
caching each base's OOF predictions (per CV seed) and Dev predictions to disk.
Meta configurations (which bases, meta_alpha, cross_trait) can then be swept
instantly over repeated CV.

INTEGRITY: this module never computes or reads Test. Test is produced exactly
once by freeze_and_test.py.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import PROJECT_ROOT, TARGETS, TRAITS, load_data, split_data
from .experiment import build_bases
from .metrics import mean_r, per_trait_r
from .stack import build_meta_features, make_splits

CACHE = os.path.join(PROJECT_ROOT, "data", "processed", "base_pred_cache")


def _spec_key(spec):
    name, kw = spec
    blob = name + "|" + json.dumps(kw, sort_keys=True, default=str)
    return name.replace("/", "_").replace(":", "_") + "_" + hashlib.sha1(blob.encode()).hexdigest()[:10]


def base_predictions(spec, seeds=(13, 42, 101)):
    """Return {'oof': {seed: (n,5)}, 'dev': (n,5)} for a base spec, cached."""
    os.makedirs(CACHE, exist_ok=True)
    key = _spec_key(spec)
    path = os.path.join(CACHE, key + ".npz")
    if os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        seeds_have = list(z["seeds"])
        if all(s in seeds_have for s in seeds):
            oof = {int(s): z[f"oof_{s}"] for s in seeds}
            return {"oof": oof, "dev": z["dev"]}

    df = load_data()
    train, dev, _test = split_data(df)
    Y = train[TARGETS].to_numpy(dtype=float)

    base = build_bases([spec])[0]
    oof = {}
    for s in seeds:
        splits = make_splits(len(train), seed=s)
        oof[s] = base.oof(train, Y, splits)
    base.fit_full(train, Y)
    dev_pred = base.predict(dev)

    save = {f"oof_{s}": oof[s] for s in seeds}
    save["dev"] = dev_pred
    save["seeds"] = np.array(list(seeds))
    np.savez(path, **save)
    return {"oof": oof, "dev": dev_pred}


def _meta_eval(base_oof_seed, base_dev, Y, Ydev, meta_alpha, cross_trait):
    """One meta fit on a single seed's OOF; return (oof_r, dev_r, oof_pred, dev_pred)."""
    sel_oof = build_meta_features(base_oof_seed, cross_trait)
    sel_dev = build_meta_features(base_dev, cross_trait)
    oof_pred = np.zeros_like(Y)
    dev_pred = np.zeros((Ydev.shape[0], len(TRAITS)))
    for i in range(len(TRAITS)):
        meta = Pipeline([("sc", StandardScaler()), ("rdg", Ridge(alpha=meta_alpha))])
        meta.fit(sel_oof(i), Y[:, i])
        oof_pred[:, i] = meta.predict(sel_oof(i))
        dev_pred[:, i] = meta.predict(sel_dev(i))
    return mean_r(Y, oof_pred), mean_r(Ydev, dev_pred), oof_pred, dev_pred


def meta_sweep(specs, seeds=(13, 42, 101), meta_alphas=(0.5, 1.0, 2.0, 4.0),
               cross_options=(False, True), verbose=True):
    """Sweep meta configs over cached base predictions; report OOF/Dev mean+/-std."""
    df = load_data()
    train, dev, _test = split_data(df)
    Y = train[TARGETS].to_numpy(dtype=float)
    Ydev = dev[TARGETS].to_numpy(dtype=float)

    preds = {tuple(s if not isinstance(s, dict) else json.dumps(s) for s in [spec[0]]): None for spec in specs}
    cache = {}
    for spec in specs:
        cache[spec[0]] = base_predictions(spec, seeds=seeds)

    results = []
    for ma in meta_alphas:
        for ct in cross_options:
            oof_rs, dev_rs = [], []
            for s in seeds:
                base_oof_seed = {name: cache[name]["oof"][s] for name in cache}
                base_dev = {name: cache[name]["dev"] for name in cache}
                o, d, _, _ = _meta_eval(base_oof_seed, base_dev, Y, Ydev, ma, ct)
                oof_rs.append(o)
                dev_rs.append(d)
            oof_rs, dev_rs = np.array(oof_rs), np.array(dev_rs)
            row = {"meta_alpha": ma, "cross_trait": ct,
                   "oof_mean": oof_rs.mean(), "oof_std": oof_rs.std(),
                   "dev_mean": dev_rs.mean(), "dev_std": dev_rs.std()}
            results.append(row)
            if verbose:
                print(f"alpha={ma:<4} cross={str(ct):<5} "
                      f"OOF={oof_rs.mean():.4f}+/-{oof_rs.std():.4f}  "
                      f"DEV={dev_rs.mean():.4f}+/-{dev_rs.std():.4f}")
    return results, cache
