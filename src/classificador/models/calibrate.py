from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


FEATURE_ORDER = [
    "raw",
    "margin",
    "knn_agree",
    "law",
    "agree_signals",
    "fora_signal",
    "tfidf",
    "emb",
    "proto",
    "assunto",
]


@dataclass
class Calibrator:
    kind: str
    model: object

    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        """Aceita vetor 1D (score) ou 2D (features; usa col 0)."""
        X = np.asarray(scores, dtype=float)
        if X.ndim == 1:
            return np.clip(self.model.predict(X), 0.0, 1.0)
        if self.kind == "lr":
            return self.model.predict_proba(X)[:, 1]
        return np.clip(self.model.predict(X[:, 0]), 0.0, 1.0)


def fit_calibrator(scores: np.ndarray, labels: np.ndarray) -> Calibrator:
    """Isotônica 1D (legado)."""
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if len(scores) < 3 or len(np.unique(labels)) < 2:
        mean = float(np.mean(labels)) if len(labels) else 0.5

        class _Const:
            def predict(self, x):
                return np.full(len(x), mean, dtype=float)

        return Calibrator(kind="iso", model=_Const())
    iso.fit(scores, labels)
    return Calibrator(kind="iso", model=iso)


def fit_feature_calibrator(X: np.ndarray, labels: np.ndarray) -> Calibrator:
    """Isotônica no score composto (col 0) — melhor separação que LR balanceada em n pequeno."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(labels, dtype=float)
    scores = X[:, 0] if X.ndim == 2 else np.asarray(X, dtype=float)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    if len(scores) < 5 or len(np.unique(y)) < 2:
        mean = float(np.mean(y)) if len(y) else 0.5

        class _Const:
            def predict(self, x):
                return np.full(len(x), mean, dtype=float)

        return Calibrator(kind="iso", model=_Const())
    iso.fit(scores, y)
    return Calibrator(kind="iso", model=iso)


def features_from_result(result) -> np.ndarray:
    cf = result.confidence_features
    c = result.best.components if result.best else {}
    # col0 = score composto (margem-ênfase) para isotônica
    return np.asarray(
        [
            cf.get("score", 0.0),
            cf.get("raw", 0.0),
            cf.get("margin", 0.0),
            cf.get("knn_agree", 0.0),
            cf.get("law", 0.0),
            cf.get("agree_signals", 0.0),
            cf.get("fora_signal", 0.0),
            c.get("tfidf", 0.0),
            c.get("emb", 0.0),
            c.get("proto", 0.0),
            c.get("assunto", 0.0),
        ],
        dtype=float,
    )


def choose_threshold(
    probs: np.ndarray,
    correct: np.ndarray,
    *,
    min_precision: float = 0.97,
    fallback_precision: float = 0.95,
    exclude: np.ndarray | None = None,
) -> dict:
    probs = np.asarray(probs, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    n = len(probs)
    if exclude is None:
        exclude = np.zeros(n, dtype=bool)
    else:
        exclude = np.asarray(exclude, dtype=bool)

    if n == 0:
        return {"tau": 0.99, "precision": 1.0, "coverage": 0.0, "n_pub": 0, "target": min_precision}

    candidates = sorted(
        set(np.round(probs[~exclude], 4).tolist()) | {0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99},
        reverse=True,
    )

    def eval_tau(tau: float, target: float) -> dict | None:
        mask = (probs >= tau) & (~exclude)
        pub = int(mask.sum())
        if pub == 0:
            return {"tau": tau, "precision": 1.0, "coverage": 0.0, "n_pub": 0, "target": target}
        prec = float(correct[mask].mean())
        cov = pub / n
        if prec + 1e-12 >= target:
            return {"tau": tau, "precision": prec, "coverage": cov, "n_pub": pub, "target": target}
        return None

    best = None
    for target in (min_precision, fallback_precision):
        for tau in candidates:
            res = eval_tau(tau, target)
            if res is None:
                continue
            if best is None or res["coverage"] > best["coverage"] or (
                res["coverage"] == best["coverage"] and res["precision"] > best["precision"]
            ):
                best = res
        if best is not None and best["n_pub"] > 0:
            break

    if best is None:
        best = {"tau": 0.99, "precision": 1.0, "coverage": 0.0, "n_pub": 0, "target": min_precision}
    return best
