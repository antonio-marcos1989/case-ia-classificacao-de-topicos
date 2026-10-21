from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression

from classificador.themes.tree import ThemeTree


@dataclass
class RankExample:
    features: np.ndarray
    label: int


class CandidateReranker:
    """Pointwise LR: P(tema certo | features do par questão–tema)."""

    def __init__(self, model: LogisticRegression | None = None):
        self.model = model

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CandidateReranker":
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=0.7)
        if len(np.unique(y)) < 2 or len(y) < 5:
            self.model = None
            return self
        clf.fit(X, y)
        self.model = clf
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if self.model is None:
            # fallback: use first feature (tfidf-ish)
            return np.clip(X[:, 0], 0, 1)
        return self.model.predict_proba(X)[:, 1]


def pair_features(
    *,
    tfidf: float,
    emb: float,
    proto: float,
    law: float,
    assunto: float,
    disc: float,
    knn: float,
    nivel: int,
    title_len: int,
) -> np.ndarray:
    return np.asarray(
        [
            tfidf,
            emb,
            proto,
            law,
            assunto,
            disc,
            knn,
            float(nivel) / 2.0,
            min(title_len, 80) / 80.0,
        ],
        dtype=float,
    )


def is_positive(pred: int, central: int | None, extras: list[int], tree: ThemeTree, fora: bool) -> bool:
    from classificador.themes.metrics import is_tolerant_match

    return is_tolerant_match(pred, central, extras, tree, fora=fora)
