from __future__ import annotations

import numpy as np
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from classificador.themes.tree import ThemeTree


class LexicalIndex:
    """TF-IDF sobre documentos de tema + fuzzy de assunto_origem."""

    def __init__(self, tree: ThemeTree, lexicon: dict[int, list[str]] | None = None):
        self.tree = tree
        self.tema_ids = tree.active(include_roots=False)
        docs = [tree.theme_document(tid, lexicon) for tid in self.tema_ids]
        self.vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            sublinear_tf=True,
        )
        self.theme_matrix = self.vectorizer.fit_transform(docs)
        self._id_to_pos = {tid: i for i, tid in enumerate(self.tema_ids)}

    def scores(self, text: str) -> np.ndarray:
        if not text.strip():
            return np.zeros(len(self.tema_ids), dtype=float)
        q = self.vectorizer.transform([text])
        sim = cosine_similarity(q, self.theme_matrix)[0]
        return np.asarray(sim, dtype=float)

    def top_k(self, text: str, k: int = 10) -> list[tuple[int, float]]:
        sim = self.scores(text)
        if len(sim) == 0:
            return []
        idx = np.argsort(-sim)[:k]
        return [(self.tema_ids[i], float(sim[i])) for i in idx]

    def assunto_fuzzy(self, assunto: str | None, top_k: int = 5) -> list[tuple[int, float]]:
        if not assunto or not str(assunto).strip():
            return []
        scored: list[tuple[int, float]] = []
        for tid in self.tema_ids:
            titulo = self.tree.titulo(tid)
            caminho = self.tree.path(tid)
            score = max(
                fuzz.token_set_ratio(assunto, titulo),
                fuzz.partial_ratio(assunto, titulo),
                fuzz.token_set_ratio(assunto, caminho),
            ) / 100.0
            scored.append((tid, score))
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]
