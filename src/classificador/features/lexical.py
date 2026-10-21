from __future__ import annotations

import re

import numpy as np
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from classificador.themes.tree import ThemeTree


def _bm25_scores(q_tf: np.ndarray, doc_tf: np.ndarray, doc_len: np.ndarray, *, k1: float = 1.5, b: float = 0.75) -> np.ndarray:
    """BM25 a partir de matrizes TF brutas (não-IDF) + IDF externo via pesos do vetorizador."""
    avgdl = float(doc_len.mean()) + 1e-12
    # q_tf: (1, V), doc_tf: (N, V)
    # Usa apenas termos presentes na query
    q_idx = np.flatnonzero(q_tf.ravel() > 0)
    if len(q_idx) == 0:
        return np.zeros(doc_tf.shape[0], dtype=float)
    scores = np.zeros(doc_tf.shape[0], dtype=float)
    for t in q_idx:
        tf = doc_tf[:, t]
        denom = tf + k1 * (1.0 - b + b * doc_len / avgdl)
        scores += (tf * (k1 + 1.0)) / (denom + 1e-12)
    return scores


class LexicalIndex:
    """TF-IDF + BM25 + char n-grams sobre documentos de tema (titulo+caminho+lexicon)."""

    def __init__(self, tree: ThemeTree, lexicon: dict[int, list[str]] | None = None):
        self.tree = tree
        self.tema_ids = tree.active(include_roots=False)
        docs = [tree.theme_document(tid, lexicon) for tid in self.tema_ids]
        self._docs = docs
        self._titles = [tree.titulo(tid).lower() for tid in self.tema_ids]
        self._paths = [tree.path(tid).lower() for tid in self.tema_ids]

        self.vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            sublinear_tf=True,
        )
        self.theme_matrix = self.vectorizer.fit_transform(docs)

        # Char n-grams capturam flexões PT (apelação/apelacao, júri/juri)
        self.char_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(4, 5),
            min_df=1,
            max_df=0.98,
            sublinear_tf=True,
        )
        self.char_matrix = self.char_vectorizer.fit_transform(docs)

        # Contagens brutas para BM25 (mesmo vocabulário word)
        self.count_vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            use_idf=False,
            norm=None,
            sublinear_tf=False,
        )
        self.count_matrix = self.count_vectorizer.fit_transform(docs).toarray().astype(float)
        self.doc_len = self.count_matrix.sum(axis=1)
        # IDF do word vectorizer para ponderar BM25
        idf = getattr(self.vectorizer, "idf_", None)
        self._bm25_idf = None
        if idf is not None:
            # Alinha vocabulários word TF-IDF ↔ count (mesmos params → mesmo vocab)
            vocab_tf = self.vectorizer.vocabulary_
            vocab_ct = self.count_vectorizer.vocabulary_
            self._bm25_idf = np.ones(len(vocab_ct), dtype=float)
            for term, j in vocab_ct.items():
                i = vocab_tf.get(term)
                if i is not None:
                    self._bm25_idf[j] = float(idf[i])

        self._id_to_pos = {tid: i for i, tid in enumerate(self.tema_ids)}

    def scores(self, text: str) -> np.ndarray:
        if not text.strip():
            return np.zeros(len(self.tema_ids), dtype=float)
        q = self.vectorizer.transform([text])
        cos = cosine_similarity(q, self.theme_matrix)[0]

        qc = self.char_vectorizer.transform([text])
        char = cosine_similarity(qc, self.char_matrix)[0]

        q_count = self.count_vectorizer.transform([text]).toarray().astype(float)
        bm25 = _bm25_scores(q_count, self.count_matrix, self.doc_len)
        if self._bm25_idf is not None:
            # Reaplica IDF termo a termo (aproximação)
            q_idx = np.flatnonzero(q_count.ravel() > 0)
            bm25 = np.zeros(len(self.tema_ids), dtype=float)
            avgdl = float(self.doc_len.mean()) + 1e-12
            k1, b = 1.5, 0.75
            for t in q_idx:
                tf = self.count_matrix[:, t]
                idf = float(self._bm25_idf[t])
                denom = tf + k1 * (1.0 - b + b * self.doc_len / avgdl)
                bm25 += idf * (tf * (k1 + 1.0)) / (denom + 1e-12)
        if bm25.max() > 1e-12:
            bm25 = bm25 / (bm25.max() + 1e-12)

        title_boost = self._title_token_boost(text)
        blended = 0.50 * cos + 0.22 * char + 0.20 * bm25 + 0.18 * title_boost
        # Normaliza para [0,1] relativo ao topo (mantém margens relativas)
        mx = float(blended.max()) if len(blended) else 0.0
        if mx > 1e-12:
            blended = blended / mx
        return np.asarray(blended, dtype=float)

    def _title_token_boost(self, text: str) -> np.ndarray:
        """Boost quando tokens distintivos do título/caminho aparecem no enunciado."""
        tnorm = text.lower()
        # Remove HTML residual
        tnorm = re.sub(r"<[^>]+>", " ", tnorm)
        out = np.zeros(len(self.tema_ids), dtype=float)
        for i, (title, path) in enumerate(zip(self._titles, self._paths)):
            # Tokens significativos do título (≥5 chars) ou bigrams legais curtos
            toks = [w for w in re.split(r"[^a-zà-ü0-9]+", title) if len(w) >= 5]
            if not toks:
                continue
            hits = sum(1 for w in toks if w in tnorm)
            if hits == 0:
                continue
            ratio = hits / max(len(toks), 1)
            # Caminho: último segmento (subtema) pesa mais
            leaf = path.split(">")[-1].strip() if path else ""
            leaf_hit = 1.0 if leaf and len(leaf) >= 5 and leaf.lower() in tnorm else 0.0
            out[i] = min(1.0, 0.55 * ratio + 0.55 * leaf_hit)
        return out

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
