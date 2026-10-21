from __future__ import annotations

from collections import defaultdict

import numpy as np

from classificador.features.embeddings import EmbeddingIndex
from classificador.themes.tree import ThemeTree


class FolhaKNN:
    """kNN ponderado por similaridade sobre embeddings da folha."""

    def __init__(
        self,
        emb: EmbeddingIndex,
        labeled: list[tuple[str, int | None, bool]],
        tree: ThemeTree,
        *,
        k: int = 7,
    ):
        """labeled: (questao_id, tema_central_id|None, fora)."""
        self.tree = tree
        self.k = k
        self.ids: list[str] = []
        self.labels: list[int | None] = []
        self.fora: list[bool] = []
        vectors = []
        for qid, tema, is_fora in labeled:
            if qid not in emb._q_pos:
                continue
            self.ids.append(qid)
            self.labels.append(tema)
            self.fora.append(is_fora)
            vectors.append(emb.questao_emb[emb._q_pos[qid]])
        self.X = np.asarray(vectors, dtype=np.float32) if vectors else np.zeros((0, 1), dtype=np.float32)

    def vote(self, questao_id: str, emb: EmbeddingIndex) -> dict:
        empty = {"tema_id": None, "confidence": 0.0, "fora_votes": 0.0, "agree": 0.0, "sim_max": 0.0}
        if self.X.shape[0] == 0 or questao_id not in emb._q_pos:
            return empty
        q = emb.questao_emb[emb._q_pos[questao_id]]
        sims = self.X @ q
        order = np.argsort(-sims)
        picked: list[int] = []
        for i in order:
            if self.ids[int(i)] == questao_id:
                continue
            picked.append(int(i))
            if len(picked) >= self.k:
                break
        if not picked:
            return empty

        weights = np.array([max(float(sims[i]), 0.0) ** 2 for i in picked], dtype=float)
        if weights.sum() <= 1e-12:
            weights = np.ones(len(picked), dtype=float)
        weights = weights / weights.sum()

        fora_votes = float(sum(weights[j] for j, i in enumerate(picked) if self.fora[i]))
        sim_max = float(max(sims[i] for i in picked))

        # Agrega peso por tema e por ramo (pai)
        tema_w: dict[int, float] = defaultdict(float)
        for j, i in enumerate(picked):
            if self.fora[i] or self.labels[i] is None:
                continue
            tid = int(self.labels[i])
            tema_w[tid] += weights[j]
            parent = self.tree.parent(tid)
            if parent is not None and self.tree.nivel(parent) >= 1:
                tema_w[parent] += 0.35 * weights[j]

        if not tema_w:
            return {
                "tema_id": None,
                "confidence": fora_votes,
                "fora_votes": fora_votes,
                "agree": fora_votes,
                "sim_max": sim_max,
            }

        best_tema = max(tema_w.items(), key=lambda kv: kv[1])[0]
        # agree = peso no mesmo ramo do vencedor
        branch_w = 0.0
        for tid, w in tema_w.items():
            if tid == best_tema or self.tree.same_branch(tid, best_tema):
                branch_w += w
        agree = float(min(1.0, branch_w))
        conf = agree * sim_max
        return {
            "tema_id": int(best_tema),
            "confidence": float(conf),
            "fora_votes": float(fora_votes),
            "agree": agree,
            "sim_max": sim_max,
            "neighbor_sims": [float(sims[i]) for i in picked],
        }

    def theme_prototypes(self, emb: EmbeddingIndex) -> dict[int, np.ndarray]:
        """Protótipos ponderados por tema + suave no pai (nível 1).

        Temas raros pesam um pouco mais (IDF-like) para não serem dominados por
        classes frequentes na folha.
        """
        buckets: dict[int, list[np.ndarray]] = defaultdict(list)
        counts: dict[int, int] = defaultdict(int)
        for qid, tema, is_fora in zip(self.ids, self.labels, self.fora):
            if is_fora or tema is None:
                continue
            tid = int(tema)
            vec = emb.questao_emb[emb._q_pos[qid]]
            buckets[tid].append(vec)
            counts[tid] += 1
            parent = self.tree.parent(tid)
            if parent is not None and self.tree.nivel(parent) == 1:
                buckets[parent].append(vec)
                counts[parent] += 1

        n_lab = max(sum(1 for f in self.fora if not f), 1)
        out = {}
        for tid, vecs in buckets.items():
            stack = np.stack(vecs, axis=0)
            # Soft IDF: temas raros → peso ligeiramente maior no mean
            idf = float(np.log(1.0 + n_lab / max(counts.get(tid, 1), 1)))
            w = np.full(len(vecs), idf, dtype=np.float32)
            w = w / (w.sum() + 1e-12)
            m = (stack * w[:, None]).sum(axis=0)
            n = np.linalg.norm(m) + 1e-12
            out[tid] = (m / n).astype(np.float32)
        return out
