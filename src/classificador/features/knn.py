from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from classificador.features.embeddings import EmbeddingIndex
from classificador.themes.tree import ThemeTree


class FolhaKNN:
    """kNN sobre embeddings das questões rotuladas da folha (offline)."""

    def __init__(
        self,
        emb: EmbeddingIndex,
        labeled: list[tuple[str, int | None, bool]],
        tree: ThemeTree,
        *,
        k: int = 5,
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
        if self.X.shape[0] == 0 or questao_id not in emb._q_pos:
            return {"tema_id": None, "confidence": 0.0, "fora_votes": 0.0, "agree": 0.0}
        q = emb.questao_emb[emb._q_pos[questao_id]]
        # exclude self if present
        sims = self.X @ q
        order = np.argsort(-sims)
        picked = []
        for i in order:
            if self.ids[int(i)] == questao_id:
                continue
            picked.append(int(i))
            if len(picked) >= self.k:
                break
        if not picked:
            return {"tema_id": None, "confidence": 0.0, "fora_votes": 0.0, "agree": 0.0}

        fora_votes = sum(1 for i in picked if self.fora[i]) / len(picked)
        temas = [self.labels[i] for i in picked if not self.fora[i] and self.labels[i] is not None]
        if not temas:
            return {
                "tema_id": None,
                "confidence": fora_votes,
                "fora_votes": fora_votes,
                "agree": fora_votes,
            }

        # Vote by branch-collapsed label (prefer exact id frequency)
        counts = Counter(temas)
        best_tema, best_n = counts.most_common(1)[0]
        # Also count same-branch votes
        branch_n = 0
        for t in temas:
            if t == best_tema or self.tree.same_branch(int(t), int(best_tema)):
                branch_n += 1
        agree = branch_n / len(picked)
        conf = agree * float(np.mean([sims[i] for i in picked]))
        return {
            "tema_id": int(best_tema),
            "confidence": float(conf),
            "fora_votes": float(fora_votes),
            "agree": float(agree),
            "neighbor_sims": [float(sims[i]) for i in picked],
        }

    def theme_prototypes(self, emb: EmbeddingIndex) -> dict[int, np.ndarray]:
        buckets: dict[int, list[np.ndarray]] = defaultdict(list)
        for qid, tema, is_fora in zip(self.ids, self.labels, self.fora):
            if is_fora or tema is None:
                continue
            buckets[int(tema)].append(emb.questao_emb[emb._q_pos[qid]])
        out = {}
        for tid, vecs in buckets.items():
            m = np.mean(np.stack(vecs, axis=0), axis=0)
            n = np.linalg.norm(m) + 1e-12
            out[tid] = (m / n).astype(np.float32)
        return out
