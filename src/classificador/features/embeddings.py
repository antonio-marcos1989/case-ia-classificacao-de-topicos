from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DEFAULT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return x / norms


class EmbeddingIndex:
    """Similaridade cosseno com embeddings pré-computados (offline)."""

    def __init__(
        self,
        tema_ids: list[int],
        theme_emb: np.ndarray,
        questao_ids: list[str],
        questao_emb: np.ndarray,
    ):
        assert theme_emb.shape[0] == len(tema_ids)
        assert questao_emb.shape[0] == len(questao_ids)
        self.tema_ids = list(tema_ids)
        self.theme_emb = _l2_normalize(np.asarray(theme_emb, dtype=np.float32))
        self.questao_ids = list(questao_ids)
        self.questao_emb = _l2_normalize(np.asarray(questao_emb, dtype=np.float32))
        self._q_pos = {qid: i for i, qid in enumerate(self.questao_ids)}
        self._t_pos = {tid: i for i, tid in enumerate(self.tema_ids)}

    def scores_for_questao(self, questao_id: str) -> np.ndarray | None:
        i = self._q_pos.get(questao_id)
        if i is None:
            return None
        return self.theme_emb @ self.questao_emb[i]

    def top_k(self, questao_id: str, k: int = 10) -> list[tuple[int, float]]:
        sim = self.scores_for_questao(questao_id)
        if sim is None:
            return []
        idx = np.argsort(-sim)[:k]
        return [(self.tema_ids[i], float(sim[i])) for i in idx]


def try_load_embeddings(artifacts_dir: str | Path) -> EmbeddingIndex | None:
    root = Path(artifacts_dir)
    meta_path = root / "embeddings_meta.json"
    theme_path = root / "theme_embeddings.npy"
    questao_path = root / "questao_embeddings.npy"
    if not (meta_path.exists() and theme_path.exists() and questao_path.exists()):
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return EmbeddingIndex(
        tema_ids=[int(x) for x in meta["tema_ids"]],
        theme_emb=np.load(theme_path),
        questao_ids=list(meta["questao_ids"]),
        questao_emb=np.load(questao_path),
    )


def build_and_cache_embeddings(
    textos_questoes: list[tuple[str, str]],
    textos_temas: list[tuple[int, str]],
    artifacts_dir: str | Path,
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = 32,
) -> EmbeddingIndex:
    """Gera embeddings localmente (requer sentence-transformers) e grava cache."""
    from sentence_transformers import SentenceTransformer

    root = Path(artifacts_dir)
    root.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(model_name)

    q_ids = [q for q, _ in textos_questoes]
    q_texts = [t for _, t in textos_questoes]
    t_ids = [t for t, _ in textos_temas]
    t_texts = [x for _, x in textos_temas]

    q_emb = model.encode(q_texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
    t_emb = model.encode(t_texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)

    np.save(root / "questao_embeddings.npy", np.asarray(q_emb, dtype=np.float32))
    np.save(root / "theme_embeddings.npy", np.asarray(t_emb, dtype=np.float32))
    meta = {
        "model_name": model_name,
        "questao_ids": q_ids,
        "tema_ids": t_ids,
    }
    (root / "embeddings_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return EmbeddingIndex(t_ids, t_emb, q_ids, q_emb)
