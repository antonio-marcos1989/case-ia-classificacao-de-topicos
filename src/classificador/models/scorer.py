from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from classificador.features.embeddings import EmbeddingIndex
from classificador.features.knn import FolhaKNN
from classificador.features.lexical import LexicalIndex
from classificador.knowledge.aliases import is_out_of_scope_disciplina, map_disciplina
from classificador.knowledge.laws import LawDictionary
from classificador.themes.tree import ThemeTree


@dataclass
class Candidate:
    tema_id: int
    score: float
    components: dict[str, float]


@dataclass
class ScoreResult:
    candidates: list[Candidate]
    best: Candidate | None
    second: Candidate | None
    sibling_tie: bool
    fora_signal: float
    law_hit: str | None
    notes: list[str]
    confidence_features: dict[str, float]


class HybridScorer:
    """Combina TF-IDF, embeddings, kNN da folha, leis e fuzzy de assunto."""

    def __init__(
        self,
        tree: ThemeTree,
        lexical: LexicalIndex,
        laws: LawDictionary,
        embeddings: EmbeddingIndex | None = None,
        knn: FolhaKNN | None = None,
        *,
        w_tfidf: float = 0.40,
        w_emb: float = 0.15,
        w_proto: float = 0.10,
        w_law: float = 0.30,
        w_assunto: float = 0.35,
        w_disc: float = 0.12,
        w_knn: float = 0.25,
        sibling_margin: float = 0.03,
    ):
        self.tree = tree
        self.lexical = lexical
        self.laws = laws
        self.embeddings = embeddings
        self.knn = knn
        self.w_tfidf = w_tfidf
        self.w_emb = w_emb
        self.w_proto = w_proto
        self.w_law = w_law
        self.w_assunto = w_assunto
        self.w_disc = w_disc
        self.w_knn = w_knn
        self.sibling_margin = sibling_margin
        self.tema_ids = list(lexical.tema_ids)
        self._pos = {t: i for i, t in enumerate(self.tema_ids)}
        self.prototypes: dict[int, np.ndarray] = {}
        if knn is not None and embeddings is not None:
            self.prototypes = knn.theme_prototypes(embeddings)

    def score_row(self, row: pd.Series) -> ScoreResult:
        text = str(row.get("texto") or "")
        qid = str(row.get("id") or "")
        assunto = row.get("assunto_origem")
        disc_origem = row.get("disciplina_origem")

        n = len(self.tema_ids)
        tfidf = self.lexical.scores(text)
        emb = np.zeros(n, dtype=float)
        if self.embeddings is not None:
            sim = self.embeddings.scores_for_questao(qid)
            if sim is not None:
                for tid, s in zip(self.embeddings.tema_ids, sim):
                    if tid in self._pos:
                        emb[self._pos[tid]] = float(s)

        proto = np.zeros(n, dtype=float)
        if self.embeddings is not None and self.prototypes and qid in self.embeddings._q_pos:
            qv = self.embeddings.questao_emb[self.embeddings._q_pos[qid]]
            for tid, pv in self.prototypes.items():
                if tid in self._pos:
                    proto[self._pos[tid]] = float(np.dot(qv, pv))

        law_boost = np.zeros(n, dtype=float)
        law_hit_name = None
        # Só credita lei se o tema também aparece no top TF-IDF (reduz falso positivo).
        tfidf_top = set(self.tema_ids[i] for i in np.argsort(-tfidf)[:25])
        law_hits = self.laws.match(text)
        for hit in law_hits:
            if hit.tema_id is None:
                continue
            related = {hit.tema_id, *self.tree.children.get(hit.tema_id, [])}
            if related.isdisjoint(tfidf_top) and hit.tema_id not in tfidf_top:
                # ainda assim permite se título da lei casa forte no texto via assunto path
                pass
            law_hit_name = hit.name
            if hit.tema_id in self._pos:
                # downgrade se fora do top lexical
                strength = hit.score if hit.tema_id in tfidf_top or not related.isdisjoint(tfidf_top) else 0.35
                law_boost[self._pos[hit.tema_id]] = max(
                    law_boost[self._pos[hit.tema_id]], strength
                )
                for child in self.tree.children.get(hit.tema_id, []):
                    if child in self._pos:
                        law_boost[self._pos[child]] = max(
                            law_boost[self._pos[child]], 0.55 * strength
                        )

        assunto_boost = np.zeros(n, dtype=float)
        for tid, s in self.lexical.assunto_fuzzy(assunto, top_k=8):
            if tid in self._pos and s >= 0.80:
                assunto_boost[self._pos[tid]] = s

        disc = map_disciplina(str(disc_origem) if disc_origem is not None else None)
        disc_boost = np.zeros(n, dtype=float)
        if disc:
            for i, tid in enumerate(self.tema_ids):
                if self.tree.disciplina(tid) == disc:
                    disc_boost[i] = 1.0

        knn_boost = np.zeros(n, dtype=float)
        knn_info = {"tema_id": None, "agree": 0.0, "fora_votes": 0.0, "confidence": 0.0}
        if self.knn is not None and self.embeddings is not None:
            knn_info = self.knn.vote(qid, self.embeddings)
            kt = knn_info.get("tema_id")
            if kt is not None and kt in self._pos:
                knn_boost[self._pos[int(kt)]] = float(knn_info.get("agree") or 0.0)
                parent = self.tree.parent(int(kt))
                if parent is not None and parent in self._pos:
                    knn_boost[self._pos[parent]] = max(
                        knn_boost[self._pos[parent]], 0.6 * float(knn_info.get("agree") or 0)
                    )

        emb_n = np.clip(emb, 0, 1)
        proto_n = np.clip(proto, 0, 1)
        combined = (
            self.w_tfidf * tfidf
            + self.w_emb * emb_n
            + self.w_proto * proto_n
            + self.w_law * law_boost
            + self.w_assunto * assunto_boost
            + self.w_disc * disc_boost
            + self.w_knn * knn_boost
        )

        # Restringe à disciplina mapeada quando o sinal existe (hierarquia leve).
        eligible = np.ones(n, dtype=bool)
        if disc:
            for i, tid in enumerate(self.tema_ids):
                if self.tree.disciplina(tid) != disc:
                    eligible[i] = False
            # se filtro zerar tudo, ignora
            if not eligible.any():
                eligible[:] = True

        order = np.argsort(-combined)
        candidates: list[Candidate] = []
        for i in order:
            if not eligible[int(i)]:
                continue
            tid = self.tema_ids[int(i)]
            candidates.append(
                Candidate(
                    tema_id=tid,
                    score=float(combined[i]),
                    components={
                        "tfidf": float(tfidf[i]),
                        "emb": float(emb_n[i]),
                        "proto": float(proto_n[i]),
                        "law": float(law_boost[i]),
                        "assunto": float(assunto_boost[i]),
                        "disc": float(disc_boost[i]),
                        "knn": float(knn_boost[i]),
                    },
                )
            )
            if len(candidates) >= 20:
                break

        best = candidates[0] if candidates else None
        second = candidates[1] if len(candidates) > 1 else None
        sibling_tie = False
        notes: list[str] = []

        # Prefer kNN theme when strong agreement and compatible with top lexical/emb
        if (
            best
            and knn_info.get("tema_id") is not None
            and float(knn_info.get("agree") or 0) >= 0.6
            and float(knn_info.get("fora_votes") or 0) < 0.4
        ):
            kt = int(knn_info["tema_id"])
            if kt in self._pos and eligible[self._pos[kt]]:
                top_ids = {c.tema_id for c in candidates[:8]}
                if kt in top_ids or self.tree.same_branch(kt, best.tema_id):
                    pos = self._pos[kt]
                    best = Candidate(
                        tema_id=kt,
                        score=float(combined[pos]) + 0.05,
                        components={
                            "tfidf": float(tfidf[pos]),
                            "emb": float(emb_n[pos]),
                            "proto": float(proto_n[pos]),
                            "law": float(law_boost[pos]),
                            "assunto": float(assunto_boost[pos]),
                            "disc": float(disc_boost[pos]),
                            "knn": float(knn_boost[pos]),
                        },
                    )
                    notes.append("knn_folha")
                    # refresh second
                    others = [c for c in candidates if c.tema_id != best.tema_id]
                    second = others[0] if others else None

        # Preferência forte por assunto quase-exato (≥0.95) dentro da disciplina
        if assunto and str(assunto).strip():
            for tid, s in self.lexical.assunto_fuzzy(assunto, top_k=3):
                if s >= 0.95 and tid in self._pos and eligible[self._pos[tid]]:
                    pos = self._pos[tid]
                    best = Candidate(
                        tema_id=tid,
                        score=max(float(combined[pos]), 0.55) + 0.1 * s,
                        components={
                            "tfidf": float(tfidf[pos]),
                            "emb": float(emb_n[pos]),
                            "proto": float(proto_n[pos]),
                            "law": float(law_boost[pos]),
                            "assunto": float(s),
                            "disc": float(disc_boost[pos]),
                            "knn": float(knn_boost[pos]),
                        },
                    )
                    notes.append("assunto_exact")
                    break

        if best and second and self.tree.are_siblings(best.tema_id, second.tema_id):
            if abs(best.score - second.score) <= self.sibling_margin:
                sibling_tie = True
                notes.append(f"empate entre irmãos {best.tema_id} e {second.tema_id}")

        fora_signal = 0.0
        if best is None or best.score < 0.18:
            fora_signal += 0.35
        if is_out_of_scope_disciplina(str(disc_origem) if disc_origem is not None else None):
            fora_signal += 0.45
            notes.append("disciplina_origem fora do recorte")
        fora_signal += 0.55 * float(knn_info.get("fora_votes") or 0.0)
        if best and best.components.get("tfidf", 0) < 0.04 and best.components.get("law", 0) == 0:
            fora_signal += 0.15
        if law_hit_name:
            notes.append(f"lei:{law_hit_name}")

        # Confidence features for calibration (not just raw score)
        margin = 0.0
        if best and second:
            margin = max(0.0, best.score - second.score)
        agree_signals = 0.0
        if best:
            c = best.components
            agree_signals = (
                (1.0 if c.get("law", 0) > 0 else 0.0)
                + (1.0 if c.get("knn", 0) >= 0.6 else 0.0)
                + (1.0 if c.get("assunto", 0) >= 0.85 else 0.0)
                + (1.0 if c.get("emb", 0) >= 0.45 else 0.0)
                + (1.0 if c.get("proto", 0) >= 0.55 else 0.0)
            ) / 5.0
        conf_feat = {
            "raw": float(best.score if best else 0.0),
            "margin": float(margin),
            "knn_agree": float(knn_info.get("agree") or 0.0),
            "law": float(best.components.get("law", 0) if best else 0.0),
            "agree_signals": float(agree_signals),
            "fora_signal": float(min(1.0, fora_signal)),
        }
        # Scalar used by calibrator: blend of strength + agreement
        conf_feat["score"] = (
            0.35 * conf_feat["raw"]
            + 0.20 * conf_feat["margin"]
            + 0.25 * conf_feat["knn_agree"]
            + 0.15 * conf_feat["law"]
            + 0.25 * conf_feat["agree_signals"]
            - 0.30 * conf_feat["fora_signal"]
        )

        return ScoreResult(
            candidates=candidates,
            best=best,
            second=second,
            sibling_tie=sibling_tie,
            fora_signal=min(1.0, fora_signal),
            law_hit=law_hit_name,
            notes=notes,
            confidence_features=conf_feat,
        )
