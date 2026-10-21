from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from classificador.features.embeddings import EmbeddingIndex
from classificador.features.knn import FolhaKNN
from classificador.features.lexical import LexicalIndex
from classificador.knowledge.aliases import (
    is_out_of_scope_disciplina,
    map_disciplina,
    ood_assunto_or_text,
)
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
    """TF-IDF + embeddings + kNN/protótipos + leis, com margem e detector de fora."""

    def __init__(
        self,
        tree: ThemeTree,
        lexical: LexicalIndex,
        laws: LawDictionary,
        embeddings: EmbeddingIndex | None = None,
        knn: FolhaKNN | None = None,
        *,
        w_tfidf: float = 0.42,
        w_emb: float = 0.10,
        w_proto: float = 0.26,
        w_law: float = 0.34,
        w_assunto: float = 0.28,
        w_disc: float = 0.08,
        w_knn: float = 0.40,
        sibling_margin: float = 0.04,
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

    def _candidate_from_pos(self, pos: int, combined: np.ndarray, comps: dict[str, np.ndarray], bonus: float = 0.0) -> Candidate:
        tid = self.tema_ids[pos]
        return Candidate(
            tema_id=tid,
            score=float(combined[pos]) + bonus,
            components={k: float(v[pos]) for k, v in comps.items()},
        )

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
        law_forced: list[int] = []
        tfidf_top = set(self.tema_ids[i] for i in np.argsort(-tfidf)[:25])
        law_hits = self.laws.match(text)
        for hit in law_hits:
            if hit.tema_id is None:
                continue
            law_hit_name = hit.name
            related = {hit.tema_id, *self.tree.children.get(hit.tema_id, [])}
            in_top = hit.tema_id in tfidf_top or not related.isdisjoint(tfidf_top)
            # Hit específico (subtema) promove mesmo com lexical fraco
            specific = bool(getattr(hit, "specific", False))
            hit_w = float(getattr(hit, "score", 1.0))
            if specific:
                strong = 1.0 * min(hit_w, 1.2)
            elif in_top:
                strong = 1.0
            else:
                strong = 0.50
            if hit.tema_id in self._pos:
                law_boost[self._pos[hit.tema_id]] = max(law_boost[self._pos[hit.tema_id]], strong)
                if specific or in_top or strong >= 0.9:
                    law_forced.append(hit.tema_id)
                for child in self.tree.children.get(hit.tema_id, []):
                    if child in self._pos:
                        law_boost[self._pos[child]] = max(law_boost[self._pos[child]], 0.45 * strong)

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
        knn_info = {"tema_id": None, "agree": 0.0, "fora_votes": 0.0, "confidence": 0.0, "sim_max": 0.0}
        if self.knn is not None and self.embeddings is not None:
            knn_info = self.knn.vote(qid, self.embeddings)
            kt = knn_info.get("tema_id")
            if kt is not None and kt in self._pos:
                agree = float(knn_info.get("agree") or 0.0)
                knn_boost[self._pos[int(kt)]] = agree
                parent = self.tree.parent(int(kt))
                if parent is not None and parent in self._pos:
                    knn_boost[self._pos[parent]] = max(knn_boost[self._pos[parent]], 0.65 * agree)

        emb_n = np.clip(emb, 0, 1)
        proto_n = np.clip(proto, 0, 1)
        comps = {
            "tfidf": tfidf,
            "emb": emb_n,
            "proto": proto_n,
            "law": law_boost,
            "assunto": assunto_boost,
            "disc": disc_boost,
            "knn": knn_boost,
        }
        combined = (
            self.w_tfidf * tfidf
            + self.w_emb * emb_n
            + self.w_proto * proto_n
            + self.w_law * law_boost
            + self.w_assunto * assunto_boost
            + self.w_disc * disc_boost
            + self.w_knn * knn_boost
        )

        # Disciplina: permite temas forçados por lei (ex. LEP em Leg. Esp. com origem Penal)
        eligible = np.ones(n, dtype=bool)
        if disc:
            for i, tid in enumerate(self.tema_ids):
                if self.tree.disciplina(tid) != disc and tid not in law_forced:
                    # também permite filhos dos forçados
                    parents_ok = False
                    for lf in law_forced:
                        if tid in self.tree.children.get(lf, []) or self.tree.same_branch(tid, lf):
                            parents_ok = True
                            break
                    if not parents_ok:
                        eligible[i] = False
            if not eligible.any():
                eligible[:] = True

        order = np.argsort(-combined)
        candidates: list[Candidate] = []
        for i in order:
            if not eligible[int(i)]:
                continue
            candidates.append(self._candidate_from_pos(int(i), combined, comps))
            if len(candidates) >= 20:
                break

        best = candidates[0] if candidates else None
        second = candidates[1] if len(candidates) > 1 else None
        sibling_tie = False
        notes: list[str] = []

        # Promove tema de lei: específico → top-12; genérico → top-5
        if law_forced and best:
            specific_forced = {
                h.tema_id
                for h in law_hits
                if h.tema_id is not None and getattr(h, "specific", False)
            }
            top_n = 12 if specific_forced else 5
            top_ids = {c.tema_id for c in candidates[:top_n]}
            # Preferir hits específicos na ordem
            ordered_forced = sorted(
                law_forced,
                key=lambda tid: (tid not in specific_forced, -float(law_boost[self._pos[tid]] if tid in self._pos else 0)),
            )
            for lf in ordered_forced:
                if lf in self._pos and eligible[self._pos[lf]] and (lf in top_ids or lf in specific_forced):
                    bonus = 0.10 if lf in specific_forced else 0.05
                    best = self._candidate_from_pos(self._pos[lf], combined, comps, bonus=bonus)
                    notes.append(f"lei_promovida:{law_hit_name}")
                    others = [c for c in candidates if c.tema_id != best.tema_id]
                    second = others[0] if others else None
                    break

        # kNN forte
        if (
            best
            and knn_info.get("tema_id") is not None
            and float(knn_info.get("agree") or 0) >= 0.55
            and float(knn_info.get("fora_votes") or 0) < 0.35
            and float(knn_info.get("sim_max") or 0) >= 0.45
        ):
            kt = int(knn_info["tema_id"])
            if kt in self._pos and eligible[self._pos[kt]]:
                top_ids = {c.tema_id for c in candidates[:10]}
                if kt in top_ids or self.tree.same_branch(kt, best.tema_id):
                    best = self._candidate_from_pos(self._pos[kt], combined, comps, bonus=0.06)
                    notes.append("knn_folha")
                    others = [c for c in candidates if c.tema_id != best.tema_id]
                    second = others[0] if others else None

        # Assunto quase-exato: só reforça candidato já competitivo; não vence lei específica
        if assunto and str(assunto).strip():
            specific_law = {
                h.tema_id
                for h in law_hits
                if h.tema_id is not None and getattr(h, "specific", False)
            }
            for tid, s in self.lexical.assunto_fuzzy(assunto, top_k=3):
                if s >= 0.96 and tid in self._pos and eligible[self._pos[tid]]:
                    if law_forced and tid not in law_forced and not any(
                        self.tree.same_branch(tid, lf) for lf in law_forced
                    ):
                        continue
                    if specific_law and tid not in specific_law and not any(
                        self.tree.same_branch(tid, lf) for lf in specific_law
                    ):
                        continue
                    if float(combined[self._pos[tid]]) < 0.48:
                        continue
                    # Não promover assunto genérico (“Ação Penal”) sobre candidatos com margem
                    if best and float(combined[self._pos[tid]]) + 0.05 < best.score:
                        continue
                    best = self._candidate_from_pos(self._pos[tid], combined, comps, bonus=0.06 * s)
                    notes.append("assunto_exact")
                    break

        if best and second and self.tree.are_siblings(best.tema_id, second.tema_id):
            # Lei específica no best → não tratar como empate (subtema desambiguado)
            law_on_best = float(best.components.get("law", 0.0))
            law_on_second = float(second.components.get("law", 0.0))
            if law_on_best >= 0.9 and law_on_best > law_on_second + 0.2:
                sibling_tie = False
            elif abs(best.score - second.score) <= self.sibling_margin:
                sibling_tie = True
                notes.append(f"empate entre irmãos {best.tema_id} e {second.tema_id}")

        # --- Detector de fora ---
        fora_signal = 0.0
        max_tfidf = float(tfidf.max()) if len(tfidf) else 0.0
        max_proto = float(proto_n.max()) if len(proto_n) else 0.0
        max_emb = float(emb_n.max()) if len(emb_n) else 0.0
        knn_sim = float(knn_info.get("sim_max") or 0.0)
        knn_fora = float(knn_info.get("fora_votes") or 0.0)

        if is_out_of_scope_disciplina(str(disc_origem) if disc_origem is not None else None):
            fora_signal += 0.60
            notes.append("disciplina_origem fora do recorte")
        if ood_assunto_or_text(
            str(assunto) if assunto is not None else None,
            text[:500],
        ):
            fora_signal += 0.40
            notes.append("assunto/texto OOD")

        fora_signal += 0.45 * knn_fora
        # Vizinhos fracos + lexical fraco → candidato a fora (mesmo com disciplina no recorte)
        if knn_sim < 0.42 and max_tfidf < 0.10 and not law_forced:
            fora_signal += 0.25
            notes.append("baixa similaridade aos rótulos da folha")
        # Nota para a política (não inflar demais o score composto)
        if knn_fora >= 0.20:
            notes.append("vizinhos da folha sugerem fora")
        if knn_fora >= 0.45 and knn_sim >= 0.5:
            fora_signal += 0.20

        # Textos muito curtos/genéricos (folha tem fora com enunciado incompleto)
        text_len = len(text.strip())
        if text_len < 220 and max_tfidf < 0.12 and not law_forced:
            fora_signal += 0.22
            notes.append("enunciado curto/genérico")
        # Stems incompletos / "complete a frase" (comum em gold-fora da folha)
        incomplete = bool(
            re.search(
                r"complete\s+a\s+frase|preencha\s+a\s+lacuna|complete\s+o\s+estudo|"
                r"explique\s+os\s+prazos|responda\s+conforme\s+o\s+contexto|_{3,}",
                text,
                flags=re.I,
            )
        )
        if incomplete:
            fora_signal += 0.42
            notes.append("enunciado incompleto")

        if best is None:
            fora_signal += 0.40
        else:
            if best.score < 0.22:
                fora_signal += 0.25
            if max_tfidf < 0.05 and best.components.get("law", 0) == 0:
                fora_signal += 0.20
            if max_proto < 0.22 and max_emb < 0.28 and best.components.get("law", 0) == 0:
                fora_signal += 0.12
            if float(assunto_boost.max() if len(assunto_boost) else 0) < 0.70 and max_tfidf < 0.07 and not law_forced:
                fora_signal += 0.10

        # Lei forte só reduz fora se lexical confirma e enunciado não é incompleto
        if law_forced and max_tfidf >= 0.08 and not incomplete:
            fora_signal *= 0.55
        elif law_forced and incomplete:
            # Incomplete + lei no distractor/stem: não abafar o sinal de fora
            fora_signal *= 0.90

        if law_hit_name and "lei_promovida" not in ";".join(notes):
            notes.append(f"lei:{law_hit_name}")

        margin = 0.0
        if best and second:
            margin = max(0.0, best.score - second.score)
        elif best:
            margin = max(0.0, best.score * 0.15)

        # Baixa margem + lexical fraco sem lei → mais incerteza OOD
        if (
            best
            and margin < 0.04
            and max_tfidf < 0.12
            and best.components.get("law", 0) < 0.5
            and knn_fora >= 0.15
        ):
            fora_signal += 0.15
            fora_signal = min(1.0, fora_signal)

        agree_signals = 0.0
        if best:
            c = best.components
            agree_signals = (
                (1.0 if c.get("law", 0) >= 0.7 else 0.5 if c.get("law", 0) > 0 else 0.0)
                + (1.0 if c.get("knn", 0) >= 0.55 else 0.0)
                + (1.0 if c.get("assunto", 0) >= 0.85 else 0.0)
                + (1.0 if c.get("proto", 0) >= 0.50 else 0.0)
                + (1.0 if margin >= 0.08 else 0.5 if margin >= 0.04 else 0.0)
            ) / 5.0

        sibling_conflict = 1.0 if sibling_tie else 0.0
        # Gap relativo (score_gap) — menos sensível à escala do combined
        score_gap = 0.0
        if best and second and best.score > 1e-9:
            score_gap = margin / (abs(best.score) + 1e-9)

        conf_feat = {
            "raw": float(best.score if best else 0.0),
            "margin": float(margin),
            "score_gap": float(score_gap),
            "knn_agree": float(knn_info.get("agree") or 0.0),
            "knn_fora": float(knn_fora),
            "law": float(best.components.get("law", 0) if best else 0.0),
            "agree_signals": float(agree_signals),
            "fora_signal": float(min(1.0, fora_signal)),
            "sibling_conflict": float(sibling_conflict),
        }
        # Score de publicação: margem + lei + gap; penaliza knn-fora e conflito entre irmãos
        conf_feat["score"] = (
            0.22 * conf_feat["raw"]
            + 1.25 * conf_feat["margin"]
            + 0.35 * conf_feat["score_gap"]
            + 0.28 * conf_feat["knn_agree"]
            + 0.38 * conf_feat["law"]
            + 0.22 * conf_feat["agree_signals"]
            - 0.40 * conf_feat["fora_signal"]
            - 0.25 * conf_feat["knn_fora"]
            - 0.35 * conf_feat["sibling_conflict"]
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
