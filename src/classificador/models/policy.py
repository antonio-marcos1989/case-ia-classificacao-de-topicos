from __future__ import annotations

from dataclasses import dataclass

from classificador.models.scorer import ScoreResult


@dataclass
class Decision:
    decisao: str  # publicar | revisar | fora
    tema_central_id: int | None
    extras_ids: list[int]
    confianca: float
    motivo: str


class DecisionPolicy:
    """Política publicar/revisar/fora. Empate entre irmãos → revisar."""

    def __init__(
        self,
        *,
        tau_publicar: float = 0.85,
        tau_fora: float = 0.50,
        min_raw_score: float = 0.12,
    ):
        self.tau_publicar = float(tau_publicar)
        self.tau_fora = float(tau_fora)
        self.min_raw_score = float(min_raw_score)

    def _extras(self, result: ScoreResult) -> list[int]:
        best = result.best
        if not best or not result.second or result.sibling_tie:
            return []
        sec = result.second
        # Só extras baratos e confiáveis: score próximo e mesma disciplina / lei
        if sec.score < 0.78 * best.score:
            return []
        if sec.components.get("law", 0) >= 0.7 or sec.components.get("assunto", 0) >= 0.9:
            return [sec.tema_id]
        if abs(sec.score - best.score) <= 0.12 and sec.components.get("disc", 0) >= 0.5:
            return [sec.tema_id]
        return []

    def _motivo_fora(self, result: ScoreResult) -> str:
        preferred = [
            n
            for n in result.notes
            if "fora do recorte" in n or n.startswith("assunto/texto OOD") or "fora" in n.lower()
        ]
        if preferred:
            return "; ".join(preferred)[:200]
        if result.notes:
            return "; ".join(result.notes)[:200]
        return "baixa aderência à taxonomia"

    def _motivo_revisar(self, result: ScoreResult, confianca: float) -> str:
        parts: list[str] = []
        if result.sibling_tie:
            parts.append(next((n for n in result.notes if "irmão" in n), "dúvida entre subtemas irmãos"))
        elif confianca < self.tau_publicar:
            parts.append(f"confiança {confianca:.2f} abaixo do limiar {self.tau_publicar:.2f}")
        if result.best and result.best.score < self.min_raw_score:
            parts.append("score lexical/embedding baixo")
        # Sinais úteis para o especialista
        for n in result.notes:
            if n.startswith("lei") or n.startswith("knn") or n.startswith("assunto") or "OOD" in n:
                if n not in parts:
                    parts.append(n)
        if result.second and result.best:
            parts.append(f"sugestão {result.best.tema_id}; alt {result.second.tema_id}")
        if not parts:
            parts.append("incerteza residual")
        return "; ".join(parts)[:200]

    def decide(self, result: ScoreResult, confianca: float) -> Decision:
        best = result.best
        if best is None:
            return Decision("fora", None, [], float(confianca), "sem candidato de tema")

        tema = best.tema_id
        extras = self._extras(result)

        # Fora com prioridade quando sinal OOD claro
        strong_ood = any(
            "fora do recorte" in n or "OOD" in n or "multi-topico" in n or "stub LEP" in n
            for n in result.notes
        )
        incomplete = any("incompleto" in n for n in result.notes)
        multi_topic = any("multi-topico" in n for n in result.notes)
        knn_fora_early = float((result.confidence_features or {}).get("knn_fora", 0.0))
        ood_note = strong_ood or incomplete
        if result.fora_signal >= self.tau_fora and (
            strong_ood or best.score < 0.40 or confianca < self.tau_publicar
        ):
            return Decision("fora", None, [], float(confianca), self._motivo_fora(result))

        # OOD/multi-tópico/stub com sinal ≥0.38 → fora
        if strong_ood and result.fora_signal >= 0.38:
            return Decision("fora", None, [], float(confianca), self._motivo_fora(result))

        if multi_topic and result.fora_signal >= 0.30:
            return Decision("fora", None, [], float(confianca), self._motivo_fora(result))

        # Incomplete sozinho NÃO força fora (muitas in-taxonomy usam "Complete a frase");
        # só com knn-fora ou sinal alto residual.
        if incomplete and result.fora_signal >= 0.45 and knn_fora_early >= 0.20:
            return Decision("fora", None, [], float(confianca), self._motivo_fora(result))

        # knn-fora forte + fora_signal moderado → fora (não só revisar)
        if result.fora_signal >= 0.38 and knn_fora_early >= 0.22 and confianca < self.tau_publicar:
            return Decision("fora", None, [], float(confianca), self._motivo_fora(result))

        if result.fora_signal >= 0.55 and strong_ood:
            return Decision("fora", None, [], float(confianca), self._motivo_fora(result))

        if result.fora_signal >= 0.70 and confianca < self.tau_publicar:
            return Decision("fora", None, [], float(confianca), self._motivo_fora(result))

        if best.score < self.min_raw_score and result.fora_signal >= 0.40:
            return Decision("fora", None, [], float(confianca), self._motivo_fora(result))

        if result.sibling_tie:
            if knn_fora_early >= 0.12 and result.fora_signal >= 0.25:
                return Decision("fora", None, [], float(confianca), self._motivo_fora(result))
            return Decision(
                "revisar", tema, extras, float(confianca), self._motivo_revisar(result, confianca)
            )

        cf = result.confidence_features or {}
        margin = float(cf.get("margin", 0.0))
        law = float(cf.get("law", 0.0))
        knn = float(cf.get("knn_agree", 0.0))
        agree = float(cf.get("agree_signals", 0.0))
        knn_fora = float(cf.get("knn_fora", 0.0))
        score_gap = float(cf.get("score_gap", 0.0))
        incomplete = any("incompleto" in n for n in result.notes)
        # Override knn-fora:
        #  (a) vizinhos votam fora com força (kf≥0.25) + lei/margem claros
        #  (b) kf leve (0.12–0.22) + lei específica + margem + enunciado completo
        # Override agressivo (kf≥0.25) exige margem alta — evita publicar gold-fora
        # com vizinhos fora + lei genérica (ex. inquérito em item fora da taxonomia).
        law_overrides_fora = (
            law >= 0.7
            and result.fora_signal < 0.30
            and not incomplete
            and (
                (knn_fora >= 0.25 and margin >= 0.35 and knn >= 0.20)
                or (
                    0.12 <= knn_fora < 0.22
                    and margin >= 0.25
                    and (knn >= 0.15 or score_gap >= 0.12)
                )
            )
        )
        knn_fora_block = knn_fora >= 0.12 and not law_overrides_fora
        strong = result.fora_signal < 0.25 and not knn_fora_block and (
            (margin >= 0.20 and law >= 0.7 and knn >= 0.18)
            or (margin >= 0.16 and knn >= 0.45)
            or (margin >= 0.18 and score_gap >= 0.10 and agree >= 0.30)
        )
        near_tau = confianca >= (self.tau_publicar - 0.12)
        pass_tau = confianca >= self.tau_publicar

        if knn_fora_block:
            return Decision(
                "revisar",
                tema,
                extras,
                float(confianca),
                "vizinhos da folha sugerem fora",
            )

        if best.score >= self.min_raw_score and result.fora_signal < 0.50:
            if pass_tau and margin >= 0.04 and (law >= 0.7 or knn >= 0.35 or margin >= 0.08 or score_gap >= 0.10):
                return Decision("publicar", tema, extras, float(confianca), "")
            if strong and near_tau:
                return Decision("publicar", tema, extras, float(confianca), "")

        return Decision(
            "revisar", tema, extras, float(confianca), self._motivo_revisar(result, confianca)
        )
