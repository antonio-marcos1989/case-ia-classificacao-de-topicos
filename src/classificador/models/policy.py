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
    """Política publicar/revisar/fora com limiares calibrados.

    Empate entre irmãos → revisar (não chutar; não subir ao pai).
    """

    def __init__(
        self,
        *,
        tau_publicar: float = 0.85,
        tau_fora: float = 0.55,
        min_raw_score: float = 0.15,
    ):
        self.tau_publicar = float(tau_publicar)
        self.tau_fora = float(tau_fora)
        self.min_raw_score = float(min_raw_score)

    def decide(self, result: ScoreResult, confianca: float) -> Decision:
        best = result.best
        if best is None:
            return Decision("fora", None, [], float(confianca), "sem candidato de tema")

        tema = best.tema_id
        extras: list[int] = []
        # extras leves: 2º candidato se score alto e não irmão
        if result.second and result.second.score >= 0.7 * best.score:
            if not result.sibling_tie:
                extras = [result.second.tema_id]

        # Fora
        if result.fora_signal >= self.tau_fora and (
            best.score < 0.35 or confianca < self.tau_publicar
        ):
            motivo = "; ".join(result.notes) if result.notes else "baixa aderência à taxonomia"
            return Decision("fora", None, [], float(confianca), motivo[:200])
        if best.score < self.min_raw_score and result.fora_signal >= 0.35:
            motivo = "; ".join(result.notes) if result.notes else "score insuficiente e sinal de fora"
            return Decision("fora", None, [], float(confianca), motivo[:200])

        # Empate entre irmãos
        if result.sibling_tie:
            motivo = result.notes[0] if result.notes else "dúvida entre subtemas irmãos"
            return Decision("revisar", tema, extras, float(confianca), motivo[:200])

        # Publicar se confiança >= tau
        if confianca >= self.tau_publicar and best.score >= self.min_raw_score:
            return Decision("publicar", tema, extras, float(confianca), "")

        # Caso contrário revisar
        if best.score < self.min_raw_score:
            motivo = "score lexical/embedding baixo"
        elif confianca < self.tau_publicar:
            motivo = f"confiança {confianca:.2f} abaixo do limiar {self.tau_publicar:.2f}"
        else:
            motivo = "incerteza residual"
        if result.notes:
            motivo = motivo + "; " + "; ".join(result.notes)
        return Decision("revisar", tema, extras, float(confianca), motivo[:200])
