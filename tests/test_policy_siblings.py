from __future__ import annotations

from classificador.models.policy import DecisionPolicy
from classificador.models.scorer import Candidate, ScoreResult


def test_sibling_tie_goes_to_revisar():
    policy = DecisionPolicy(tau_publicar=0.5)
    res = ScoreResult(
        candidates=[],
        best=Candidate(11, 0.8, {}),
        second=Candidate(12, 0.79, {}),
        sibling_tie=True,
        fora_signal=0.0,
        law_hit=None,
        notes=["empate entre irmãos 11 e 12"],
        confidence_features={},
    )
    d = policy.decide(res, confianca=0.99)
    assert d.decisao == "revisar"
    assert "irmão" in d.motivo or "empate" in d.motivo
