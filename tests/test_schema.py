from __future__ import annotations

from pathlib import Path

import pandas as pd

from classificador.data.loaders import load_questoes, load_temas, load_folha
from classificador.knowledge.laws import LawDictionary
from classificador.pipeline import PUBLICACAO_COLUMNS


ROOT = Path(__file__).resolve().parents[1]
ANEXOS = ROOT / "data" / "anexos"


def test_anexos_counts():
    assert len(load_questoes(ANEXOS / "questoes.jsonl")) == 659
    temas = load_temas(ANEXOS / "temas.csv")
    assert len(temas) == 377
    assert (temas.status == "active").sum() == 375
    assert len(load_folha(ANEXOS / "folha_verdade.csv")) == 100


def test_law_maria():
    from classificador.knowledge.laws import load_law_dictionary

    laws = load_law_dictionary(ROOT / "artifacts" / "laws_v0.yaml")
    hits = laws.match("Segundo a Lei 11.340/2006 (Maria da Penha), medida protetiva")
    assert any(h.name == "maria_da_penha" for h in hits)


def test_publicacao_schema_if_present():
    path = ROOT / "publicacao.csv"
    if not path.exists():
        path = ROOT / "outputs" / "publicacao.csv"
    if not path.exists():
        return
    df = pd.read_csv(path, dtype=str).fillna("")
    assert list(df.columns) == PUBLICACAO_COLUMNS
    assert len(df) == 659
    assert set(df["decisao"]) <= {"publicar", "revisar", "fora"}
    for _, row in df.iterrows():
        if row["decisao"] in {"publicar", "revisar"}:
            assert row["tema_central_id"] != ""
        if row["decisao"] == "fora":
            assert row["tema_central_id"] == ""
            assert row["motivo"] != ""
        if row["decisao"] == "revisar":
            assert row["motivo"] != ""
        conf = float(row["confianca"])
        assert 0.0 <= conf <= 1.0
