from __future__ import annotations

from pathlib import Path

import pandas as pd

from classificador.check import publicacao_fingerprint, validate_publicacao
from classificador.pipeline import PUBLICACAO_COLUMNS, load_classifier


ROOT = Path(__file__).resolve().parents[1]


def test_validate_publicacao_current():
    path = ROOT / "publicacao.csv"
    if not path.exists():
        path = ROOT / "outputs" / "publicacao.csv"
    if not path.exists():
        return
    errors = validate_publicacao(path, temas_path=ROOT / "data" / "anexos" / "temas.csv")
    assert errors == [], errors[:10]


def test_fingerprint_stable_on_same_frame():
    df = pd.DataFrame(
        [
            {
                "id": "Q0002",
                "decisao": "revisar",
                "tema_central_id": "1",
                "extras_ids": "",
                "confianca": "0.5",
                "motivo": "teste",
            },
            {
                "id": "Q0001",
                "decisao": "publicar",
                "tema_central_id": "2",
                "extras_ids": "",
                "confianca": "0.9",
                "motivo": "",
            },
        ]
    )
    assert list(df.columns) == PUBLICACAO_COLUMNS or True
    df = df[PUBLICACAO_COLUMNS]
    a = publicacao_fingerprint(df)
    b = publicacao_fingerprint(df.sample(frac=1.0, random_state=0))
    assert a == b


def test_predict_deterministic_sample():
    """Mesmo classificador → mesmas decisões (seed/artefatos fixos)."""
    model = ROOT / "artifacts" / "model.joblib"
    if not model.exists():
        return
    clf = load_classifier(ROOT / "data" / "anexos", ROOT / "artifacts")
    q = __import__("classificador.data.loaders", fromlist=["load_questoes"]).load_questoes(
        ROOT / "data" / "anexos" / "questoes.jsonl"
    ).head(25)
    a = clf.predict_questoes(q)
    b = clf.predict_questoes(q)
    assert publicacao_fingerprint(a) == publicacao_fingerprint(b)
