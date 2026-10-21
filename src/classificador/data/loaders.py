from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .clean import questao_texto


def _as_path(path: str | Path) -> Path:
    return Path(path)


def load_temas(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(_as_path(path))
    required = {
        "tema_id",
        "parent_id",
        "disciplina",
        "nivel",
        "titulo",
        "caminho",
        "descricao",
        "status",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"temas.csv missing columns: {sorted(missing)}")
    df["tema_id"] = df["tema_id"].astype(int)
    df["parent_id"] = pd.to_numeric(df["parent_id"], errors="coerce")
    df["nivel"] = df["nivel"].astype(int)
    return df


def load_folha(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(_as_path(path))
    required = {
        "id",
        "tema_central_id",
        "tema_central_caminho",
        "extras_ids",
        "fora_da_taxonomia",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"folha_verdade.csv missing columns: {sorted(missing)}")
    df["tema_central_id"] = pd.to_numeric(df["tema_central_id"], errors="coerce")
    df["fora_da_taxonomia"] = df["fora_da_taxonomia"].fillna(0).astype(int)

    def parse_extras(val: object) -> list[int]:
        if pd.isna(val) or str(val).strip() == "":
            return []
        return [int(x) for x in str(val).split(";") if str(x).strip()]

    df["extras_ids_list"] = df["extras_ids"].map(parse_extras)
    return df


def load_questoes(path: str | Path) -> pd.DataFrame:
    rows: list[dict] = []
    with _as_path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    required = {
        "id",
        "enunciado_html",
        "comando",
        "alternativas",
        "disciplina_origem",
        "assunto_origem",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"questoes.jsonl missing columns: {sorted(missing)}")
    df["texto"] = [
        questao_texto(r.enunciado_html, r.comando, r.alternativas)
        for r in df.itertuples(index=False)
    ]
    df["texto_curto"] = [
        questao_texto(r.enunciado_html, r.comando, r.alternativas, include_alternativas=False)
        for r in df.itertuples(index=False)
    ]
    return df
