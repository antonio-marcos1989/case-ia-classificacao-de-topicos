#!/usr/bin/env python3
"""EDA leve (Fase 1) — estatísticas dos anexos sem LLM."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from classificador.data.loaders import load_folha, load_questoes, load_temas
from classificador.knowledge.aliases import map_disciplina, is_out_of_scope_disciplina
from classificador.themes.tree import ThemeTree


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anexos", type=Path, default=Path("data/anexos"))
    args = ap.parse_args()

    temas = load_temas(args.anexos / "temas.csv")
    questoes = load_questoes(args.anexos / "questoes.jsonl")
    folha = load_folha(args.anexos / "folha_verdade.csv")
    tree = ThemeTree(temas)

    print("=== Temas ===")
    print(temas.groupby(["disciplina", "nivel"]).size().unstack(fill_value=0))
    print("inactive:", temas.loc[temas.status != "active", ["tema_id", "titulo"]].to_dict("records"))

    print("\n=== Questões ===")
    print("n=", len(questoes))
    print("disciplina_origem top:")
    print(questoes["disciplina_origem"].value_counts().head(15))
    mapped = questoes["disciplina_origem"].map(map_disciplina)
    print("mapped:", mapped.value_counts(dropna=False).head(10))
    oos = questoes["disciplina_origem"].map(is_out_of_scope_disciplina).sum()
    print("out_of_scope_disciplina approx:", int(oos))
    print("texto len median:", questoes["texto"].str.len().median())
    print("comando vazio:", int((questoes["comando"].fillna("").str.strip() == "").sum()))

    print("\n=== Folha ===")
    print("fora:", int(folha.fora_da_taxonomia.sum()))
    labeled = folha[folha.fora_da_taxonomia == 0]
    niveles = []
    for tid in labeled["tema_central_id"].dropna().astype(int):
        niveles.append(tree.nivel(tid))
    print("nivel central:", pd.Series(niveles).value_counts().to_dict())
    print("com extras:", int((folha["extras_ids"].fillna("").astype(str).str.len() > 0).sum()))

    # Ambiguidade LEP / Maria
    print("\n=== Duplicatas de título entre disciplinas ===")
    titles = temas[temas.status == "active"].groupby("titulo")["disciplina"].nunique()
    dups = titles[titles > 1]
    print("títulos em >1 disciplina:", len(dups))
    for t in ["Lei Maria da Penha", "Lei de Execução Penal", "Tráfico de Drogas"]:
        print(t, temas.loc[temas.titulo == t, ["tema_id", "disciplina", "nivel"]].to_dict("records"))


if __name__ == "__main__":
    main()
