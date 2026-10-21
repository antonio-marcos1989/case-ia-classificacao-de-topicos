#!/usr/bin/env python3
"""Gera cache de embeddings locais (Fase 3). Requer sentence-transformers.

Predict NÃO chama este script nem a rede: só lê artifacts/*.npy se existirem.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from classificador.data.loaders import load_questoes, load_temas
from classificador.features.embeddings import build_and_cache_embeddings, DEFAULT_MODEL_NAME
from classificador.pipeline import build_lexicon_from_folha
from classificador.data.loaders import load_folha
from classificador.themes.tree import ThemeTree


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anexos", type=Path, default=Path("data/anexos"))
    ap.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    ap.add_argument("--model", default=DEFAULT_MODEL_NAME)
    args = ap.parse_args()

    temas = load_temas(args.anexos / "temas.csv")
    questoes = load_questoes(args.anexos / "questoes.jsonl")
    folha = load_folha(args.anexos / "folha_verdade.csv")
    tree = ThemeTree(temas)
    lexicon = build_lexicon_from_folha(folha, tree)

    tema_ids = tree.active(include_roots=False)
    textos_temas = [(tid, tree.theme_document(tid, lexicon)) for tid in tema_ids]
    textos_questoes = [(str(r.id), str(r.texto)) for r in questoes.itertuples(index=False)]

    idx = build_and_cache_embeddings(
        textos_questoes, textos_temas, args.artifacts, model_name=args.model
    )
    print(f"cached {len(idx.questao_ids)} questoes + {len(idx.tema_ids)} temas → {args.artifacts}")


if __name__ == "__main__":
    main()
