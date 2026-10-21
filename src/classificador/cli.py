from __future__ import annotations

import argparse
import sys
from pathlib import Path

from classificador.pipeline import (
    default_paths,
    load_classifier,
    train_classifier,
    write_previsao,
    write_publicacao,
)
from classificador.data.loaders import load_questoes


def _repo_root() -> Path:
    # src/classificador/cli.py → repo root
    return Path(__file__).resolve().parents[2]


def cmd_build(args: argparse.Namespace) -> int:
    root = Path(args.root) if args.root else _repo_root()
    paths = default_paths(root)
    anexos = Path(args.anexos) if args.anexos else paths["anexos"]
    artifacts = Path(args.artifacts) if args.artifacts else paths["artifacts"]
    clf = train_classifier(
        anexos,
        artifacts,
        target_precision=args.target_precision,
        n_splits=args.folds,
    )
    print("build ok")
    print("metrics:", {k: clf.metrics.get(k) for k in ("precision", "coverage", "tau", "n_publicadas", "embeddings_loaded")})
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    root = Path(args.root) if args.root else _repo_root()
    paths = default_paths(root)
    anexos = Path(args.anexos) if args.anexos else paths["anexos"]
    artifacts = Path(args.artifacts) if args.artifacts else paths["artifacts"]
    out_dir = Path(args.output_dir) if args.output_dir else paths["outputs"]
    model_path = artifacts / "model.joblib"
    if not model_path.exists():
        print("artifacts/model.joblib ausente — rodando build primeiro...", file=sys.stderr)
        train_classifier(anexos, artifacts, target_precision=args.target_precision)

    clf = load_classifier(anexos, artifacts)
    questoes = load_questoes(anexos / "questoes.jsonl")
    pub = clf.predict_questoes(questoes)
    if len(pub) != 659:
        print(f"aviso: esperado 659 linhas, obtido {len(pub)}", file=sys.stderr)

    publicacao_path = out_dir / "publicacao.csv"
    previsao_path = out_dir / "previsao.txt"
    # also copy to repo root for entrega
    write_publicacao(pub, publicacao_path)
    write_previsao(clf.metrics, previsao_path)
    if args.also_root:
        write_publicacao(pub, root / "publicacao.csv")
        write_previsao(clf.metrics, root / "previsao.txt")

    counts = pub["decisao"].value_counts().to_dict()
    print(f"wrote {publicacao_path}")
    print(f"wrote {previsao_path}")
    print("decisoes:", counts)
    print("tau:", clf.policy.tau_publicar)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="classificador", description="Classificação de tópicos (offline)")
    parser.add_argument("--root", default=None, help="Raiz do repositório")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build", help="Treina calibrador/limiar e grava artifacts/model.joblib")
    p_build.add_argument("--anexos", default=None)
    p_build.add_argument("--artifacts", default=None)
    p_build.add_argument("--target-precision", type=float, default=0.97)
    p_build.add_argument("--folds", type=int, default=5)
    p_build.set_defaults(func=cmd_build)

    p_pred = sub.add_parser("predict", help="Gera publicacao.csv e previsao.txt offline")
    p_pred.add_argument("--anexos", default=None)
    p_pred.add_argument("--artifacts", default=None)
    p_pred.add_argument("--output-dir", default=None)
    p_pred.add_argument("--target-precision", type=float, default=0.97)
    p_pred.add_argument("--also-root", action="store_true", default=True)
    p_pred.add_argument("--no-also-root", action="store_false", dest="also_root")
    p_pred.set_defaults(func=cmd_predict)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
