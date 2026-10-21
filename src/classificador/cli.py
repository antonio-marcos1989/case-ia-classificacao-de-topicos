from __future__ import annotations

import argparse
import sys
from pathlib import Path

from classificador.check import publicacao_fingerprint, validate_publicacao, write_fingerprint
from classificador.pipeline import (
    default_paths,
    load_classifier,
    score_shift_sanity,
    train_classifier,
    write_previsao,
    write_publicacao,
)
from classificador.data.loaders import load_folha, load_questoes


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

    metrics = dict(clf.metrics or {})
    metrics["score_shift"] = score_shift_sanity(pub)
    # Recall fora na folha completa (sanidade; hold-out já está em metrics)
    try:
        folha = load_folha(anexos / "folha_verdade.csv")
        gold_fora = set(folha.loc[folha["fora_da_taxonomia"].astype(int) == 1, "id"].astype(str))
        pred_fora = set(pub.loc[pub["decisao"] == "fora", "id"].astype(str))
        hit = len(gold_fora & pred_fora)
        metrics["fora_recall_folha"] = float(hit / len(gold_fora)) if gold_fora else None
        metrics["fora_hit_folha"] = int(hit)
        metrics["fora_gold_folha"] = int(len(gold_fora))
    except Exception as exc:  # noqa: BLE001
        print(f"aviso: fora_recall_folha falhou: {exc}", file=sys.stderr)

    publicacao_path = out_dir / "publicacao.csv"
    previsao_path = out_dir / "previsao.txt"
    # also copy to repo root for entrega
    write_publicacao(pub, publicacao_path)
    write_previsao(metrics, previsao_path)
    if args.also_root:
        write_publicacao(pub, root / "publicacao.csv")
        write_previsao(metrics, root / "previsao.txt")

    counts = pub["decisao"].value_counts().to_dict()
    digest = publicacao_fingerprint(pub)
    write_fingerprint(artifacts / "publicacao.sha256", digest)
    print(f"wrote {publicacao_path}")
    print(f"wrote {previsao_path}")
    print("decisoes:", counts)
    print("tau:", clf.policy.tau_publicar)
    print("shift:", metrics.get("score_shift", {}).get("delta_conf_mean"), "fora_folha:", metrics.get("fora_recall_folha"))
    print("sha256:", digest)
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    """Schema + temas ativos + fingerprint (Fase 6)."""
    root = Path(args.root) if args.root else _repo_root()
    paths = default_paths(root)
    anexos = Path(args.anexos) if args.anexos else paths["anexos"]
    artifacts = Path(args.artifacts) if args.artifacts else paths["artifacts"]
    pub_path = Path(args.publicacao) if args.publicacao else root / "publicacao.csv"
    if not pub_path.exists():
        alt = paths["outputs"] / "publicacao.csv"
        if alt.exists():
            pub_path = alt
    errors = validate_publicacao(pub_path, temas_path=anexos / "temas.csv")
    if errors:
        print("CHECK FAIL", file=sys.stderr)
        for e in errors[:40]:
            print(" -", e, file=sys.stderr)
        if len(errors) > 40:
            print(f" ... +{len(errors) - 40} erros", file=sys.stderr)
        return 1

    df = __import__("pandas").read_csv(pub_path, dtype=str).fillna("")
    digest = publicacao_fingerprint(df)
    fp_path = artifacts / "publicacao.sha256"
    print("CHECK OK")
    print("rows:", len(df))
    print("sha256:", digest)
    print("decisoes:", df["decisao"].value_counts().to_dict())
    if fp_path.exists():
        expected = fp_path.read_text(encoding="utf-8").strip().split()[0]
        if expected != digest:
            print(f"AVISO: fingerprint diverge de {fp_path}", file=sys.stderr)
            print(f"  esperado: {expected}", file=sys.stderr)
            print(f"  atual:    {digest}", file=sys.stderr)
            if args.strict_hash:
                return 2
        else:
            print("fingerprint: match", fp_path)
    elif args.strict_hash:
        print(f"CHECK FAIL: {fp_path} ausente", file=sys.stderr)
        return 2
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

    p_check = sub.add_parser("check", help="Valida schema/temas ativos/fingerprint de publicacao.csv")
    p_check.add_argument("--anexos", default=None)
    p_check.add_argument("--artifacts", default=None)
    p_check.add_argument("--publicacao", default=None)
    p_check.add_argument(
        "--strict-hash",
        action="store_true",
        help="Falha se artifacts/publicacao.sha256 ausente ou divergente",
    )
    p_check.set_defaults(func=cmd_check)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
