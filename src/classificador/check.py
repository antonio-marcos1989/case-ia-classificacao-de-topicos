from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from classificador.data.loaders import load_temas
from classificador.pipeline import PUBLICACAO_COLUMNS


REQUIRED_DECISIONS = {"publicar", "revisar", "fora"}


def publicacao_fingerprint(df: pd.DataFrame) -> str:
    """Hash estável do conteúdo de entrega (ordem por id, colunas canônicas)."""
    out = df.copy()
    for c in PUBLICACAO_COLUMNS:
        if c not in out.columns:
            raise ValueError(f"coluna ausente: {c}")
    out = out[PUBLICACAO_COLUMNS].fillna("").astype(str)
    out = out.sort_values("id").reset_index(drop=True)
    payload = out.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_publicacao(
    path: Path,
    *,
    temas_path: Path | None = None,
    expected_n: int = 659,
) -> list[str]:
    """Valida schema/regras de entrega. Retorna lista de erros (vazia = ok)."""
    errors: list[str] = []
    path = Path(path)
    if not path.exists():
        return [f"arquivo ausente: {path}"]

    df = pd.read_csv(path, dtype=str).fillna("")
    if list(df.columns) != PUBLICACAO_COLUMNS:
        errors.append(f"colunas esperadas {PUBLICACAO_COLUMNS}, obtidas {list(df.columns)}")
    if len(df) != expected_n:
        errors.append(f"esperado {expected_n} linhas, obtido {len(df)}")

    ids = set(df["id"].astype(str))
    expected_ids = {f"Q{i:04d}" for i in range(1, expected_n + 1)}
    missing = sorted(expected_ids - ids)
    extra = sorted(ids - expected_ids)
    if missing:
        errors.append(f"ids faltando (ex.): {missing[:5]}")
    if extra:
        errors.append(f"ids extras (ex.): {extra[:5]}")
    if df["id"].duplicated().any():
        errors.append("ids duplicados")

    bad_dec = set(df["decisao"]) - REQUIRED_DECISIONS
    if bad_dec:
        errors.append(f"decisoes invalidas: {sorted(bad_dec)}")

    active: set[str] = set()
    if temas_path and Path(temas_path).exists():
        temas = load_temas(temas_path)
        active = set(temas.loc[temas["status"] == "active", "tema_id"].astype(int).astype(str))

    for i, row in df.iterrows():
        dec = row["decisao"]
        tema = str(row["tema_central_id"]).strip()
        motivo = str(row["motivo"]).strip()
        try:
            conf = float(row["confianca"])
        except ValueError:
            errors.append(f"linha {i} id={row['id']}: confianca nao numerica")
            continue
        if not (0.0 <= conf <= 1.0):
            errors.append(f"{row['id']}: confianca fora de [0,1]: {conf}")

        if dec in {"publicar", "revisar"}:
            if not tema:
                errors.append(f"{row['id']}: tema_central_id obrigatorio em {dec}")
            elif active and tema not in active:
                errors.append(f"{row['id']}: tema {tema} nao esta active")
        if dec == "fora":
            if tema:
                errors.append(f"{row['id']}: fora deve ter tema_central_id vazio")
            if not motivo:
                errors.append(f"{row['id']}: motivo obrigatorio em fora")
        if dec == "revisar" and not motivo:
            errors.append(f"{row['id']}: motivo obrigatorio em revisar")

        extras = str(row["extras_ids"]).strip()
        if extras:
            for part in extras.split(";"):
                part = part.strip()
                if not part:
                    continue
                if active and part not in active:
                    errors.append(f"{row['id']}: extra {part} nao esta active")

    return errors


def write_fingerprint(path: Path, digest: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(digest + "\n", encoding="utf-8")
