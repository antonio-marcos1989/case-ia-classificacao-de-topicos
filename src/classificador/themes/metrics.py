from __future__ import annotations

from typing import Iterable

from .tree import ThemeTree


def _accepted_ids(central: int | None, extras: Iterable[int] | None) -> set[int]:
    out: set[int] = set()
    if central is not None and not (isinstance(central, float) and central != central):
        try:
            out.add(int(central))
        except (TypeError, ValueError):
            pass
    if extras:
        for e in extras:
            try:
                out.add(int(e))
            except (TypeError, ValueError):
                continue
    return out


def is_tolerant_match(
    predicted: int | None,
    central: int | None,
    extras: Iterable[int] | None,
    tree: ThemeTree,
    *,
    fora: bool = False,
) -> bool:
    """Acerto tolerante: central, extra, ou nó do mesmo ramo (pai/filho).

    Publicar outro subtema do mesmo pai (irmão) = erro.
    """
    if predicted is None:
        return bool(fora)
    if fora:
        return False
    accepted = _accepted_ids(central, extras)
    if not accepted:
        return False
    pred = int(predicted)
    if pred in accepted:
        return True
    for a in accepted:
        if tree.same_branch(pred, a):
            return True
    return False


def precision_coverage(
    decisions: list[str],
    predicted: list[int | None],
    gold_central: list[int | None],
    gold_extras: list[list[int]],
    gold_fora: list[bool],
    tree: ThemeTree,
) -> dict[str, float]:
    """Precisão nas publicadas + cobertura = publicadas / total."""
    n = len(decisions)
    published = [i for i, d in enumerate(decisions) if d == "publicar"]
    coverage = len(published) / n if n else 0.0
    if not published:
        return {"precision": 1.0, "coverage": 0.0, "n_publicadas": 0, "n_total": n}
    hits = 0
    for i in published:
        if is_tolerant_match(
            predicted[i],
            gold_central[i],
            gold_extras[i],
            tree,
            fora=bool(gold_fora[i]),
        ):
            hits += 1
    return {
        "precision": hits / len(published),
        "coverage": coverage,
        "n_publicadas": len(published),
        "n_total": n,
        "n_hits": hits,
    }
