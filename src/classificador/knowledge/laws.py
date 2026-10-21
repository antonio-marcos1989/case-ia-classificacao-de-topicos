from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class LawHit:
    name: str
    tema_id: int | None
    score: float
    reason: str
    specific: bool = False


@dataclass
class LawDictionary:
    rules: list[dict] = field(default_factory=list)

    def match(self, text: str) -> list[LawHit]:
        hits: list[LawHit] = []
        if not text:
            return hits
        for rule in self.rules:
            if not any(re.search(p, text, flags=re.I) for p in rule.get("patterns", [])):
                continue
            tema_id = rule.get("prefer")
            specific = False

            # Subtema específico (ex.: apelação → 5562)
            for sm in rule.get("subtema_map") or []:
                if any(re.search(p, text, flags=re.I) for p in sm.get("patterns", [])):
                    tema_id = sm.get("tema_id", tema_id)
                    specific = True
                    break

            # Desambiguação processual (LEP → PP)
            proc_signals = rule.get("processual_signals") or []
            if proc_signals and rule.get("processual_prefer"):
                if any(re.search(p, text, flags=re.I) for p in proc_signals):
                    tema_id = rule["processual_prefer"]
                    specific = True

            avoid = set(int(x) for x in (rule.get("avoid_tema_ids") or []))
            if tema_id is not None and int(tema_id) in avoid:
                for cand in rule.get("tema_ids") or []:
                    if int(cand) not in avoid:
                        tema_id = int(cand)
                        break

            if tema_id is None:
                continue
            hits.append(
                LawHit(
                    name=rule["name"],
                    tema_id=int(tema_id),
                    score=1.15 if specific else 1.0,
                    reason=f"lei/termo:{rule['name']}",
                    specific=specific,
                )
            )

        # Resolução de conflitos entre regras sobrepostas
        names = {h.name for h in hits}
        if "recursos" in names and "jecrim" in names:
            rec = next(h for h in hits if h.name == "recursos")
            if rec.specific or re.search(r"\bapela[cç][aã]o\b|\bRESE\b|recurso\s+em\s+sentido", text, re.I):
                hits = [h for h in hits if h.name != "jecrim"]
                names = {h.name for h in hits}

        if "prova_testemunhal" in names:
            hits = [h for h in hits if h.name not in {"juri", "inquerito"}]
            names = {h.name for h in hits}

        if "procedimento_ordinario" in names:
            # Conexão/continência + transação: tema é procedimento ordinário, não júri/JECRIM
            hits = [h for h in hits if h.name not in {"juri", "jecrim"}]
            names = {h.name for h in hits}

        return hits

    def best_tema(self, text: str) -> LawHit | None:
        hits = self.match(text)
        if not hits:
            return None
        # Prefere hit específico (subtema) e depois score
        hits = sorted(hits, key=lambda h: (h.specific, h.score), reverse=True)
        return hits[0]


def load_law_dictionary(path: str | Path | None = None) -> LawDictionary:
    if path is None:
        return LawDictionary(rules=[])
    p = Path(path)
    if not p.exists():
        return LawDictionary(rules=[])
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    rules = data.get("rules", data) if isinstance(data, dict) else data
    return LawDictionary(rules=list(rules or []))


def load_lexicon_seeds(path: str | Path | None = None) -> dict[int, list[str]]:
    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    seeds = data.get("seeds") or {}
    out: dict[int, list[str]] = {}
    for k, terms in seeds.items():
        out[int(k)] = [str(t) for t in (terms or [])]
    return out
