from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_LAW_RULES: list[dict] = []  # prefer YAML artifact


@dataclass
class LawHit:
    name: str
    tema_id: int | None
    score: float
    reason: str


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
            proc_signals = rule.get("processual_signals") or []
            if proc_signals and rule.get("processual_prefer"):
                if any(re.search(p, text, flags=re.I) for p in proc_signals):
                    tema_id = rule["processual_prefer"]
            if tema_id is None:
                continue
            hits.append(
                LawHit(
                    name=rule["name"],
                    tema_id=int(tema_id),
                    score=1.0,
                    reason=f"lei/termo:{rule['name']}",
                )
            )
        return hits

    def best_tema(self, text: str) -> LawHit | None:
        hits = self.match(text)
        return hits[0] if hits else None


def load_law_dictionary(path: str | Path | None = None) -> LawDictionary:
    if path is None:
        return LawDictionary(rules=[])
    p = Path(path)
    if not p.exists():
        return LawDictionary(rules=[])
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    rules = data.get("rules", data) if isinstance(data, dict) else data
    return LawDictionary(rules=list(rules or []))
