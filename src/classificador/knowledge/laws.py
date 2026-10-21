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


# Corta só em marcadores de alternativa HTML/letra — NÃO em "Assinale"
# (muitos enunciados citam a lei nas assertivas I/II após "assinale").
_STEM_CUTS = [
    r"<span\b",
    r"<br\s*/?>\s*[A-Ea-e][\).\]]",
    r"\n\s*[A-Ea-e][\).\]]\s",
    r"\bComplete\s+a\s+frase\b",
]


def question_stem(text: str, *, max_chars: int = 700) -> str:
    """Enunciado sem alternativas HTML/letra (reduz falso positivo em distractors)."""
    if not text:
        return ""
    cut = len(text)
    for pat in _STEM_CUTS:
        m = re.search(pat, text, flags=re.I)
        if m and m.start() > 80:
            cut = min(cut, m.start())
    stem = text[:cut].strip()
    if len(stem) < 80:
        stem = text[:max_chars]
    return stem[:max_chars]


@dataclass
class LawDictionary:
    rules: list[dict] = field(default_factory=list)

    def match(self, text: str) -> list[LawHit]:
        hits: list[LawHit] = []
        if not text:
            return hits
        # Casamento preferencial no stem; fallback no texto completo se stem vazio
        stem = question_stem(text)
        blob = stem if len(stem) >= 40 else text

        for rule in self.rules:
            if not any(re.search(p, blob, flags=re.I) for p in rule.get("patterns", [])):
                continue
            tema_id = rule.get("prefer")
            specific = False

            for sm in rule.get("subtema_map") or []:
                if any(re.search(p, blob, flags=re.I) for p in sm.get("patterns", [])):
                    tema_id = sm.get("tema_id", tema_id)
                    specific = True
                    break

            proc_signals = rule.get("processual_signals") or []
            if proc_signals and rule.get("processual_prefer"):
                if any(re.search(p, blob, flags=re.I) for p in proc_signals):
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

        names = {h.name for h in hits}
        if "recursos" in names and "jecrim" in names:
            rec = next(h for h in hits if h.name == "recursos")
            if rec.specific or re.search(r"\bapela[cç][aã]o\b|\bRESE\b|recurso\s+em\s+sentido", blob, re.I):
                hits = [h for h in hits if h.name != "jecrim"]
                names = {h.name for h in hits}

        # Prova testemunhal / cadeia de custódia vencem leis citadas só em distractors
        if "prova_testemunhal" in names:
            hits = [h for h in hits if h.name not in {"juri", "inquerito", "drogas", "jecrim"}]
            names = {h.name for h in hits}

        if "cadeia_custodia" in names:
            hits = [h for h in hits if h.name not in {"juri", "inquerito", "drogas"}]
            names = {h.name for h in hits}

        if "procedimento_ordinario" in names and "jecrim" in names:
            # Competência dos Juizados + conexão: tema JECRIM, não ordinário
            if re.search(
                r"juizados?\s+especiais?\s+criminais?|compet[eê]ncia\s+dos\s+juizados",
                blob,
                re.I,
            ):
                hits = [h for h in hits if h.name != "procedimento_ordinario"]
            else:
                # Reunião de processos / juízo comum → ordinário (júri/JECRIM só contextuais)
                hits = [h for h in hits if h.name not in {"jecrim", "juri"}]
            names = {h.name for h in hits}
        elif "procedimento_ordinario" in names:
            hits = [h for h in hits if h.name not in {"juri", "jecrim"}]
            names = {h.name for h in hits}

        if "antiterrorismo" in names:
            hits = [h for h in hits if h.name not in {"inquerito", "jecrim"}]
            names = {h.name for h in hits}

        # Arquivamento de inquérito vence "prescrição" citada só como alternativa
        if "inquerito" in names and "prescricao" in names:
            if re.search(r"arquiv|inqu[eé]rito\s+policial", blob, re.I):
                hits = [h for h in hits if h.name != "prescricao"]
                names = {h.name for h in hits}

        # Hit específico vence genérico do mesmo cluster
        if len(hits) > 1:
            specific_hits = [h for h in hits if h.specific]
            if specific_hits and len(specific_hits) < len(hits):
                # Mantém específicos + hits de nomes distintos sem genérico conflitante
                keep_names = {h.name for h in specific_hits}
                hits = [h for h in hits if h.specific or h.name not in keep_names]

        return hits

    def best_tema(self, text: str) -> LawHit | None:
        hits = self.match(text)
        if not hits:
            return None
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
