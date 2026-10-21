from __future__ import annotations

import re
import unicodedata

DISCIPLINA_ALIASES: dict[str, str] = {
    "direito penal": "Penal",
    "penal": "Penal",
    "direito processual penal": "Processo Penal",
    "processo penal": "Processo Penal",
    "processual penal": "Processo Penal",
    "legislacao penal especial": "Direito Penal - Legislação Especial",
    "legislação penal especial": "Direito Penal - Legislação Especial",
    "direito penal - legislacao especial": "Direito Penal - Legislação Especial",
    "direito penal - legislação especial": "Direito Penal - Legislação Especial",
    "legislacao penal especial": "Direito Penal - Legislação Especial",
    "criminologia": "Criminologia",
}

CANONICAL_DISCIPLINAS = {
    "Penal",
    "Processo Penal",
    "Direito Penal - Legislação Especial",
    "Criminologia",
}

OUT_OF_SCOPE_DISCIPLINAS = {
    "penal militar",
    "direito penal militar",
    "processual penal militar",
    "direito processual penal militar",
    "constitucional",
    "direito constitucional",
    "tributario",
    "tributário",
    "direito tributario",
    "direito tributário",
    "ambiental",
    "direito ambiental",
    "administrativo",
    "direito administrativo",
    "civil",
    "direito civil",
    "direito civil - legislacao especial",
    "direito civil - legislação especial",
    "trabalho",
    "direito do trabalho",
    "disciplina unificada",
}

# Assunto/texto âncoras tipicamente fora do recorte (além da disciplina)
OOD_ASSUNTO_PATTERNS = [
    r"direito\s+constitucional",
    r"controle\s+de\s+constitucionalidade",
    r"mandado\s+de\s+seguran[cç]a\s+coletivo",
    r"direito\s+tribut[aá]rio",
    r"direito\s+ambiental",
    r"direito\s+administrativo",
    r"penal\s+militar",
    r"justi[cç]a\s+militar",
    r"c[oó]digo\s+penal\s+militar",
]


def _fold(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", s.lower()).strip()


def map_disciplina(origem: str | None) -> str | None:
    if not origem:
        return None
    key = _fold(origem)
    if key in DISCIPLINA_ALIASES:
        return DISCIPLINA_ALIASES[key]
    for alias, canon in DISCIPLINA_ALIASES.items():
        if alias in key or key in alias:
            return canon
    return None


def is_out_of_scope_disciplina(origem: str | None) -> bool:
    if not origem:
        return False
    key = _fold(origem)
    if not key:
        return False
    if key in OUT_OF_SCOPE_DISCIPLINAS:
        return True
    mapped = map_disciplina(origem)
    if mapped is not None:
        return False
    return any(o in key for o in OUT_OF_SCOPE_DISCIPLINAS)


def ood_assunto_or_text(assunto: str | None, text: str | None = None) -> bool:
    blob = f"{assunto or ''} {text or ''}"
    if not blob.strip():
        return False
    return any(re.search(p, blob, flags=re.I) for p in OOD_ASSUNTO_PATTERNS)
