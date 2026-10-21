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
    "trabalho",
    "direito do trabalho",
}


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
    if key in OUT_OF_SCOPE_DISCIPLINAS:
        return True
    mapped = map_disciplina(origem)
    return mapped is None and any(o in key for o in OUT_OF_SCOPE_DISCIPLINAS)
