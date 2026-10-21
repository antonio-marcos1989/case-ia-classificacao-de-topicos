from __future__ import annotations

import re
import unicodedata
from typing import Any

from bs4 import BeautifulSoup

_WS = re.compile(r"\s+")
_ART = re.compile(r"\bart\.?\s*", re.I)
_LEI = re.compile(r"\blei\s*n[ºo°\.]*\s*", re.I)


def clean_html(html: str | None) -> str:
    if not html:
        return ""
    soup = BeautifulSoup(html, "lxml")
    text = soup.get_text(separator=" ", strip=True)
    return normalize_text(text)


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\xa0", " ")
    text = _ART.sub("artigo ", text)
    text = _LEI.sub("lei ", text)
    text = _WS.sub(" ", text).strip()
    return text


def questao_texto(
    enunciado_html: str | None,
    comando: str | None = None,
    alternativas: dict[str, Any] | None = None,
    *,
    include_alternativas: bool = True,
) -> str:
    """Concatena enunciado (peso alto) + comando + alternativas."""
    parts: list[str] = []
    enunciado = clean_html(enunciado_html)
    if enunciado:
        parts.append(enunciado)
    cmd = normalize_text(comando or "")
    if cmd:
        parts.append(cmd)
    if include_alternativas and alternativas:
        alts = []
        for key in sorted(alternativas.keys()):
            alts.append(normalize_text(str(alternativas[key] or "")))
        if alts:
            parts.append(" ".join(alts))
    return " ".join(parts)
