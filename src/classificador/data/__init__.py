from .loaders import load_folha, load_questoes, load_temas
from .clean import clean_html, questao_texto

__all__ = [
    "load_folha",
    "load_questoes",
    "load_temas",
    "clean_html",
    "questao_texto",
]
