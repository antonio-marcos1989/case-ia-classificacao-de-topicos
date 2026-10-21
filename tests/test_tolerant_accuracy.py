from __future__ import annotations

import pandas as pd

from classificador.themes.tree import ThemeTree
from classificador.themes.metrics import is_tolerant_match


def _mini_tree() -> ThemeTree:
    df = pd.DataFrame(
        [
            {"tema_id": 1, "parent_id": None, "disciplina": "Penal", "nivel": 0, "titulo": "Penal", "caminho": "Penal", "descricao": "", "status": "active"},
            {"tema_id": 10, "parent_id": 1, "disciplina": "Penal", "nivel": 1, "titulo": "Pai", "caminho": "Penal > Pai", "descricao": "", "status": "active"},
            {"tema_id": 11, "parent_id": 10, "disciplina": "Penal", "nivel": 2, "titulo": "FilhoA", "caminho": "Penal > Pai > FilhoA", "descricao": "", "status": "active"},
            {"tema_id": 12, "parent_id": 10, "disciplina": "Penal", "nivel": 2, "titulo": "FilhoB", "caminho": "Penal > Pai > FilhoB", "descricao": "", "status": "active"},
            {"tema_id": 20, "parent_id": 1, "disciplina": "Penal", "nivel": 1, "titulo": "Outro", "caminho": "Penal > Outro", "descricao": "", "status": "active"},
        ]
    )
    return ThemeTree(df)


def test_central_and_parent_ok():
    tree = _mini_tree()
    assert is_tolerant_match(11, 11, [], tree)
    assert is_tolerant_match(10, 11, [], tree)  # pai
    assert is_tolerant_match(11, 10, [], tree)  # filho do central


def test_sibling_is_error():
    tree = _mini_tree()
    assert not is_tolerant_match(12, 11, [], tree)


def test_extra_ok():
    tree = _mini_tree()
    assert is_tolerant_match(20, 11, [20], tree)


def test_fora():
    tree = _mini_tree()
    assert is_tolerant_match(None, None, [], tree, fora=True)
    assert not is_tolerant_match(11, None, [], tree, fora=True)
