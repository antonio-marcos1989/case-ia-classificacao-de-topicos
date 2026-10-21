from __future__ import annotations

from collections import defaultdict

import pandas as pd


class ThemeTree:
    """Árvore de temas + helpers de parentesco (pai/filho/irmãos)."""

    def __init__(self, temas: pd.DataFrame):
        self.df = temas.copy()
        self.by_id: dict[int, dict] = {
            int(r.tema_id): r._asdict() for r in self.df.itertuples(index=False)
        }
        self.children: dict[int | None, list[int]] = defaultdict(list)
        for tid, row in self.by_id.items():
            parent = row["parent_id"]
            key = None if pd.isna(parent) else int(parent)
            self.children[key].append(tid)

        self.ancestors: dict[int, list[int]] = {}
        for tid in self.by_id:
            chain: list[int] = []
            cur = self.by_id[tid]["parent_id"]
            seen: set[int] = set()
            while not pd.isna(cur):
                cid = int(cur)
                if cid in seen:
                    break
                seen.add(cid)
                chain.append(cid)
                cur = self.by_id.get(cid, {}).get("parent_id")
            self.ancestors[tid] = chain

    def active(self, include_roots: bool = False) -> list[int]:
        out = []
        for tid, row in self.by_id.items():
            if row["status"] != "active":
                continue
            if not include_roots and int(row["nivel"]) == 0:
                continue
            out.append(tid)
        return out

    def parent(self, tema_id: int) -> int | None:
        p = self.by_id[tema_id]["parent_id"]
        if pd.isna(p):
            return None
        return int(p)

    def siblings(self, tema_id: int) -> list[int]:
        p = self.parent(tema_id)
        return [c for c in self.children.get(p, []) if c != tema_id]

    def are_siblings(self, a: int, b: int) -> bool:
        if a == b:
            return False
        pa, pb = self.parent(a), self.parent(b)
        return pa is not None and pa == pb

    def same_branch(self, a: int, b: int) -> bool:
        """Verdadeiro se a é ancestral de b, b de a, ou a==b."""
        if a == b:
            return True
        if a in self.ancestors.get(b, []):
            return True
        if b in self.ancestors.get(a, []):
            return True
        return False

    def path(self, tema_id: int) -> str:
        return str(self.by_id[tema_id].get("caminho") or "")

    def titulo(self, tema_id: int) -> str:
        return str(self.by_id[tema_id].get("titulo") or "")

    def disciplina(self, tema_id: int) -> str:
        return str(self.by_id[tema_id].get("disciplina") or "")

    def nivel(self, tema_id: int) -> int:
        return int(self.by_id[tema_id]["nivel"])

    def theme_document(self, tema_id: int, lexicon: dict[int, list[str]] | None = None) -> str:
        row = self.by_id[tema_id]
        parts = [str(row.get("titulo") or ""), str(row.get("caminho") or "")]
        desc = str(row.get("descricao") or "")
        if desc and not desc.startswith("Subtema de"):
            parts.append(desc)
        if lexicon and tema_id in lexicon:
            parts.extend(lexicon[tema_id])
        return " ".join(p for p in parts if p)
