"""分子图数据模型：原子、键、分子。

键级约定：
  1 单键, 2 双键, 3 三键, 4 芳香键（仅出现在两个芳香原子之间）
原子的 aromatic 标记与芳香键一致，由 parser/aromatic 模块维护。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .elements import element_of, atomic_number


# 键级 -> SMILES 符号
BOND_SYMBOL = {1: "-", 2: "=", 3: "#", 4: ":"}
SYMBOL_BOND = {"-": 1, "=": 2, "#": 3, ":": 4, "/": 1, "\\": 1}


@dataclass
class Atom:
    index: int                      # 在分子中的下标
    symbol: str                     # 原始符号（可能是小写芳香符号）
    charge: int = 0
    isotope: Optional[int] = None
    explicit_h: int = 0             # 方括号内显式氢数量
    bracket: bool = False           # 是否写了方括号
    aromatic: bool = False          # 是否为芳香原子
    # 通配/查询属性（仅查询片段使用）
    wildcard: bool = False          # * 任意原子
    aromatic_query: bool = False    # a：任意芳香原子
    aliphatic_query: bool = False   # A：任意脂肪原子
    _label_cache: Optional[str] = None  # label() 结果缓存

    @property
    def element(self) -> str:
        if self.wildcard or self.aromatic_query or self.aliphatic_query:
            return "*"
        return element_of(self.symbol)

    @property
    def number(self) -> int:
        return atomic_number(self.symbol)

    def label(self) -> str:
        """用于指纹/规范化的原子标签（解析后属性不变，可缓存）。"""
        if self._label_cache is not None:
            return self._label_cache
        parts = [self.element]
        if self.isotope is not None:
            parts.insert(0, str(self.isotope))
        if self.charge:
            parts.append("chg%d" % self.charge)
        if self.bracket:
            parts.append("h%d" % self.explicit_h)
        if self.aromatic:
            parts.append("ar")
        self._label_cache = "|".join(parts)
        return self._label_cache


@dataclass
class Bond:
    a1: int
    a2: int
    order: int = 1
    in_ring: bool = False
    # 查询片段专用
    is_ring_query: bool = False     # 如 -r：要求该键是环内键
    is_chain_query: bool = False    # 如 -c：要求该键是链键


class Mol:
    def __init__(self) -> None:
        self.atoms: List[Atom] = []
        self.bonds: List[Bond] = []
        self.adj: Dict[int, List[int]] = {}     # atom idx -> bond idx
        self.name: str = ""
        self._implicit_h: Optional[List[int]] = None
        # 最小环基缓存：键集合变化时由 add_bond 失效
        self._cycle_cache: Optional[List[List[int]]] = None

    # ---- 基本增删 ----
    def add_atom(self, atom: Atom) -> int:
        atom.index = len(self.atoms)
        self.atoms.append(atom)
        self.adj[atom.index] = []
        self._cycle_cache = None
        return atom.index

    def add_bond(self, a1: int, a2: int, order: int = 1,
                 is_ring_query: bool = False,
                 is_chain_query: bool = False) -> int:
        for j in self.adj[a1]:
            b = self.bonds[j]
            if (b.a1 == a2 or b.a2 == a2):
                raise ValueError("原子 %d 与 %d 之间已存在键" % (a1, a2))
        bidx = len(self.bonds)
        self.bonds.append(Bond(a1, a2, order,
                               is_ring_query=is_ring_query,
                               is_chain_query=is_chain_query))
        self.adj[a1].append(bidx)
        self.adj[a2].append(bidx)
        self._cycle_cache = None
        return bidx

    def n_atoms(self) -> int:
        return len(self.atoms)

    def n_bonds(self) -> int:
        return len(self.bonds)

    def neighbors(self, ai: int) -> List[int]:
        return [self.other(bi, ai) for bi in self.adj[ai]]

    def bond_between(self, a1: int, a2: int) -> Optional[Bond]:
        for bi in self.adj[a1]:
            b = self.bonds[bi]
            if b.a1 == a2 or b.a2 == a2:
                return b
        return None

    def bond_index_between(self, a1: int, a2: int) -> Optional[int]:
        for bi in self.adj[a1]:
            b = self.bonds[bi]
            if b.a1 == a2 or b.a2 == a2:
                return bi
        return None

    def other(self, bond_idx: int, ai: int) -> int:
        b = self.bonds[bond_idx]
        return b.a2 if b.a1 == ai else b.a1

    def degree(self, ai: int) -> int:
        return len(self.adj[ai])

    def components(self) -> List[List[int]]:
        seen = [False] * self.n_atoms()
        comps: List[List[int]] = []
        for s in range(self.n_atoms()):
            if seen[s]:
                continue
            stack = [s]
            seen[s] = True
            comp = []
            while stack:
                x = stack.pop()
                comp.append(x)
                for n in self.neighbors(x):
                    if not seen[n]:
                        seen[n] = True
                        stack.append(n)
            comps.append(comp)
        return comps

    def total_h(self, ai: int) -> int:
        """显式氢 + 隐式氢。需要先调用 valence.compute_implicit_h。"""
        h = self.atoms[ai].explicit_h
        if self._implicit_h is not None:
            h += self._implicit_h[ai]
        return h

    def formula(self) -> str:
        """分子式（Hill 记法）。"""
        counts: Dict[str, int] = {}
        for a in self.atoms:
            counts[a.element] = counts.get(a.element, 0) + 1
            h = self.total_h(a.index)
            if h:
                counts["H"] = counts.get("H", 0) + h
        keys = sorted(counts)
        if "C" in counts:
            ordered = (["C"] + (["H"] if "H" in counts else []) +
                       [k for k in keys if k not in ("C", "H")])
        else:
            ordered = keys
        return "".join(k + (str(counts[k]) if counts[k] > 1 else "")
                       for k in ordered)

    def copy(self) -> "Mol":
        m = Mol()
        m.name = self.name
        for a in self.atoms:
            na = Atom(index=0, symbol=a.symbol, charge=a.charge,
                      isotope=a.isotope, explicit_h=a.explicit_h,
                      bracket=a.bracket, aromatic=a.aromatic,
                      wildcard=a.wildcard,
                      aromatic_query=a.aromatic_query,
                      aliphatic_query=a.aliphatic_query)
            m.add_atom(na)
        for b in self.bonds:
            m.add_bond(b.a1, b.a2, b.order, b.is_ring_query,
                       b.is_chain_query)
            m.bonds[-1].in_ring = b.in_ring
        if self._implicit_h is not None:
            m._implicit_h = list(self._implicit_h)
        return m
