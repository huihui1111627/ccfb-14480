# -*- coding: utf-8 -*-
"""分子图数据结构：原子、键、分子。

设计要点：
- Atom 保存元素、电荷、同位素、显式 H、芳香标记；隐式 H 在规范化阶段算出后填入。
- Bond 用 order 表示键级（1/2/3），aromatic 表示芳香键；查询用的通配/环内标记放在 query 字段。
- Molecule 用邻接表存图，附带解析来源（不参与相等性判断）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class Atom:
    idx: int
    symbol: str                 # 元素符号，大写开头（芳香原子仍存 "c" 小写直到 arom 阶段统一）
    charge: int = 0
    isotope: int = 0            # 0 表示未指定同位素
    explicit_h: int = -1        # -1 表示未在括号中指定（交给默认价推算）
    aromatic: bool = False
    # 查询（SMARTS-lite）专用属性，普通分子为 False/None
    is_wildcard: bool = False   # * 任意原子
    in_ring_query: Optional[bool] = None     # True=必须环内 False=必须非环
    ring_size_query: Optional[int] = None    # 要求所在环的大小
    h_query: Optional[int] = None            # 要求总氢数
    # 规范化时算出
    implicit_h: int = 0
    class_label: Optional[int] = None        # [C:1] 原子映射类号（查询/输出用）

    def copy(self, idx: int) -> "Atom":
        return Atom(
            idx=idx, symbol=self.symbol, charge=self.charge,
            isotope=self.isotope, explicit_h=self.explicit_h,
            aromatic=self.aromatic, is_wildcard=self.is_wildcard,
            in_ring_query=self.in_ring_query, ring_size_query=self.ring_size_query,
            h_query=self.h_query, implicit_h=self.implicit_h,
            class_label=self.class_label,
        )

    @property
    def total_h(self) -> int:
        if self.explicit_h == -1:
            return self.implicit_h
        return self.explicit_h + self.implicit_h


@dataclass
class Bond:
    a1: int
    a2: int
    order: int = 1              # 1 单 2 双 3 三；芳香键 order=1 且 aromatic=True
    aromatic: bool = False
    # 查询专用
    is_wildcard: bool = False   # ~ 任意键
    in_ring_query: Optional[bool] = None  # @ 环内键 / @! 非环键

    def other(self, i: int) -> int:
        return self.a2 if i == self.a1 else self.a1


@dataclass
class Molecule:
    atoms: List[Atom] = field(default_factory=list)
    bonds: List[Bond] = field(default_factory=list)
    name: str = ""
    # 解析/规范化辅助
    adj: Dict[int, List[int]] = field(default_factory=dict)   # atom idx -> bond idx
    fragments: List[List[int]] = field(default_factory=list)  # 连通分量（原子 idx）

    # ---------- 基础图操作 ----------
    def add_atom(self, atom: Atom) -> int:
        atom.idx = len(self.atoms)
        self.atoms.append(atom)
        self.adj[atom.idx] = []
        return atom.idx

    def add_bond(self, a1: int, a2: int, order: int = 1,
                 aromatic: bool = False, is_wildcard: bool = False,
                 in_ring_query: Optional[bool] = None) -> Bond:
        b = Bond(a1, a2, order=order, aromatic=aromatic,
                 is_wildcard=is_wildcard, in_ring_query=in_ring_query)
        bi = len(self.bonds)
        self.bonds.append(b)
        self.adj[a1].append(bi)
        self.adj[a2].append(bi)
        return b

    def bond_between(self, a1: int, a2: int) -> Optional[Bond]:
        for bi in self.adj[a1]:
            b = self.bonds[bi]
            if b.other(a1) == a2:
                return b
        return None

    def neighbors(self, i: int) -> List[int]:
        return [self.bonds[bi].other(i) for bi in self.adj[i]]

    def degree(self, i: int) -> int:
        return len(self.adj[i])

    def bond(self, bi: int) -> Bond:
        return self.bonds[bi]

    def compute_fragments(self) -> List[List[int]]:
        seen: Set[int] = set()
        frags: List[List[int]] = []
        for start in range(len(self.atoms)):
            if start in seen:
                continue
            stack = [start]
            comp: List[int] = []
            seen.add(start)
            while stack:
                x = stack.pop()
                comp.append(x)
                for y in self.neighbors(x):
                    if y not in seen:
                        seen.add(y)
                        stack.append(y)
            frags.append(sorted(comp))
        self.fragments = frags
        return frags

    def clone(self) -> "Molecule":
        m = Molecule(name=self.name)
        for a in self.atoms:
            m.add_atom(a.copy(len(m.atoms)))
        for b in self.bonds:
            m.add_bond(b.a1, b.a2, order=b.order, aromatic=b.aromatic,
                       is_wildcard=b.is_wildcard, in_ring_query=b.in_ring_query)
        m.fragments = [list(f) for f in self.fragments]
        return m

    # ---------- 环检测 ----------
    def ring_bonds(self) -> Set[int]:
        """返回所有属于某个环的键的索引（并集，基于 biconnected 分量的简化实现）。"""
        return set(self._sssr_bond_set())

    def ring_atoms(self) -> Set[int]:
        out: Set[int] = set()
        for bi in self._sssr_bond_set():
            b = self.bonds[bi]
            out.add(b.a1)
            out.add(b.a2)
        return out

    def _sssr_bond_set(self) -> List[int]:
        """最小环基（环的并集足以判定环内/环外，SSSR 不是必需但尺寸小）。

        用每个连通分量的生成树，余边各产生一个基础环；再用高斯消元
        在线性空间里去掉线性相关的环，保留独立基（SSSR 规模）。
        """
        self.compute_fragments()
        basis_cycles: List[Set[int]] = []
        for comp in self.fragments:
            compset = set(comp)
            # BFS 生成树
            parent: Dict[int, Tuple[int, int]] = {}
            tree_bonds: Set[int] = set()
            root = comp[0]
            parent[root] = (-1, -1)
            queue = [root]
            head = 0
            while head < len(queue):
                x = queue[head]
                head += 1
                for bi in self.adj[x]:
                    y = self.bonds[bi].other(x)
                    if y not in compset:
                        continue
                    if y in parent:
                        continue
                    parent[y] = (x, bi)
                    tree_bonds.add(bi)
                    queue.append(y)
            # 余边 -> 基础环
            for x in comp:
                for bi in self.adj[x]:
                    b = self.bonds[bi]
                    if bi in tree_bonds or b.a2 != x and b.a1 != x:
                        pass
                    y = b.other(x)
                    if bi in tree_bonds or y < x:
                        continue
                    # 非树边，沿父链找路径
                    cyc = {bi}
                    u, v = x, y
                    depth_u = self._depth_to(root, u, parent)
                    depth_v = self._depth_to(root, v, parent)
                    while depth_u > depth_v:
                        pu, pbi = parent[u]
                        cyc.add(pbi)
                        u = pu
                        depth_u -= 1
                    while depth_v > depth_u:
                        pv, pbi = parent[v]
                        cyc.add(pbi)
                        v = pv
                        depth_v -= 1
                    while u != v:
                        pu, pbi1 = parent[u]
                        pv, pbi2 = parent[v]
                        cyc.add(pbi1)
                        cyc.add(pbi2)
                        u, v = pu, pv
                    basis_cycles.append(cyc)
        # 高斯消元去线性相关（键集合按 GF(2)）
        independent: List[Set[int]] = []
        pivots: Dict[int, Set[int]] = {}
        for cyc in basis_cycles:
            cur = set(cyc)
            while cur:
                pivot = max(cur)  # 确定性选主元
                if pivot in pivots:
                    cur ^= pivots[pivot]
                else:
                    pivots[pivot] = cur
                    independent.append(cur)
                    break
        bond_set: Set[int] = set()
        for cyc in independent:
            bond_set |= cyc
        self._sssr = independent
        return sorted(bond_set)

    @staticmethod
    def _depth_to(root: int, node: int, parent: Dict[int, Tuple[int, int]]) -> int:
        d = 0
        while node != root:
            node = parent[node][0]
            d += 1
        return d

    def sssr(self) -> List[Set[int]]:
        self._sssr_bond_set()
        return self._sssr

    def ring_sizes_of_atom(self, i: int) -> Set[int]:
        out = set()
        for cyc in self.sssr():
            atoms = set()
            for bi in cyc:
                b = self.bonds[bi]
                atoms.add(b.a1)
                atoms.add(b.a2)
            if i in atoms:
                out.add(len(atoms))
        return out

    # ---------- 键价和 ----------
    def bond_valence_sum(self, i: int) -> float:
        """该原子当前成键的价数和（芳香键按 1.5 计）。"""
        total = 0.0
        for bi in self.adj[i]:
            b = self.bonds[bi]
            total += 1.5 if b.aromatic else b.order
        return total
