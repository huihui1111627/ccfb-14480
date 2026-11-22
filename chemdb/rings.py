"""环感知：最小环基（Horton 算法 + GF(2) 高斯消元）与环内键标记。"""
from __future__ import annotations

from collections import deque
from typing import Dict, List, Set, Tuple

from .molecule import Mol


def _edge_key(b: int, mol: Mol) -> Tuple[int, int]:
    bd = mol.bonds[b]
    return (bd.a1, bd.a2) if bd.a1 < bd.a2 else (bd.a2, bd.a1)


def minimum_cycle_basis(mol: Mol) -> List[List[int]]:
    """返回最小环基，每个环用原子下标列表表示（按环上顺序）。

    Horton 思路：以每个顶点为根做 BFS 生成最短路径树，
    对每条非树边得到候选环，按长度排序后在 GF(2) 上贪心取线性无关者。
    结果缓存在 mol 上，图的原子/键增删会使缓存失效。
    """
    if mol._cycle_cache is not None:
        return mol._cycle_cache
    n = mol.n_atoms()
    # 每条边一个 bit 位
    edge_bit: Dict[Tuple[int, int], int] = {}
    for bi in range(mol.n_bonds()):
        edge_bit[_edge_key(bi, mol)] = bi

    candidates: List[Tuple[int, frozenset, List[int]]] = []

    for root in range(n):
        parent = {root: -1}
        pedge: Dict[int, int] = {}
        dist = {root: 0}
        q = deque([root])
        while q:
            u = q.popleft()
            for bi in mol.adj[u]:
                v = mol.other(bi, u)
                if v not in dist:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    pedge[v] = bi
                    q.append(v)
        # 对该根可达的每条非树边构造候选环
        for bi in range(mol.n_bonds()):
            bd = mol.bonds[bi]
            u, v = bd.a1, bd.a2
            if u not in dist or v not in dist:
                continue
            if pedge.get(u) == bi or pedge.get(v) == bi:
                continue  # 树边
            # u->root 与 v->root 的路径拼接 + 非树边
            pu = _path_to_root(u, parent)
            pv = _path_to_root(v, parent)
            i = len(pu) - 1
            j = len(pv) - 1
            while i >= 0 and j >= 0 and pu[i] == pv[j]:
                i -= 1
                j -= 1
            lca_up = pu[i + 1]
            cycle_atoms = pu[: i + 2] + list(reversed(pv[: j + 1]))
            if len(cycle_atoms) < 3:
                continue
            edge_set = _cycle_edges(cycle_atoms, mol, bi, lca_up)
            candidates.append(
                (len(cycle_atoms), frozenset(edge_set), cycle_atoms))

    # 去重 + 按环长排序
    uniq: Dict[frozenset, List[int]] = {}
    for length, eset, atoms in candidates:
        uniq.setdefault(eset, atoms)
    ordered = sorted(uniq.items(), key=lambda kv: len(kv[0]))

    # GF(2) 高斯消元，保证取到最小线性无关集合（最小环基）
    basis: Dict[int, frozenset] = {}
    result: List[List[int]] = []
    for eset, atoms in ordered:
        vec = 0
        for e in eset:
            vec |= 1 << e
        reduced = vec
        pivot = reduced.bit_length() - 1
        while pivot >= 0 and pivot in basis:
            reduced ^= _bits_to_int(basis[pivot])
            pivot = reduced.bit_length() - 1
        if reduced:
            basis[pivot] = eset
            result.append(atoms)
        if len(result) == mol.n_bonds() - n + len(mol.components()):
            break
    mol._cycle_cache = result
    return result


def _bits_to_int(s: frozenset) -> int:
    v = 0
    for e in s:
        v |= 1 << e
    return v


def _path_to_root(x: int, parent: Dict[int, int]) -> List[int]:
    path = [x]
    while parent[x] != -1:
        x = parent[x]
        path.append(x)
    return path


def _cycle_edges(atoms: List[int], mol: Mol,
                 extra_edge: int, lca: int) -> Set[int]:
    edges: Set[int] = set()
    for k in range(len(atoms) - 1):
        bi = mol.bond_index_between(atoms[k], atoms[k + 1])
        if bi is not None:
            edges.add(bi)
    edges.add(extra_edge)
    return edges


def mark_ring_bonds(mol: Mol) -> List[List[int]]:
    """把环内键 in_ring 置 True，返回最小环基列表。

    环内键标记用线性时间的生成树法（非树边与其树路径上的边都在环内），
    最小环基只在调用方真正需要环列表时才计算并走缓存。
    """
    for b in mol.bonds:
        b.in_ring = False
    _flag_ring_bonds(mol)
    return minimum_cycle_basis(mol)


def _flag_ring_bonds(mol: Mol) -> None:
    """生成树 + 基本回路：线性时间标记所有环内键/环原子。"""
    n = mol.n_atoms()
    seen = [False] * n
    parent_bond: Dict[int, int] = {}
    parent_atom: Dict[int, int] = {}
    non_tree_edges: List[int] = []
    for start in range(n):
        if seen[start]:
            continue
        seen[start] = True
        stack = [start]
        while stack:
            u = stack.pop()
            for bi in mol.adj[u]:
                v = mol.other(bi, u)
                if not seen[v]:
                    seen[v] = True
                    parent_atom[v] = u
                    parent_bond[v] = bi
                    stack.append(v)
                elif parent_bond.get(u) != bi and parent_bond.get(v) != bi:
                    if bi not in non_tree_edges:
                        non_tree_edges.append(bi)
    # 每条非树边与其两端在生成树上的路径构成一个基本回路
    ring_bonds: Set[int] = set()
    for bi in non_tree_edges:
        b = mol.bonds[bi]
        ring_bonds.add(bi)
        u, v = b.a1, b.a2
        # 沿父边把 u、v 抬到公共祖先
        depth_u = _tree_depth(parent_atom, u)
        depth_v = _tree_depth(parent_atom, v)
        while depth_u > depth_v:
            pb = parent_bond[u]
            ring_bonds.add(pb)
            u = parent_atom[u]
            depth_u -= 1
        while depth_v > depth_u:
            pb = parent_bond[v]
            ring_bonds.add(pb)
            v = parent_atom[v]
            depth_v -= 1
        while u != v:
            ring_bonds.add(parent_bond[u])
            ring_bonds.add(parent_bond[v])
            u = parent_atom[u]
            v = parent_atom[v]
    for bi in ring_bonds:
        mol.bonds[bi].in_ring = True


def _tree_depth(parent_atom: Dict[int, int], x: int) -> int:
    d = 0
    while x in parent_atom:
        x = parent_atom[x]
        d += 1
    return d


def ring_atom_set(mol: Mol) -> Set[int]:
    """环原子集合：直接由 in_ring 标记得到（线性时间），无需最小环基。"""
    s: Set[int] = set()
    for b in mol.bonds:
        if b.in_ring:
            s.add(b.a1)
            s.add(b.a2)
    return s


def ring_bond_set(mol: Mol) -> Set[int]:
    return {bi for bi, b in enumerate(mol.bonds) if b.in_ring}
