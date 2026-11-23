# -*- coding: utf-8 -*-
"""骨架（scaffold）归类：Murcko 骨架的自研实现。

做法：
1. 标记所有环原子与连接环的连接子（linker）原子。
2. 反复剥离只挂在非环原子上的末端非环原子（侧链），
   直到剩下“环 + 环间连接子”组成的框架。
3. 无环分子归入无骨架桶（acrylic），用空串表示。
4. 对骨架图规范化，得到稳定骨架键，用于统计与归类。
"""
from __future__ import annotations

from typing import Dict, List, Set, Tuple

from .canonical import canonicalize, canonical_smiles
from .graph import Molecule


def murcko_scaffold(mol: Molecule) -> Molecule:
    """返回 Murcko 骨架分子（可能为空图，表示无环/无骨架）。"""
    ring_atoms = mol.ring_atoms()
    if not ring_atoms:
        return Molecule()

    keep = set(ring_atoms)
    # 连接子：位于两个环原子路径上的非环原子。
    # 简化但正确的做法：保留所有“移除后会使环集合断开”的非环原子——
    # 即落在环原子两两最短路径上的非环原子。
    linker = _linker_atoms(mol, ring_atoms)
    keep |= linker

    old_to_new: Dict[int, int] = {}
    scaffold = Molecule()
    for old in sorted(keep):
        old_to_new[old] = scaffold.add_atom(mol.atoms[old].copy(-1))
    for b in mol.bonds:
        if b.a1 in keep and b.a2 in keep:
            scaffold.add_bond(old_to_new[b.a1], old_to_new[b.a2],
                              order=b.order, aromatic=b.aromatic,
                              is_wildcard=b.is_wildcard,
                              in_ring_query=b.in_ring_query)
    scaffold.compute_fragments()
    # 剥离孤立的非环碎片（不连任何环的残留）
    _drop_disconnected_acyclic(scaffold)
    return scaffold


def _linker_atoms(mol: Molecule, ring_atoms: Set[int]) -> Set[int]:
    """找出连接两个（或更多）环区域的非环原子。"""
    # 以环原子为多源 BFS 起点，记录每个非环原子被多少个不同环区域触及。
    # 更直接：枚举环原子之间的最短路径，路径上的非环原子即连接子。
    linker: Set[int] = set()
    ring_list = sorted(ring_atoms)
    # 用 BFS 距离父节点，限制配对规模（分子通常不大）
    for idx, src in enumerate(ring_list):
        parent = _bfs_parent(mol, src)
        for dst in ring_list[idx + 1:]:
            if dst not in parent:
                continue
            path = _walk_path(parent, src, dst)
            for node in path:
                if node not in ring_atoms:
                    linker.add(node)
    return linker


def _bfs_parent(mol: Molecule, src: int) -> Dict[int, int]:
    parent = {src: -1}
    queue = [src]
    head = 0
    while head < len(queue):
        x = queue[head]
        head += 1
        for y in mol.neighbors(x):
            if y not in parent:
                parent[y] = x
                queue.append(y)
    return parent


def _walk_path(parent: Dict[int, int], src: int, dst: int) -> List[int]:
    path = [dst]
    cur = dst
    while cur != src:
        cur = parent[cur]
        path.append(cur)
    return path


def _drop_disconnected_acyclic(mol: Molecule) -> None:
    ring = mol.ring_atoms()
    keep: Set[int] = set()
    for frag in mol.compute_fragments():
        fragset = set(frag)
        if fragset & ring:
            keep |= fragset
    if len(keep) == len(mol.atoms):
        return
    # 原地重建
    old_to_new: Dict[int, int] = {}
    rebuilt = Molecule()
    for old in sorted(keep):
        old_to_new[old] = rebuilt.add_atom(mol.atoms[old].copy(-1))
    for b in mol.bonds:
        if b.a1 in keep and b.a2 in keep:
            rebuilt.add_bond(old_to_new[b.a1], old_to_new[b.a2],
                             order=b.order, aromatic=b.aromatic)
    mol.atoms = rebuilt.atoms
    mol.bonds = rebuilt.bonds
    mol.adj = rebuilt.adj
    mol.compute_fragments()


def scaffold_key(mol: Molecule) -> str:
    """规范骨架键：空串表示无环分子。"""
    scaf = murcko_scaffold(mol)
    if not scaf.atoms:
        return ""
    canon = canonicalize(scaf)
    return canonical_smiles(canon)


def scaffold_family_counts(mols: List[Molecule]) -> List[Tuple[str, int]]:
    """统计一批分子的骨架出现频次，按次数降序、骨架键升序排列。"""
    counts: Dict[str, int] = {}
    for m in mols:
        key = scaffold_key(m)
        counts[key] = counts.get(key, 0) + 1
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
