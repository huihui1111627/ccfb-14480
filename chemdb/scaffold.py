"""骨架归类：Bemis-Murcko 骨架及其通用（generic）形式。

做法：反复删除非环的端基原子（度为 1 且不在环内），
保留全部环以及连接环的链；无环分子骨架为空串。
通用骨架把元素统一替换成通配原子，只保留连接关系与键级/芳香性，
用于按“拓扑骨架”聚类统计。
"""
from __future__ import annotations

from typing import Optional

from .molecule import Atom, Mol
from .rings import mark_ring_bonds, _flag_ring_bonds
from .canonical import canonical_smiles


def murcko_scaffold(mol: Mol) -> Optional[Mol]:
    # 只依赖线性时间的环内键标记，不计算最小环基
    if not any(b.in_ring for b in mol.bonds):
        from .rings import mark_ring_bonds
        mark_ring_bonds(mol)
    if not any(b.in_ring for b in mol.bonds):
        return None
    # 反复剥端基（度为 1 且不在环内），剩下的环与连接链即骨架
    alive = set(range(mol.n_atoms()))
    while True:
        leaves = [ai for ai in alive
                  if _live_degree(mol, ai, alive) == 1
                  and not any(mol.bonds[bi].in_ring
                              for bi in mol.adj[ai])]
        if not leaves:
            break
        alive.difference_update(leaves)
    if not alive:
        return None
    return _extract(mol, alive)


def generic_scaffold(mol: Mol) -> Optional[Mol]:
    scaf = murcko_scaffold(mol)
    if scaf is None:
        return None
    for a in scaf.atoms:
        if not a.aromatic:
            a.wildcard = True
            a.symbol = "*"
        else:
            # 通用骨架保留芳香性以区分芳香/脂肪环体系
            a.aromatic_query = True
            a.symbol = "*"
    return scaf


def _live_degree(mol: Mol, ai: int, alive: set) -> int:
    return sum(1 for bi in mol.adj[ai]
               if mol.other(bi, ai) in alive)


def _extract(mol: Mol, alive: set) -> Mol:
    out = Mol()
    remap: dict = {}
    for ai in sorted(alive):
        old = mol.atoms[ai]
        na = Atom(0, old.symbol, charge=old.charge, isotope=old.isotope,
                  explicit_h=old.explicit_h, bracket=old.bracket,
                  aromatic=old.aromatic)
        remap[ai] = out.add_atom(na)
    for b in mol.bonds:
        if b.a1 in alive and b.a2 in alive:
            out.add_bond(remap[b.a1], remap[b.a2], b.order)
    _flag_ring_bonds(out)
    return out


def scaffold_key(mol: Mol, generic: bool = False) -> str:
    scaf = generic_scaffold(mol) if generic else murcko_scaffold(mol)
    if scaf is None:
        return ""
    from .valence import compute_implicit_h
    compute_implicit_h(scaf)
    return canonical_smiles(scaf)


def both_scaffold_keys(mol: Mol):
    """一次剥壳同时返回 (具体骨架 key, 通用骨架 key)，无环均为空串。"""
    scaf = murcko_scaffold(mol)
    if scaf is None:
        return "", ""
    from .valence import compute_implicit_h
    compute_implicit_h(scaf)
    specific = canonical_smiles(scaf)
    generic_mol = generic_scaffold(scaf)
    compute_implicit_h(generic_mol)
    generic_key = canonical_smiles(generic_mol)
    return specific, generic_key
