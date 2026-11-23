# -*- coding: utf-8 -*-
"""子结构匹配：VF2 风格的带回溯子图同构。

化学语义约束：
- 片段芳香原子（小写或被感知为芳香）只能命中目标的芳香原子，
  脂肪原子只能命中脂肪原子，互不混配。
- 芳香键 ":" 只匹配目标芳香键；脂肪单/双/三键不匹配芳香键。
- 通配原子 * 匹配任意元素（仍受环/氢等查询标记约束）；
  通配键 ~ 匹配任意键；@ / @! 限定环内/环外键。
- 电荷、同位素按精确值匹配；括号显式 H 按总氢数匹配；
  R/!R、r<n>/!r<n> 按目标 SSSR 环归属匹配。

目标分子在库里是凯库勒化后的整数键级图，芳香性在这里按需重新感知并缓存。
"""
from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Optional, Set, Tuple

from .aromatic import perceive_aromaticity
from .graph import Molecule


class QueryPrepareError(Exception):
    pass


# --------------------------------------------------------------------------
# 目标芳香性缓存（按对象 id，避免重复 SSSR/感知）
# --------------------------------------------------------------------------
def target_aromatic(mol: Molecule) -> Tuple[Set[int], Set[int]]:
    key = id(mol)
    cache = target_aromatic._cache  # type: ignore
    if key in cache:
        return cache[key]
    aa, ab = perceive_aromaticity(mol)
    # 控制缓存规模：超过阈值清空（库检索按分子顺序访问，命中率仍高）
    if len(cache) > 4096:
        cache.clear()
    cache[key] = (aa, ab)
    return aa, ab


target_aromatic._cache = {}  # type: ignore


def prepare_query(mol: Molecule) -> Molecule:
    """查询片段的准备：填隐氢并感知芳香性，但保留芳香标记（不做凯库勒化）。"""
    from .aromatic import implicit_hydrogens
    implicit_hydrogens(mol)
    aa, ab = perceive_aromaticity(mol)
    for i in aa:
        mol.atoms[i].aromatic = True
    for bi in ab:
        mol.bonds[bi].aromatic = True
    mol.compute_fragments()
    return mol


# --------------------------------------------------------------------------
# 原子/键相容性
# --------------------------------------------------------------------------
def atom_compatible(qmol: Molecule, qi: int,
                    tmol: Molecule, ti: int,
                    t_arom_atoms: Set[int]) -> bool:
    qa = qmol.atoms[qi]
    ta = tmol.atoms[ti]

    if qa.is_wildcard:
        pass  # 元素通配，继续检查环/氢等标记
    else:
        # 元素必须一致（查询符号统一用大写比较，芳香标记单独判断）
        if qa.symbol.upper() != ta.symbol.upper():
            return False
        # 芳香/脂肪严格区分
        if qa.aromatic and ti not in t_arom_atoms:
            return False
        if (not qa.aromatic) and ti in t_arom_atoms:
            return False
        # 电荷、同位素精确匹配
        if qa.charge != ta.charge:
            return False
        if qa.isotope != 0 and qa.isotope != ta.isotope:
            return False

    # 环内/环外原子查询
    if qa.in_ring_query is not None:
        in_ring = ti in tmol.ring_atoms()
        if qa.in_ring_query != in_ring:
            return False
    if qa.ring_size_query is not None:
        sizes = tmol.ring_sizes_of_atom(ti)
        want = qa.ring_size_query
        if want > 0 and want not in sizes:
            return False
        if want < 0 and (-want) in sizes:
            return False
    # 氢数查询：只有显式写了 H（h_query 或括号 H）才约束目标总氢数；
    # 裸原子和只带环标记的括号原子（如 [CR]）不限制氢。
    if qa.h_query is not None:
        if ta.total_h != qa.h_query:
            return False
    elif not qa.is_wildcard and qa.explicit_h > 0:
        if ta.total_h != qa.total_h:
            return False
    return True


def bond_compatible(qmol: Molecule, qb,
                    tmol: Molecule, tb,
                    t_arom_bonds: Set[int]) -> bool:
    if qb.is_wildcard:
        pass  # 键型通配，仍检查环标记
    elif qb.aromatic:
        if id(tb) and _bond_id(tmol, tb) not in t_arom_bonds:
            return False
    else:
        # 脂肪确定键级：不能落到芳香键上，且键级相等
        if _bond_id(tmol, tb) in t_arom_bonds:
            return False
        if qb.order != tb.order:
            return False
    if qb.in_ring_query is not None:
        in_ring = _bond_id(tmol, tb) in tmol.ring_bonds()
        if qb.in_ring_query != in_ring:
            return False
    return True


def _bond_id(mol: Molecule, b) -> int:
    return mol.bonds.index(b)


# --------------------------------------------------------------------------
# VF2 风格子图同构
# --------------------------------------------------------------------------
def substructure_matches(qmol: Molecule, tmol: Molecule,
                         max_matches: Optional[int] = None
                         ) -> List[Dict[int, int]]:
    """返回所有（或前 max_matches 个）映射：查询原子 idx -> 目标原子 idx。"""
    if len(qmol.atoms) > len(tmol.atoms):
        return []
    t_arom_atoms, t_arom_bonds = target_aromatic(tmol)

    # 候选表：按原子静态属性预筛
    candidates: Dict[int, List[int]] = {}
    for qi in range(len(qmol.atoms)):
        cand = [ti for ti in range(len(tmol.atoms))
                if atom_compatible(qmol, qi, tmol, ti, t_arom_atoms)]
        if not cand:
            return []
        candidates[qi] = cand

    # 多片段（点断）查询：各连通分量独立匹配，组合在最后笛卡尔展开
    qfrags = qmol.compute_fragments()
    if len(qfrags) > 1:
        return _match_disconnected(qmol, tmol, qfrags, candidates,
                                   t_arom_bonds, max_matches)

    # 查询原子排序：约束强（候选少、度大、环内）优先
    order = sorted(
        range(len(qmol.atoms)),
        key=lambda i: (len(candidates[i]), -qmol.degree(i), i))

    mapping: Dict[int, int] = {}
    used: Set[int] = set()
    results: List[Dict[int, int]] = []

    def backtrack(k: int) -> bool:
        if max_matches is not None and len(results) >= max_matches:
            return True
        if k == len(order):
            results.append(dict(mapping))
            return max_matches is not None and len(results) >= max_matches
        qi = order[k]
        for ti in candidates[qi]:
            if ti in used:
                continue
            if not _check_partial_edges(qmol, qi, ti, mapping, tmol,
                                        t_arom_bonds):
                continue
            mapping[qi] = ti
            used.add(ti)
            backtrack(k + 1)
            used.discard(ti)
            del mapping[qi]
            if max_matches is not None and len(results) >= max_matches:
                return True
        return False

    backtrack(0)
    return results


def _check_partial_edges(qmol: Molecule, qi: int, ti: int,
                         mapping: Dict[int, int], tmol: Molecule,
                         t_arom_bonds: Set[int]) -> bool:
    """新配对 (qi->ti) 与已映射邻居之间的键必须全部相容。"""
    for qbi in qmol.adj[qi]:
        qb = qmol.bonds[qbi]
        qj = qb.other(qi)
        if qj not in mapping:
            continue
        tj = mapping[qj]
        tb = tmol.bond_between(ti, tj)
        if tb is None:
            return False
        if not bond_compatible(qmol, qb, tmol, tb, t_arom_bonds):
            return False
    return True


def _match_disconnected(qmol, tmol, qfrags, candidates,
                        t_arom_bonds, max_matches):
    """点断片段：逐分量求匹配，要求各分量占用不同目标原子。"""
    per_frag: List[List[Dict[int, int]]] = []
    for frag in qfrags:
        sub = _fragment_query(qmol, frag)
        mats = substructure_matches(sub, tmol)
        if not mats:
            return []
        # 把子图局部 idx 还原回全局 idx
        restored = [{frag[k]: v for k, v in m.items()} for m in mats]
        per_frag.append(restored)

    results: List[Dict[int, int]] = []

    def combine(k: int, acc: Dict[int, int], used: Set[int]) -> None:
        if max_matches is not None and len(results) >= max_matches:
            return
        if k == len(per_frag):
            results.append(dict(acc))
            return
        for m in per_frag[k]:
            if set(m.values()) & used:
                continue
            acc.update(m)
            combine(k + 1, acc, used | set(m.values()))
            for qk in m:
                del acc[qk]

    combine(0, {}, set())
    return results


def _fragment_query(qmol: Molecule, frag: List[int]) -> Molecule:
    """抽出查询的一个连通分量，重新编号为独立 Molecule。"""
    sub = Molecule()
    old_to_new: Dict[int, int] = {}
    for old in frag:
        old_to_new[old] = sub.add_atom(qmol.atoms[old].copy(-1))
    for b in qmol.bonds:
        if b.a1 in old_to_new and b.a2 in old_to_new:
            sub.add_bond(old_to_new[b.a1], old_to_new[b.a2],
                         order=b.order, aromatic=b.aromatic,
                         is_wildcard=b.is_wildcard,
                         in_ring_query=b.in_ring_query)
    sub.compute_fragments()
    return sub


def has_substructure(qmol: Molecule, tmol: Molecule) -> bool:
    return bool(substructure_matches(qmol, tmol, max_matches=1))
