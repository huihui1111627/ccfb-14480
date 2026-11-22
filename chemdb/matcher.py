"""子结构匹配：带化学语义的回溯子图同构（VF2 风格）。

语义规则：
  * 芳香原子只能匹配芳香原子，脂肪原子只能匹配脂肪原子；
  * 元素、电荷、同位素必须一致（查询用 * / a / A 放宽）；
  * 键级必须一致；芳香键只能匹配芳香键；~ 任意键；
  * -r / =r 等要求目标键为环内键，-c 要求为链键；
  * 查询原子的隐式/显式氢约束：目标氢数 >= 查询氢数。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .molecule import Mol


def atoms_compatible(q: "Atom-like", t: "Atom-like",
                     qmol: Mol, tmol: Mol) -> bool:
    if q.wildcard:
        return True
    if q.aromatic_query:
        return t.aromatic
    if q.aliphatic_query:
        return not t.aromatic
    # 芳香/脂肪必须同类
    if q.aromatic != t.aromatic:
        return False
    if q.element != t.element:
        return False
    if q.charge != t.charge:
        return False
    if q.isotope is not None and q.isotope != t.isotope:
        return False
    # 氢约束：只有查询里“显式写出”的氢（方括号 H，如 [nH]、[CH3]）
    # 才是必须满足的最小要求。裸原子的隐式氢可被取代基替换，
    # 不参与约束，否则苯片段匹配不上带取代基的芳香碳。
    if q.bracket and q.explicit_h > tmol.total_h(t.index):
        return False
    return True


def bonds_compatible(qb, tb) -> bool:
    if qb.order == 0:  # ~ 任意键
        pass
    elif qb.order == 4:
        if tb.order != 4:
            return False
    else:
        if qb.order != tb.order:
            return False
    if qb.is_ring_query and not tb.in_ring:
        return False
    if qb.is_chain_query and tb.in_ring:
        return False
    return True


def _candidate_atoms(qmol: Mol, tmol: Mol) -> Dict[int, List[int]]:
    cands: Dict[int, List[int]] = {}
    for qa in qmol.atoms:
        c = []
        for ta in tmol.atoms:
            if atoms_compatible(qa, ta, qmol, tmol):
                c.append(ta.index)
        cands[qa.index] = c
    return cands


def substructure_matches(qmol: Mol, tmol: Mol,
                         max_matches: Optional[int] = None) -> List[Dict[int, int]]:
    """返回所有匹配（查询原子 -> 目标原子 的映射），max_matches 截断。"""
    if qmol.n_atoms() > tmol.n_atoms():
        return []
    cands = _candidate_atoms(qmol, tmol)
    if any(len(v) == 0 for v in cands.values()):
        return []
    # 选择查询原子顺序：候选少、度大优先（MRV 思想）
    order = sorted(qmol.atoms,
                   key=lambda a: (len(cands[a.index]), -qmol.degree(a.index),
                                  a.index))
    order_idx = [a.index for a in order]
    mapping: Dict[int, int] = {}
    used: set = set()
    results: List[Dict[int, int]] = []

    def feasible(qi: int, ti: int) -> bool:
        # 已映射邻居的键约束
        for qbi in qmol.adj[qi]:
            qn = qmol.other(qbi, qi)
            if qn not in mapping:
                continue
            tn = mapping[qn]
            tb = tmol.bond_between(ti, tn)
            if tb is None:
                return False
            if not bonds_compatible(qmol.bonds[qbi], tb):
                return False
        # 前瞻：未映射的查询邻居，目标原子必须有足够未用候选邻居
        for qbi in qmol.adj[qi]:
            qn = qmol.other(qbi, qi)
            if qn in mapping:
                continue
            avail = 0
            for tbi in tmol.adj[ti]:
                tn = tmol.other(tbi, ti)
                if tn in used:
                    continue
                if bonds_compatible(qmol.bonds[qbi], tmol.bonds[tbi]) \
                        and atoms_compatible(qmol.atoms[qn],
                                             tmol.atoms[tn], qmol, tmol):
                    avail += 1
            if avail == 0:
                return False
        return True

    def backtrack(k: int) -> None:
        if max_matches is not None and len(results) >= max_matches:
            return
        if k == len(order_idx):
            results.append(dict(mapping))
            return
        qi = order_idx[k]
        for ti in cands[qi]:
            if ti in used:
                continue
            if not feasible(qi, ti):
                continue
            mapping[qi] = ti
            used.add(ti)
            backtrack(k + 1)
            used.discard(ti)
            del mapping[qi]
            if max_matches is not None and len(results) >= max_matches:
                return

    backtrack(0)
    return results


def has_substructure(qmol: Mol, tmol: Mol) -> bool:
    return bool(substructure_matches(qmol, tmol, max_matches=1))
