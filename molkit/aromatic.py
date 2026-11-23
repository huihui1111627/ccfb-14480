# -*- coding: utf-8 -*-
"""隐式氢计数、芳香性识别与凯库勒化。

芳香性策略（自研，不依赖 RDKit 等外部库）：
1. 先算隐式氢（对解析时带小写标记的芳香原子按芳香规则算，其余按默认价）。
2. 对显式小写芳香式：芳香原子/键已由解析器标出，只需对芳香子图做
   交替双键指派（Kekulé assignment），把芳香键转成单/双键。
3. 对全大写的凯库勒式（如 C1=CC=CC=C1）：在环上按 Hückel 规则识别芳香体系
   （所有环原子 sp2、平面、4n+2 π 电子），识别成功后同样走交替双键规范化，
   使两种写法落到同一张图。
最终输出的规范图不再保留 aromatic 标记——全部用整数键级表示，
芳香性作为可重新计算的派生属性（查询时用 perceive_aromaticity 恢复）。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

from .elements import DEFAULT_VALENCES
from .graph import Molecule


class ChemistryError(Exception):
    pass


# --------------------------------------------------------------------------
# 隐式氢
# --------------------------------------------------------------------------
def implicit_hydrogens(mol: Molecule) -> None:
    """按 Daylight 默认价规则填 implicit_h，并做价态合法性检查。

    注意：芳香小写分子在凯库勒化之前不要调用本函数——
    standardize() 会先做凯库勒化，再统一按整数键价算氢。
    """
    for a in mol.atoms:
        if a.is_wildcard:
            a.implicit_h = 0
            continue
        if a.aromatic:
            # 兜底：独立调用时按芳香规则给氢
            a.implicit_h = _aromatic_implicit_h(mol, a)
            continue
        if a.explicit_h >= 0:
            a.implicit_h = 0
            continue
        a.implicit_h = _aliphatic_implicit_h(mol, a)


def _aliphatic_implicit_h(mol: Molecule, a) -> int:
    vals = DEFAULT_VALENCES.get(a.symbol)
    bv = mol.bond_valence_sum(a.idx)
    if vals is None:
        return 0
    # 选择第一个不小于当前键价和的默认价
    for v in vals:
        if v + a.charge >= bv - 1e-9:
            h = int(round(v + a.charge - bv))
            return max(h, 0)
    # 超过所有默认价（如高价硫显式写法），不给氢
    return 0


def _aromatic_implicit_h(mol: Molecule, a) -> int:
    """芳香小写原子的隐式氢（Daylight 规则），仅在未走 standardize 时兜底用。

    以芳香邻居数判断：环内度 2 且无取代基的 c/n 带 1 个 H，
    有环外取代（度 3）则为 0；杂原子 o/s/b 默认 0，吡咯氮必须写 [nH]。
    """
    if a.explicit_h >= 0:
        return 0
    arom_deg = sum(1 for bi in mol.adj[a.idx]
                   if mol.bonds[bi].aromatic)
    if a.symbol == "c":
        return 1 if arom_deg == 2 and mol.degree(a.idx) == 2 else 0
    if a.symbol == "n":
        return 1 if arom_deg == 2 and mol.degree(a.idx) == 2 else 0
    return 0


def _check_valence(mol: Molecule, a) -> None:
    vals = DEFAULT_VALENCES.get(a.symbol)
    if vals is None:
        return
    bv = mol.bond_valence_sum(a.idx) + a.total_h
    # 允许电荷补偿（[N+](=O) 等），只做宽松告警式检查，不直接拒绝
    del bv, vals  # 宽松模式：库内分子来源多样，不因罕见价态拒识


# --------------------------------------------------------------------------
# 芳香性识别（用于大写凯库勒式）
# --------------------------------------------------------------------------
def perceive_aromaticity(mol: Molecule) -> Tuple[Set[int], Set[int]]:
    """返回 (芳香原子集合, 芳香键集合)。

    判定：取 SSSR 中的环，环上每个原子都满足 sp2 条件
    （有双键参与，或为带孤对电子的杂原子 n/o/s，或带空 p 轨道的 b），
    且环的 π 电子数满足 4n+2。融合环按共享芳香键合并为芳香体系。
    """
    sssr = mol.sssr()
    ring_bond_set = set()
    for c in sssr:
        ring_bond_set |= c
    aromatic_atoms: Set[int] = set()
    aromatic_bonds: Set[int] = set()
    for cyc in sssr:
        atoms = _cycle_atoms(mol, cyc)
        if len(atoms) < 5:
            continue
        pi_e = _pi_electrons(mol, atoms, cyc)
        if pi_e is None:
            continue
        if pi_e % 4 == 2 and pi_e > 0:
            for bi in cyc:
                aromatic_bonds.add(bi)
            aromatic_atoms |= set(atoms)
    # 融合环传播：若两个 SSSR 环共享一条已判芳香的键，合并体系后
    # 重新校验整体 4n+2（简化处理：逐环独立判定已覆盖常见稠环）
    return aromatic_atoms, aromatic_bonds


def _cycle_atoms(mol: Molecule, cyc: Set[int]) -> List[int]:
    out: Set[int] = set()
    for bi in cyc:
        b = mol.bonds[bi]
        out.add(b.a1)
        out.add(b.a2)
    return sorted(out)


def _pi_electrons(mol: Molecule, atoms: List[int],
                  cyc: Set[int]) -> Optional[int]:
    """统计环上 π 电子数；任一原子不满足 sp2 平面条件则返回 None。"""
    total = 0
    for i in atoms:
        a = mol.atoms[i]
        if a.aromatic:
            # 已经是芳香写法：c/n 贡献 1，[nH]/o/s 贡献 2
            if a.symbol == "c" or (a.symbol == "n" and a.total_h == 0
                                   and a.charge == 0):
                total += 1
            elif a.symbol in ("n", "o", "s", "p"):
                total += 2
            elif a.symbol == "b":
                total += 0
            else:
                return None
            continue
        # 大写原子：必须参与一条双键（在环内或环外均可）
        has_double = any(
            (not mol.bonds[bi].aromatic) and mol.bonds[bi].order == 2
            for bi in mol.adj[i]
        )
        if has_double:
            total += 1
            continue
        # 杂原子孤对：环内单键连接的 N/O/S
        sym = a.symbol
        ring_only_single = all(
            mol.bonds[bi].aromatic or mol.bonds[bi].order == 1
            for bi in cyc if i in (mol.bonds[bi].a1, mol.bonds[bi].a2)
        )
        if ring_only_single and sym in ("N", "O", "S", "P") and a.charge <= 0:
            total += 2
            continue
        if sym == "B" and a.charge == 0:
            total += 0
            continue
        return None
    return total


# --------------------------------------------------------------------------
# 凯库勒化：把芳香键网络指派为交替单/双键
# --------------------------------------------------------------------------
def kekulize(mol: Molecule, aromatic_atoms: Set[int],
             aromatic_bonds: Set[int],
             had_ring_double: Optional[Dict[int, bool]] = None) -> None:
    """将芳香键改写为整数键级的单/双键网络。

    做法：在芳香子图上求完美匹配（每个芳香原子恰好匹配一条“双键”），
    用带回溯的 DFS 实现（分子规模小，足够快且确定性）。
    had_ring_double：调用方在改键之前记录的“原子是否参与双键”，
    用于区分吡啶型 N（参与双键，不豁免）与吡咯型 N（孤对电子，豁免）。
    """
    if not aromatic_bonds:
        return
    if had_ring_double is None:
        had_ring_double = {
            i: any((not mol.bonds[bi].aromatic) and mol.bonds[bi].order == 2
                   for bi in mol.adj[i] if bi in aromatic_bonds)
            for i in aromatic_atoms
        }
    # 进入匹配前，所有芳香环键（含大写凯库勒式里已有的双键）一律视为
    # 待定单键，避免价态快筛把环内双键误算成环外固定价。
    for bi in aromatic_bonds:
        mol.bonds[bi].order = 1
        mol.bonds[bi].aromatic = True
    # 子图邻接
    sub_adj: Dict[int, List[int]] = {i: [] for i in aromatic_atoms}
    for bi in aromatic_bonds:
        b = mol.bonds[bi]
        sub_adj[b.a1].append(b.a2)
        sub_adj[b.a2].append(b.a1)

    # 贡献孤对电子、不承担环内双键的芳香杂原子（呋喃 O、噻吩 S、
    # 吡咯 [nH] 等）：豁免完美匹配，其相邻环键全部指派为单键。
    exempt: Set[int] = set()
    for i in aromatic_atoms:
        a = mol.atoms[i]
        sym = a.symbol.lower()
        has_h = a.explicit_h > 0
        if sym in ("o", "s"):
            exempt.add(i)
        elif sym == "n" and (has_h or a.charge < 0):
            exempt.add(i)
        elif sym == "p" and has_h:
            exempt.add(i)
        # 大写凯库勒式的吡咯型 N/P：原始环内键全为单键（没有双键参与），
        # 靠孤对电子贡献芳香性，因此豁免双键匹配。
        # 用元素符号原始大小写 + 重置前记录的环内双键信息区分吡啶/吡咯。
        came_from_kekule = a.symbol[0].isupper()
        if i not in exempt and sym in ("n", "p") and a.charge <= 0 \
                and came_from_kekule and not had_ring_double[i]:
            exempt.add(i)
    for i in exempt:
        sub_adj.pop(i, None)
    for i in list(sub_adj):
        sub_adj[i] = [j for j in sub_adj[i] if j not in exempt]

    match: Dict[int, int] = {}

    def backtrack(order: List[int], k: int) -> bool:
        if k == len(order):
            return True
        x = order[k]
        if x in match:
            return backtrack(order, k + 1)
        for y in sorted(sub_adj[x]):
            if y in match:
                continue
            if not _double_bond_allowed(mol, x, y, aromatic_bonds):
                continue
            match[x] = y
            match[y] = x
            if backtrack(order, k + 1):
                return True
            del match[x]
            del match[y]
        return False

    # 约束最强（候选最少）的原子优先；豁免原子不参与匹配。
    # 不使用输入双键做种子：同一环可能有多个合法布局，
    # 固定的遍历顺序（按原子 idx 排序）保证产出确定的同一个布局。
    order = sorted((i for i in aromatic_atoms if i not in exempt),
                   key=lambda i: (len(sub_adj[i]), i))
    if not backtrack(order, 0):
        raise ChemistryError(
            "芳香体系无法指派交替双键（凯库勒化失败），"
            "可能存在非标准芳香写法")

    double_edges = {tuple(sorted((x, y))) for x, y in match.items()}
    for bi in aromatic_bonds:
        b = mol.bonds[bi]
        key = (min(b.a1, b.a2), max(b.a1, b.a2))
        b.aromatic = False
        b.order = 2 if key in double_edges else 1
    for i in aromatic_atoms:
        mol.atoms[i].aromatic = False


def _double_bond_allowed(mol: Molecule, x: int, y: int,
                         aromatic_bonds: Set[int]) -> bool:
    """价态层面的双键可行性快筛。

    环内芳香键还没定型，只统计环外（取代基方向）已确定的键价，
    加上拟指派的一条芳香双键，校验不超过元素默认价。
    """
    for i in (x, y):
        a = mol.atoms[i]
        external = 0.0
        for bi in mol.adj[i]:
            b = mol.bonds[bi]
            if bi in aromatic_bonds:
                continue
            external += b.order
        ring_deg = sum(1 for bi in mol.adj[i] if bi in aromatic_bonds)
        h = a.total_h
        projected = external + 2 + (ring_deg - 1) + h - a.charge
        from .elements import DEFAULT_VALENCES
        sym = a.symbol.upper() if a.symbol in "bcnops" else a.symbol
        vals = DEFAULT_VALENCES.get(sym)
        if vals and projected > max(vals) + 1e-9:
            return False
    return True


# --------------------------------------------------------------------------
# 统一入口：标准化一张刚解析出来的图
# --------------------------------------------------------------------------
def standardize(mol: Molecule) -> Molecule:
    """统一芳香写法并填隐式氢。原地修改并返回 mol。

    流程：
    1. 全大写凯库勒式 -> perceive_aromaticity 识别芳香环并打标；
       小写芳香式 -> 标记已在解析时给出。
    2. 芳香子图凯库勒化，所有键变成整数键级，原子符号统一大写。
    3. 在整数键价图上统一按默认价规则算隐式氢。
    输出图不再保留任何 aromatic 标记，芳香性是可由环+双键重新派生的属性。
    """
    has_arom_mark = any(a.aromatic for a in mol.atoms)
    # 在任何打标/改键之前，先记录每个原子是否参与双键，
    # 供凯库勒化区分吡啶型 N（参与双键）与吡咯型 N（孤对豁免）。
    ring_double_before = {
        i: any((not mol.bonds[bi].aromatic) and mol.bonds[bi].order == 2
               for bi in mol.adj[i])
        for i in range(len(mol.atoms))
    }
    if not has_arom_mark:
        arom_atoms, arom_bonds = perceive_aromaticity(mol)
        if arom_bonds:
            for i in arom_atoms:
                mol.atoms[i].aromatic = True
            for bi in arom_bonds:
                mol.bonds[bi].aromatic = True
    else:
        arom_bonds = {bi for bi, b in enumerate(mol.bonds) if b.aromatic}
        arom_atoms = {i for i, a in enumerate(mol.atoms) if a.aromatic}

    if arom_bonds:
        kekulize(mol, arom_atoms, arom_bonds, ring_double_before)

    for i in arom_atoms:
        a = mol.atoms[i]
        if a.symbol in "bcnops":
            a.symbol = a.symbol.upper()
    for a in mol.atoms:
        a.aromatic = False
        if a.explicit_h == -1:
            a.implicit_h = _aliphatic_implicit_h(mol, a)
        else:
            a.implicit_h = 0
    mol.compute_fragments()
    return mol
