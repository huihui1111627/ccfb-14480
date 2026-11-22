"""价键规则：隐式氢计算与价键校验。"""
from __future__ import annotations

from typing import Dict, List

from .elements import AROMATIC_TO_ALIPHATIC, DEFAULT_VALENCES
from .molecule import Mol

# 带电荷时的目标连接数（覆盖 SMILES 中最常见的情况）
CHARGED_TARGET: Dict[str, Dict[int, int]] = {
    "N": {1: 4, -1: 2},
    "O": {1: 3, -1: 1},
    "C": {1: 3, -1: 3},
    "S": {1: 3, -1: 1},
    "P": {1: 4, -1: 2},
}


def bond_order_sum(mol: Mol, ai: int, skip_aromatic: bool = False) -> int:
    total = 0
    for bi in mol.adj[ai]:
        b = mol.bonds[bi]
        if skip_aromatic and b.order == 4:
            continue
        total += b.order
    return total


def _aromatic_target(mol: Mol, ai: int) -> int:
    """芳香原子参与芳香体系的目标连接数（键级和，按凯库勒式计）。"""
    a = mol.atoms[ai]
    el = a.element
    charge = a.charge
    aromatic_bonds = [mol.bonds[bi] for bi in mol.adj[ai]
                      if mol.bonds[bi].order == 4]
    n_ar = len(aromatic_bonds)
    exo = sum(b.order for b in (mol.bonds[bi] for bi in mol.adj[ai])
              if b.order != 4)
    if el == "C":
        base = 4
    elif el == "N":
        if charge == 1:
            base = 4
        elif charge == -1:
            base = 2
        else:
            # 吡啶型 n(2 芳香邻居) 目标3；吡咯型 [nH] 目标3
            base = 3
    elif el in ("O", "S", "Se"):
        if charge == 1:
            base = 3
        else:
            base = 2
    elif el == "P":
        base = 3 if charge == 0 else (4 if charge == 1 else 2)
    elif el == "B":
        base = 3 if charge == 0 else 2
    else:
        base = 3
    # 由芳香键贡献：每个芳香邻居按 1.5 等效 → 整体上 2*n_ar + 双键数，
    # 这里用目标减外连键级的方式给出需要的隐式氢/显式氢空间。
    return base, n_ar, exo


def compute_implicit_h(mol: Mol) -> List[int]:
    """计算每个原子的隐式氢数量，并写入 mol._implicit_h。

    规则要点：
      * 方括号原子默认无隐式氢（[C] 即碳自由基价态，按 SMILES 规范）；
      * 芳香原子按芳香价键模型处理；
      * 有机子集裸原子按默认价表补氢。
    """
    implicit = [0] * mol.n_atoms()
    for a in mol.atoms:
        ai = a.index
        if a.wildcard or a.aromatic_query or a.aliphatic_query:
            continue
        if a.aromatic:
            implicit[ai] = _aromatic_h(mol, ai)
            continue
        if a.bracket:
            implicit[ai] = 0
            continue
        el = a.element
        orders = bond_order_sum(mol, ai)
        target = _target_valence(el, a.charge)
        if target is None:
            implicit[ai] = 0
            continue
        h = target - orders
        implicit[ai] = max(h, 0)
    mol._implicit_h = implicit
    return implicit


def _target_valence(el: str, charge: int):
    if el in CHARGED_TARGET and charge in CHARGED_TARGET[el]:
        return CHARGED_TARGET[el][charge]
    vals = DEFAULT_VALENCES.get(el)
    if not vals:
        return None
    return vals[0]


def _aromatic_h(mol: Mol, ai: int) -> int:
    a = mol.atoms[ai]
    el = a.element
    base, n_ar, exo = _aromatic_target(mol, ai)
    # 芳香键的“实际连接贡献”按凯库勒化后介于 n_ar 与 n_ar+1 之间；
    # 隐式氢 = 目标价 - 外连键级 - (芳香键最高可能贡献)
    aromatic_contribution_max = n_ar + 1 if n_ar > 0 else 0
    h = base - exo - aromatic_contribution_max - a.explicit_h
    return max(h, 0)


def check_valence(mol: Mol) -> List[str]:
    """返回价键告警列表（空列表表示全部在常见价态内）。"""
    warnings: List[str] = []
    for a in mol.atoms:
        if a.wildcard or a.aromatic_query or a.aliphatic_query:
            continue
        if a.aromatic:
            continue  # 芳香体系由 kekulize 校验
        el = a.element
        allowed = DEFAULT_VALENCES.get(el)
        if not allowed:
            continue
        orders = bond_order_sum(mol, a.index)
        h = mol.total_h(a.index)
        used = orders + h
        if el in CHARGED_TARGET and a.charge in CHARGED_TARGET[el]:
            if used != CHARGED_TARGET[el][a.charge]:
                warnings.append("原子 %d(%s) 价键数 %d 与电荷 %+d 不符"
                                % (a.index, el, used, a.charge))
        elif used not in allowed:
            warnings.append("原子 %d(%s) 的价键数 %d 不在常见价态 %s 内"
                            % (a.index, el, used, allowed))
    return warnings
