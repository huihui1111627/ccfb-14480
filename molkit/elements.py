# -*- coding: utf-8 -*-
"""元素周期表基础数据（原子序数、默认价态、元素符号）。

不依赖任何第三方库，只放纯数据与查表函数。
"""
from __future__ import annotations

# 原子序数 -> 元素符号
SYMBOLS = [
    "#",  # 占位，使下标等于原子序数
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy",
    "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt",
    "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn",
    "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf",
    "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
]

SYMBOL_TO_Z = {s: i for i, s in enumerate(SYMBOLS) if i > 0}

# 各元素常见（默认）价态列表，用于隐式氢计数。
# 参考 Daylight SMILES 的 default valence 表。
DEFAULT_VALENCES = {
    "H": [1], "He": [0],
    "Li": [1], "Be": [2], "B": [3], "C": [4], "N": [3, 5], "O": [2],
    "F": [1], "Ne": [0],
    "Na": [1], "Mg": [2], "Al": [3], "Si": [4], "P": [3, 5],
    "S": [2, 4, 6], "Cl": [1, 3, 5, 7], "Ar": [0],
    "K": [1], "Ca": [2], "Sc": [3], "Ti": [4], "V": [5], "Cr": [3, 6],
    "Mn": [2, 4, 7], "Fe": [2, 3, 6], "Co": [2, 3], "Ni": [2, 3],
    "Cu": [1, 2], "Zn": [2],
    "Ga": [3], "Ge": [4], "As": [3, 5], "Se": [2, 4, 6],
    "Br": [1, 3, 5, 7], "Kr": [0],
    "Rb": [1], "Sr": [2], "Y": [3], "Zr": [4], "Nb": [5], "Mo": [4, 6],
    "Ru": [2, 3, 4], "Rh": [3], "Pd": [2, 4], "Ag": [1], "Cd": [2],
    "In": [3], "Sn": [2, 4], "Sb": [3, 5], "Te": [2, 4, 6],
    "I": [1, 3, 5, 7], "Xe": [0],
    "Cs": [1], "Ba": [2],
    "W": [6], "Pt": [2, 4], "Au": [1, 3], "Hg": [2],
    "Tl": [1, 3], "Pb": [2, 4], "Bi": [3, 5],
}

# 不带括号的有机子集原子（SMILES 裸写白名单）
ORGANIC_SUBSET = {
    "B", "C", "N", "O", "P", "S", "F", "Cl", "Br", "I",
    # 芳香小写形式在解析处单独处理
}
AROMATIC_ORGANIC = {"b", "c", "n", "o", "p", "s"}

# 元素的典型杂化/芳香参与信息（最小集合，够用即可）
HALOGENS = {"F", "Cl", "Br", "I"}

# 芳香原子允许的价态辅助：芳香杂环 n/o/s/p/b 的键价折算
# 芳香键在算价时按 1.5 处理，这里用 (双键数, 单键数) 的整数规则代替。


def atomic_number(symbol: str) -> int:
    return SYMBOL_TO_Z[symbol]


def default_valences(symbol: str):
    return DEFAULT_VALENCES.get(symbol)


def is_metal(symbol: str) -> bool:
    z = SYMBOL_TO_Z.get(symbol)
    if z is None:
        return False
    # 简化的金属判断：碱/碱土/过渡/后过渡金属
    metal_zs = set()
    for a, b in [(3, 4), (11, 13), (19, 32), (37, 51), (55, 84), (87, 116)]:
        metal_zs.update(range(a, b + 1))
    # 去掉类非金属
    metal_zs.difference_update({5, 14, 15, 16, 33, 34, 52, 53, 85})
    return z in metal_zs
