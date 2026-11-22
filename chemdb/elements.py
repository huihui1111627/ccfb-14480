"""元素周期表与价键规则（纯标准库，内置数据，无外部依赖）。"""
from __future__ import annotations

# 原子序数
ATOMIC_NUMBER = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9,
    "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16,
    "Cl": 17, "Ar": 18, "K": 19, "Ca": 20, "Br": 35, "I": 53,
    "Fe": 26, "Cu": 29, "Zn": 30, "Se": 34, "Sn": 50, "As": 33,
}
SYMBOL_OF_NUMBER = {v: k for k, v in ATOMIC_NUMBER.items()}

# SMILES 有机子集：不写方括号即可使用的元素（脂肪形式）
ORGANIC_SUBSET = {"B", "C", "N", "O", "P", "S", "F", "Cl", "Br", "I"}

# 常见元素允许的价态（按 OpenSMILES 常见惯例，用于隐式氢与价键校验）
DEFAULT_VALENCES = {
    "B": [3], "C": [4], "N": [3, 5], "O": [2], "P": [3, 5],
    "S": [2, 4, 6], "F": [1], "Cl": [1], "Br": [1], "I": [1],
    "Si": [4], "Se": [2, 4, 6], "H": [1],
}

# 芳香小写原子
AROMATIC_SYMBOLS = {"b", "c", "n", "o", "p", "s", "se"}
AROMATIC_TO_ALIPHATIC = {"b": "B", "c": "C", "n": "N", "o": "O",
                         "p": "P", "s": "S", "se": "Se"}

# 可芳香化的元素（脂肪写法下也能被芳香感知识别）
AROMATIZABLE = {"B", "C", "N", "O", "P", "S", "Se"}


def element_of(symbol: str) -> str:
    """统一成大写元素符号。"""
    if symbol in AROMATIC_SYMBOLS:
        return AROMATIC_TO_ALIPHATIC[symbol]
    return symbol


def atomic_number(symbol: str) -> int:
    sym = element_of(symbol)
    return ATOMIC_NUMBER.get(sym, 0)


def is_aromatic_symbol(symbol: str) -> bool:
    return symbol in AROMATIC_SYMBOLS
