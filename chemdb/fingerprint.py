"""路径哈希结构指纹。

枚举分子中长度 1..max_len 的所有简单路径，把每条路径的规范字符串
（原子标签 + 键级序列，双向取字典序较小者）哈希成指纹位。
位向量用一个 Python int 承载，天然支持位运算与 bit_count。

子集性质（筛选不漏的保证）：
  若片段 P 是分子 M 的子图，则 P 的每条具体路径都对应 M 中的一条路径，
  规范化字符串相同、哈希位相同，因此 bits(P) ⊆ bits(M)。
  含通配原子/任意键的查询路径不参与哈希（无法保证特征具体），
  只让具体路径贡献位，子集关系依然成立。
"""
from __future__ import annotations

from typing import Dict, List, Set

from .molecule import Mol

MASK64 = (1 << 64) - 1


def popcount(x: int) -> int:
    """整数置位数（兼容 Python 3.9，3.10+ 才有 int.bit_count）。"""
    return bin(x).count("1")


def _hash64(data: str, seed: int) -> int:
    """FNV-1a 64 位，seed 用于双哈希。"""
    h = (1469598103934665603 ^ seed) & MASK64
    for ch in data.encode("utf-8"):
        h ^= ch
        h = (h * 1099511628211) & MASK64
    return h


def _atom_token(mol: Mol, ai: int) -> str:
    a = mol.atoms[ai]
    if a.wildcard or a.aromatic_query or a.aliphatic_query:
        return "*"  # 查询原子：路径不具体
    return a.label()


_BOND_TOK = {1: "-", 2: "=", 3: "#", 4: ":", 0: "~"}


def _bond_token(order: int) -> str:
    return _BOND_TOK.get(order, "?")


def path_features(mol: Mol, max_len: int = 7) -> Set[str]:
    """枚举所有简单路径（以边数计 1..max_len）的规范特征串。"""
    features: Set[str] = set()
    n = mol.n_atoms()
    # 预计算原子标签与邻接（邻居下标, 键级），避免内层重复查表
    atom_tok = [_atom_token(mol, i) for i in range(n)]
    adjacency = []
    for ai in range(n):
        lst = []
        for bi in mol.adj[ai]:
            b = mol.bonds[bi]
            nxt = b.a2 if b.a1 == ai else b.a1
            lst.append((nxt, b.order))
        adjacency.append(lst)

    for ai in range(n):
        if atom_tok[ai] != "*":
            features.add("L0:" + atom_tok[ai])

    def walk(ai: int, visited: int, fwd: str, rev: str, depth: int) -> None:
        if depth >= max_len:
            return
        for nxt, order in adjacency[ai]:
            if order == 0 or (visited >> nxt) & 1:
                continue
            atok = atom_tok[nxt]
            if atok == "*":
                continue
            bt = _BOND_TOK.get(order, "?")
            nfwd = fwd + bt + atok
            nrev = atok + bt + rev
            features.add("L%d:%s" % (depth + 1,
                                     nfwd if nfwd < nrev else nrev))
            walk(nxt, visited | (1 << nxt), nfwd, nrev, depth + 1)

    for start in range(n):
        if atom_tok[start] != "*":
            walk(start, 1 << start, atom_tok[start],
                 atom_tok[start], 0)
    return features


def fingerprint(mol: Mol, n_bits: int = 1024,
                max_len: int = 7) -> int:
    bits = 0
    for feat in path_features(mol, max_len):
        h1 = _hash64(feat, 0)
        h2 = _hash64(feat, 0x9E3779B97F4A7C15)
        bits |= 1 << (h1 % n_bits)
        bits |= 1 << (h2 % n_bits)
    return bits


def bit_positions(mol: Mol, n_bits: int = 1024,
                  max_len: int = 7) -> List[int]:
    fp = fingerprint(mol, n_bits, max_len)
    out: List[int] = []
    while fp:
        low = fp & -fp
        out.append(low.bit_length() - 1)
        fp ^= low
    return out


def tanimoto(fp1: int, fp2: int) -> float:
    inter = popcount(fp1 & fp2)
    union = popcount(fp1 | fp2)
    return inter / union if union else 0.0
