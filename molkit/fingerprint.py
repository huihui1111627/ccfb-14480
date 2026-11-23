# -*- coding: utf-8 -*-
"""结构指纹：路径指纹 + 环境指纹，用于子结构预筛与相似性。

核心性质（需求 4 的不漏筛保证）：
  对片段 Q 与分子 M，若 Q 是 M 的子结构，则 Q 中出现的每一条带原子/键
  属性的路径，都能在 M 的某次匹配覆盖的顶点上找到同属性路径，
  因此 hash(Q 的路径) ⊆ hash(M 的路径)，即指纹位集合满足子集关系。
  预筛只淘汰不满足子集关系的分子，保证不漏。

指纹用“整数集合”表示（位号空间取模到固定长度后可压缩成位串）。
哈希全部用 hashlib，确定性且无第三方依赖。
"""
from __future__ import annotations

import hashlib
from typing import Dict, FrozenSet, List, Set, Tuple

from .graph import Molecule

FP_SIZE = 1024          # 位串长度（取模空间）
MAX_PATH_LEN = 7        # 路径最大边数（0..MAX_PATH_LEN）
ENV_RADIUS = 2          # 原子环境（圆形指纹）半径


def _hash_int(text: str) -> int:
    h = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big")


# --------------------------------------------------------------------------
# 原子/键的属性 token（化学意义在指纹里保留）
# --------------------------------------------------------------------------
def _aromatic_lookup(mol: Molecule) -> Set[int]:
    """返回该图当前的芳香原子集合。

    查询片段保留了解析/感知出的芳香标记；库内目标是凯库勒化整数键级图，
    这里按需重新感知，使两者的芳香 token 落在同一空间，保证子集性。
    """
    marked = {i for i, a in enumerate(mol.atoms) if a.aromatic}
    if marked:
        return marked
    from .aromatic import perceive_aromaticity
    aa, _ = perceive_aromaticity(mol)
    return aa


def atom_token(mol: Molecule, i: int, arom_set: Set[int] = None) -> str:
    """指纹原子 token：只保留子图匹配下不变的内禀属性。

    度、隐氢数、是否环内都会因片段嵌入目标而改变，不能进 token，
    否则片段指纹位无法成为目标指纹位的子集（漏筛）。
    环信息由路径/环境的拓扑结构本身表达。
    """
    a = mol.atoms[i]
    if a.is_wildcard:
        return "*"
    # 芳香小写与凯库勒大写统一为大写元素符号；芳香性单独由 ar 位表达
    sym = a.symbol.upper() if a.symbol in "bcnops" else a.symbol
    is_arom = 1 if ((arom_set is not None and i in arom_set)
                    or a.aromatic) else 0
    return f"{sym}(q{a.charge},i{a.isotope},ar{is_arom})"


def bond_token(b, aromatic_bonds: Set[int] = None, bond_idx: int = -1) -> str:
    # 芳香键：查询里显式标 aromatic；目标里落在感知出的芳香环键集合内
    if b.aromatic or (aromatic_bonds is not None
                      and bond_idx in aromatic_bonds):
        return ":"
    if b.is_wildcard:
        return "~"
    return {1: "1", 2: "2", 3: "3"}.get(b.order, "1")


# --------------------------------------------------------------------------
# 路径枚举（非回溯简单路径，DFS）
# --------------------------------------------------------------------------
def enumerate_paths(mol: Molecule, max_len: int = MAX_PATH_LEN
                    ) -> Set[str]:
    """枚举所有长度 0..max_len 的简单路径，返回路径签名集合。

    每条路径签名是规范化后的字符串，使不同遍历方向得到同一项。
    """
    arom_atoms = _aromatic_lookup(mol)
    from .aromatic import perceive_aromaticity
    _, arom_bonds = perceive_aromaticity(mol)
    arom_bonds = arom_bonds | {bi for bi, b in enumerate(mol.bonds)
                               if b.aromatic}
    out: Set[str] = set()
    n = len(mol.atoms)
    for start in range(n):
        _dfs_path(mol, start, start, {start},
                  [atom_token(mol, start, arom_atoms)],
                  max_len, out, arom_atoms, arom_bonds)
    return out


def _dfs_path(mol: Molecule, start: int, cur: int, visited: Set[int],
              tokens: List[str], max_len: int, out: Set[str],
              arom_atoms: Set[int], arom_bonds: Set[int]) -> None:
    # 当前路径入指纹（两个方向取字典序最小，保证无向路径唯一）
    out.add(_canonical_path_string(tokens))
    if len(tokens) // 2 >= max_len:
        return
    for bi in mol.adj[cur]:
        b = mol.bonds[bi]
        nxt = b.other(cur)
        if nxt in visited:
            continue
        visited.add(nxt)
        new_tokens = tokens + [
            bond_token(b, arom_bonds, bi),
            atom_token(mol, nxt, arom_atoms)]
        _dfs_path(mol, start, nxt, visited, new_tokens, max_len, out,
                  arom_atoms, arom_bonds)
        visited.discard(nxt)


def _canonical_path_string(tokens: List[str]) -> str:
    """路径的无向规范字符串：正反两个方向取较小者。"""
    forward = " ".join(tokens)
    reverse = " ".join(reversed(tokens))
    return min(forward, reverse)


# --------------------------------------------------------------------------
# 圆形环境指纹（补充路径指纹对分支环境的表达）
# --------------------------------------------------------------------------
def environment_features(mol: Molecule, radius: int = ENV_RADIUS
                         ) -> Set[str]:
    """每个原子在半径 0..radius 内的逐层环境签名。"""
    arom_atoms = _aromatic_lookup(mol)
    from .aromatic import perceive_aromaticity
    _, arom_bonds = perceive_aromaticity(mol)
    arom_bonds = arom_bonds | {bi for bi, b in enumerate(mol.bonds)
                               if b.aromatic}
    features: Set[str] = set()
    # 第 0 层：原子自身
    current: Dict[int, str] = {i: atom_token(mol, i, arom_atoms)
                               for i in range(len(mol.atoms))}
    for i in range(len(mol.atoms)):
        features.add(f"env0:{current[i]}")
    for r in range(1, radius + 1):
        nxt: Dict[int, str] = {}
        for i in range(len(mol.atoms)):
            neigh = []
            for bi in mol.adj[i]:
                b = mol.bonds[bi]
                j = b.other(i)
                neigh.append(
                    f"{bond_token(b, arom_bonds, bi)}>{current[j]}")
            sig = current[i] + "[" + ",".join(sorted(neigh)) + "]"
            nxt[i] = sig
            features.add(f"env{r}:{sig}")
        current = nxt
    return features


# --------------------------------------------------------------------------
# 指纹组装
# --------------------------------------------------------------------------
def fingerprint_set(mol: Molecule) -> Set[int]:
    """路径指纹的整数位号集合。

    子集判断只依赖路径指纹（环境指纹含度/环，不具子集性，单独存放）。
    """
    return {_hash_int("path:" + sig) for sig in enumerate_paths(mol)}


def environment_set(mol: Molecule) -> Set[int]:
    """圆形环境指纹位号，仅用于相似性，不参与子集预筛。"""
    return {_hash_int("env:" + sig)
            for sig in environment_features(mol)}


def fingerprint_bits(mol: Molecule, size: int = FP_SIZE) -> Set[int]:
    """取模到 [0,size) 的路径指纹位号集合，用于位串/Tanimoto/子集预筛。"""
    return {h % size for h in fingerprint_set(mol)}


def concrete_query_bits(mol: Molecule, size: int = FP_SIZE) -> Set[int]:
    """查询专用：只保留“完全由确定原子与确定键构成”的路径位。

    通配原子 * 能匹配任意元素、通配键 ~ 能匹配任意键，含通配的路径
    无法对目标产生可排除的约束；若把它们也算进子集判断，反而会误杀
    （查询位在目标中根本不存在）。这里剔除所有含通配的路径，
    剩下的确定路径仍满足子集性，保证不漏筛。
    """
    bits: Set[int] = set()
    for sig in enumerate_paths(mol):
        if "*" in sig or "~" in sig:
            continue
        bits.add(_hash_int("path:" + sig) % size)
    return bits


def bitstring(mol: Molecule, size: int = FP_SIZE) -> str:
    bits = fingerprint_bits(mol, size)
    return "".join("1" if i in bits else "0" for i in range(size))


def popcount(mol: Molecule, size: int = FP_SIZE) -> int:
    return len(fingerprint_bits(mol, size))


# --------------------------------------------------------------------------
# 子集预筛与 Tanimoto
# --------------------------------------------------------------------------
def is_subset_fp(query_bits: Set[int], target_bits: Set[int]) -> bool:
    """query 的位是否全部出现在 target 中（不漏筛的核心判断）。"""
    return query_bits.issubset(target_bits)


def tanimoto(bits_a: Set[int], bits_b: Set[int]) -> float:
    if not bits_a and not bits_b:
        return 1.0
    inter = len(bits_a & bits_b)
    union = len(bits_a | bits_b)
    return inter / union if union else 0.0


def tanimoto_bounds(query_pop: int, target_pop: int) -> Tuple[float, float]:
    """给定两个指纹的位数（popcount），Tanimoto 的上下界。

    交集最大为 min(qa, qb)，最小为 max(0, qa+qb-SIZE)。
    用于在不计算交集的情况下，用位数快速跳过不可能达阈值的分子。
    """
    if query_pop == 0 and target_pop == 0:
        return 1.0, 1.0
    max_inter = min(query_pop, target_pop)
    union_lo = max(query_pop, target_pop)          # 交集最大时并集最小
    upper = max_inter / union_lo if union_lo else 0.0
    min_inter = max(0, query_pop + target_pop - FP_SIZE)
    union_hi = query_pop + target_pop - min_inter
    lower = min_inter / union_hi if union_hi else 0.0
    return lower, upper


def could_meet_threshold(query_pop: int, target_pop: int,
                         threshold: float) -> bool:
    """仅用位数判断该分子是否可能达到 Tanimoto 阈值（上界 >= 阈值）。"""
    _, upper = tanimoto_bounds(query_pop, target_pop)
    return upper >= threshold - 1e-12
