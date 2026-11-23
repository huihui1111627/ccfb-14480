# -*- coding: utf-8 -*-
"""随机合理分子生成器（用于验收基准，不依赖外部化学库）。

策略：基于“带开放连接点的片段库”做随机组装 + 少量纯生长。
- 片段来自常见药物骨架/官能团，每个片段标注开放价点。
- 随机选片段，通过开放点以随机键级连接，每步检查不超过元素默认价，
  从而保证产出的分子价态合理、可被解析器重新解析。
- 生成后用本库自己的 parse + standardize 验证，无法通过的丢弃。
随机数全部来自外部注入的 random.Random，保证可复现。
"""
from __future__ import annotations

import random
from typing import List, Optional, Tuple

from .aromatic import standardize
from .canonical import canonicalize
from .parser import parse_smiles

# 常用环状骨架（自带芳香/脂肪环），连接点用取代位置体现
RING_FRAGMENTS = [
    "c1ccccc1",        # 苯
    "C1CCCCC1",        # 环己烷
    "c1ccncc1",        # 吡啶
    "c1ccoc1",         # 呋喃
    "c1cc[nH]c1",      # 吡咯
    "c1ccsc1",         # 噻吩
    "c1ccc2ccccc2c1",  # 萘
    "c1cnc2ncnc(N)c2n1",  # 嘌呤类（容错，失败会被丢弃）
    "C1CCCC1",         # 环戊烷
    "c1ccc2[nH]ccc2c1",   # 吲哚类
    "C1CCNCC1",        # 哌啶
    "c1ccoc1cc",       # 取代呋喃（容错）
]

# 官能团/侧链片段（写成带一个连接点的形式，连接点用前导原子体现）
SIDE_FRAGMENTS = [
    "C", "CC", "CCC", "C(C)C", "O", "N", "F", "Cl", "Br",
    "C#N", "C=O", "C(=O)O", "C(=O)N", "NO", "N(=O)=O",
    "OC", "NC", "SC", "C(F)(F)F", "CCl", "CC(=O)O",
    "C(=O)OC", "N(C)C", "OCC", "CCS",
]


def _random_smiles(rng: random.Random, max_ring_frags: int = 2) -> str:
    """组装一个随机 SMILES。

    以单环骨架为主体，在环原子的取代位上挂侧链；多环通过连接子偶联。
    侧链优先挂在环上（环原子有开放价），降低价态冲突导致的非法率。
    """
    n_rings = rng.randint(1, max_ring_frags)
    base = rng.choice(RING_FRAGMENTS)
    # 环之间通过短连接子偶联
    for _ in range(n_rings - 1):
        linker = rng.choice(["C", "CC", "O", "N"])
        base = f"{base}{linker}({rng.choice(RING_FRAGMENTS)})"
    # 在环上挂随机数目的侧链（括号取代，价态由验证兜底）
    n_side = rng.randint(1, 4)
    for _ in range(n_side):
        frag = rng.choice(SIDE_FRAGMENTS)
        base = f"{base}({frag})"
    if rng.random() < 0.04:
        base += "." + rng.choice(["[Na+]", "[Cl-]", "[K+]"])
    return base


def generate(rng: random.Random, count: int,
             progress_every: int = 0) -> Tuple[List[str], int]:
    """生成 count 个通过本库解析+标准化验证的 SMILES。

    返回 (有效写法列表, 被丢弃的候选数)。
    """
    out: List[str] = []
    rejected = 0
    attempts = 0
    while len(out) < count:
        attempts += 1
        s = _random_smiles(rng)
        if not _is_valid(s):
            rejected += 1
            continue
        out.append(s)
        if progress_every and len(out) % progress_every == 0:
            print(f"  生成 {len(out)}/{count}（尝试 {attempts}，丢弃 {rejected}）")
    return out, rejected


def _is_valid(smiles: str) -> bool:
    try:
        mol = parse_smiles(smiles)
        standardize(mol)
        canonicalize(mol)
        return len(mol.atoms) >= 2
    except Exception:
        return False


def generate_distinct(rng: random.Random, count: int,
                      progress_every: int = 0) -> Tuple[List[str], int, int]:
    """生成 count 个规范结构互不相同的分子。

    返回 (SMILES 列表, 去重丢弃数, 非法丢弃数)。
    """
    seen = set()
    out: List[str] = []
    dup = 0
    invalid = 0
    attempts = 0
    while len(out) < count:
        attempts += 1
        if attempts > count * 60:
            break  # 片段空间耗尽保护
        s = _random_smiles(rng)
        try:
            mol = canonicalize(standardize(parse_smiles(s)))
        except Exception:
            invalid += 1
            continue
        from .canonical import canonical_hash
        h = canonical_hash(mol)
        if h in seen:
            dup += 1
            continue
        seen.add(h)
        out.append(s)
        if progress_every and len(out) % progress_every == 0:
            print(f"  去重生成 {len(out)}/{count}")
    return out, dup, invalid
