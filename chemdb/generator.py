"""随机“合理”分子生成器（用于验收/压测，不依赖任何化学信息学库）。

策略：从真实环系模板出发，直接在分子图上增量挂接侧链、用连接基拼接环系，
最后只做一次芳香归一与规范化。全过程不反复解析 SMILES，保证吞吐。
输出后用自家 parser 重新解析 + 价键校验，只保留合法分子。
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional

from .parser import parse_smiles, ParseError
from .aromaticity import AromaticityError
from .canonical import canonical_smiles
from .valence import check_valence, compute_implicit_h
from .rings import mark_ring_bonds
from .aromaticity import normalize_aromaticity
from .molecule import Atom, Mol


RING_TEMPLATES = [
    "c1ccccc1", "c1ccncc1", "c1ccoc1", "c1ccsc1", "c1cc[nH]c1",
    "c1cnc[nH]1", "c1ccncn1", "c1ccc2ccccc2c1", "c1ccc2[nH]ccc2c1",
    "c1ccc2ncccc2c1", "c1ccc2cc3ccccc3cc2c1",
    "C1CCCCC1", "C1CCCC1", "C1CCNCC1", "C1CCOCC1",
]

# 侧链片段（去掉连接 * 后的 SMILES，首原子即连接点）
SIDE_CHAINS = [
    "C", "CC", "CCC", "C(C)C", "CC(C)C", "O", "OC", "OCC", "N", "NC",
    "N(C)C", "C(=O)O", "C(=O)OC", "C(=O)N", "C(=O)NC", "F", "Cl",
    "Br", "I", "C#N", "SC", "S(=O)(=O)C", "OC(=O)C", "NC(=O)C",
]

LINKERS = ["C", "CC", "CCC", "O", "N", "CC(C)C", "OC", "NC"]


def _clone_atom_into(src: Atom, mol: Mol) -> int:
    return mol.add_atom(Atom(
        0, src.symbol, charge=src.charge, isotope=src.isotope,
        explicit_h=src.explicit_h, bracket=src.bracket,
        aromatic=src.aromatic, wildcard=src.wildcard,
        aromatic_query=src.aromatic_query,
        aliphatic_query=src.aliphatic_query))


def _append_fragment(mol: Mol, anchor: int, frag_smiles: str) -> Optional[int]:
    """把侧链片段的首原子连到 anchor，返回片段首原子下标。"""
    try:
        frag = parse_smiles(frag_smiles)
    except ParseError:
        return None
    remap: Dict[int, int] = {}
    for fa in frag.atoms:
        remap[fa.index] = _clone_atom_into(fa, mol)
    for fb in frag.bonds:
        mol.add_bond(remap[fb.a1], remap[fb.a2], fb.order)
    head = remap[0]
    # 键级：芳香环上的连接默认单键
    mol.add_bond(anchor, head, 1)
    return head


def _aromatic_carbon_points(mol: Mol) -> List[int]:
    return [a.index for a in mol.atoms
            if a.aromatic and a.element == "C" and mol.degree(a.index) == 2]


def _pick_attach(mol: Mol, rng: random.Random) -> int:
    pts = _aromatic_carbon_points(mol)
    if pts:
        return rng.choice(pts)
    cands = [a.index for a in mol.atoms if mol.degree(a.index) <= 3]
    return rng.choice(cands)


def _fuse_ring(base: Mol, ring_smiles: str, linker: str,
               rng: random.Random) -> bool:
    """用连接基把一个新环系接到 base 上，全部在图上增量完成。"""
    try:
        ring = parse_smiles(ring_smiles)
        link = parse_smiles(linker)
    except ParseError:
        return False
    anchor_a = _pick_attach(base, rng)
    anchor_b = _pick_attach(ring, rng)
    # 复制 ring
    rmap: Dict[int, int] = {}
    for ra in ring.atoms:
        rmap[ra.index] = _clone_atom_into(ra, base)
    for rb in ring.bonds:
        base.add_bond(rmap[rb.a1], rmap[rb.a2], rb.order)
    # 复制 linker 并串联 anchor_a - linker - ring.anchor_b
    lmap: Dict[int, int] = {}
    for la in link.atoms:
        lmap[la.index] = _clone_atom_into(la, base)
    for lb in link.bonds:
        base.add_bond(lmap[lb.a1], lmap[lb.a2], lb.order)
    base.add_bond(anchor_a, lmap[0], 1)
    base.add_bond(lmap[link.n_atoms() - 1], rmap[anchor_b], 1)
    return True


def _finalize(mol: Mol) -> Optional[str]:
    """归一 + 价键校验 + 规范化，返回规范 SMILES；不合法返回 None。"""
    try:
        mark_ring_bonds(mol)
        normalize_aromaticity(mol)
        compute_implicit_h(mol)
        if check_valence(mol):
            return None
        canon = canonical_smiles(mol)
        # 回环验证：规范串必须能被自家 parser 重新解析
        reparsed = parse_smiles(canon)
        if canonical_smiles(reparsed) != canon:
            return None
        return canon
    except (ParseError, AromaticityError, ValueError, KeyError):
        return None


def generate_molecule(rng: random.Random,
                      max_rings: int = 3,
                      max_sidechains: int = 5) -> Optional[str]:
    try:
        mol = parse_smiles(rng.choice(RING_TEMPLATES))
        n_rings = rng.randint(1, max_rings)
        for _ in range(n_rings - 1):
            if not _fuse_ring(mol, rng.choice(RING_TEMPLATES),
                              rng.choice(LINKERS), rng):
                return None
        # 在所有可用芳香碳上随机挂侧链
        points = _aromatic_carbon_points(mol)
        rng.shuffle(points)
        n_sub = rng.randint(0, min(max_sidechains, len(points)))
        for ai in points[:n_sub]:
            _append_fragment(mol, ai, rng.choice(SIDE_CHAINS))
        return _finalize(mol)
    except (ParseError, AromaticityError, ValueError, KeyError):
        return None


def random_writings(canon: str, rng: random.Random) -> List[str]:
    """给定规范 SMILES，产出同一分子的多种等价写法。"""
    mol = parse_smiles(canon)
    out = {canon}
    try:
        from .aromaticity import kekulize
        kmol = kekulize(mol)
        out.add(canonical_smiles(kmol, aromatic=False))
    except AromaticityError:
        pass
    for _ in range(3):
        out.add(_random_dfs_smiles(mol, rng))
    return list(out)


def _random_dfs_smiles(mol: Mol, rng: random.Random) -> str:
    start = rng.randrange(mol.n_atoms())
    ring_no: Dict[tuple, int] = {}
    counter = [0]
    color: Dict[int, int] = {start: 1}
    parent: Dict[int, Optional[int]] = {start: None}

    def rkey(u, v):
        return (u, v) if u < v else (v, u)

    def build(ai):
        nbrs = mol.neighbors(ai)
        rng.shuffle(nbrs)
        kids, backs = [], []
        for v in nbrs:
            if v == parent[ai]:
                continue
            cv = color.get(v, 0)
            if cv == 0:
                parent[v] = ai
                color[v] = 1
                kids.append(v)
                build(v)
            elif cv == 1:
                backs.append(v)
                k = rkey(ai, v)
                if k not in ring_no:
                    counter[0] += 1
                    ring_no[k] = counter[0]
        color[ai] = 2
        return kids, backs

    tree: Dict[int, List[int]] = {}
    backs_map: Dict[int, List[int]] = {}
    # 先建树
    stack = [start]
    color2 = {start: 1}
    par2 = {start: None}
    while stack:
        x = stack.pop()
        nbrs = mol.neighbors(x)
        rng.shuffle(nbrs)
        kids, backs = [], []
        for v in nbrs:
            if v == par2[x]:
                continue
            if v not in color2:
                par2[v] = x
                color2[v] = 1
                kids.append(v)
                stack.append(v)
            elif color2[v] == 1:
                backs.append(v)
                k = rkey(x, v)
                if k not in ring_no:
                    counter[0] += 1
                    ring_no[k] = counter[0]
        tree[x] = kids
        backs_map[x] = backs
        color2[x] = 2

    from .canonical import _atom_text, _bond_text, _ring_digit, _depth
    # 深度（基于 parent）
    depth: Dict[int, int] = {}

    def ddepth(x):
        if x in depth:
            return depth[x]
        depth[x] = 0 if par2[x] is None else ddepth(par2[x]) + 1
        return depth[x]

    open_at: Dict[int, List[str]] = {}
    close_at: Dict[int, List[str]] = {}
    for (u, v), no in ring_no.items():
        du, dv = ddepth(u), ddepth(v)
        shallow, deep = (u, v) if du < dv else (v, u)
        b = mol.bonds[mol.bond_index_between(u, v)]
        bt = _bond_text(mol, u, v, b.order)
        d = _ring_digit(no)
        open_at.setdefault(shallow, []).append(bt + d)
        close_at.setdefault(deep, []).append(d)

    def emit(ai) -> str:
        parts = [_atom_text(mol, ai)]
        parts.extend(open_at.get(ai, []))
        parts.extend(close_at.get(ai, []))
        kids = tree[ai]
        for k, v in enumerate(kids):
            b = mol.bonds[mol.bond_index_between(ai, v)]
            tok = _bond_text(mol, ai, v, b.order) + emit(v)
            if len(kids) > 1 and k < len(kids) - 1:
                parts.append("(" + tok + ")")
            else:
                parts.append(tok)
        return "".join(parts)

    return emit(start)


def generate_dataset(n: int, seed: int = 42) -> List[str]:
    rng = random.Random(seed)
    out: List[str] = []
    seen = set()
    tries = 0
    while len(out) < n and tries < n * 30:
        tries += 1
        smi = generate_molecule(rng)
        if smi and smi not in seen:
            seen.add(smi)
            out.append(smi)
    return out
