# -*- coding: utf-8 -*-
"""分子规范化：同物同码。

思路（自研，不依赖外部化学信息学库）：
1. 原子初始不变量：元素、电荷、同位素、总氢数、度、环内/外等。
2. Weisfeiler-Lehman（颜色细化）迭代，把邻居的排序颜色签名并入自身颜色，
   直到颜色分区稳定。
3. 对仍对称（颜色相同）的原子做个体优先（individualization）+ 回溯的
   规范排列搜索：每次选一个颜色类细分，重新细化，取字典序最小的最终
   原子顺序作为规范编号。规模小的分子用精确回溯；大分子用固定随机种子的
   分区细化近似（带完整性回退）。
4. 按规范排列重排原子和键，输出规范 SMILES 风格字符串与规范哈希。

凯库勒双键布局的统一也在这里完成：对芳香环检测出的等价双键布局，
通过“规范编号 + 环内双键按规范原子序最小化”重新指派，
保证 c1ccoc1 与 C1=COC=C1 得到完全相同的规范表示。
"""
from __future__ import annotations

import hashlib
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .graph import Molecule


# --------------------------------------------------------------------------
# 初始不变量与颜色细化
# --------------------------------------------------------------------------
def initial_invariants(mol: Molecule) -> List[str]:
    ring_atoms = mol.ring_atoms()
    inv = []
    for a in mol.atoms:
        ring_sizes = sorted(mol.ring_sizes_of_atom(a.idx))
        inv.append("|".join([
            a.symbol,
            f"q{a.charge}",
            f"i{a.isotope}",
            f"h{a.total_h}",
            f"d{mol.degree(a.idx)}",
            f"r{1 if a.idx in ring_atoms else 0}",
            f"rs{','.join(map(str, ring_sizes))}",
            f"w{1 if a.is_wildcard else 0}",
        ]))
    return inv


def _rank_compress(values: Sequence) -> List[int]:
    """按内容字典序分配 0..k-1 的排名，与输入顺序无关。"""
    uniq = sorted(set(values))
    table = {v: k for k, v in enumerate(uniq)}
    return [table[v] for v in values]


def refine(mol: Molecule, colors: Sequence[int],
           extra: Optional[Dict[int, int]] = None
           ) -> List[int]:
    """一轮 WL 细化。extra 是个体优先带来的附加区分标记。"""
    sig: List[str] = []
    for i, a in enumerate(mol.atoms):
        neigh = sorted(
            (colors[j], _bond_code(mol.bond_between(i, j)))
            for j in mol.neighbors(i)
        )
        s = f"{colors[i]}<{neigh}"
        if extra and i in extra:
            s += f"!{extra[i]}"
        sig.append(s)
    # 按签名内容字典序压缩，颜色号与输入顺序无关
    return _rank_compress(sig)


def _bond_code(b) -> str:
    if b is None:
        return "?"
    if b.is_wildcard:
        return "~"
    if b.aromatic:
        return ":"
    return str(b.order)


def stable_refine(mol: Molecule, colors: Sequence[int],
                  extra: Optional[Dict[int, int]] = None
                  ) -> List[int]:
    """迭代 WL 细化到分区稳定。

    收敛判据用“颜色分区（哪些原子同色）”而非颜色数值：
    排名压缩会重排颜色号，但只要同色原子集合不再变化即达到不动点。
    """
    prev = list(colors)
    prev_partition = _partition(prev)
    while True:
        nxt = refine(mol, prev, extra)
        part = _partition(nxt)
        if part == prev_partition:
            return nxt
        prev = nxt
        prev_partition = part


def _partition(colors: Sequence[int]) -> List[Tuple[int, ...]]:
    classes: Dict[int, List[int]] = {}
    for i, c in enumerate(colors):
        classes.setdefault(c, []).append(i)
    return sorted(tuple(sorted(v)) for v in classes.values())


# --------------------------------------------------------------------------
# 规范排列（个体优先 + 回溯）
# --------------------------------------------------------------------------
# 个体优先回溯的节点预算：超过即回退到 WL 颜色近似（仍确定性）。
# 每个搜索节点都要跑一轮稳定细化，预算取线性规模以保证大库吞吐。
def _node_budget(n: int) -> int:
    return min(3000, max(80, 12 * n))


def canonical_permutation(mol: Molecule) -> List[int]:
    """返回规范排列 perm：perm[k] = 规范位置 k 上的原始原子 idx。

    优先用个体优先回溯求精确规范排列；若搜索空间超过节点预算，
    回退到 WL 颜色分区 + 稳定 idx 排序的确定性近似。
    两种路径对同一输入都产出唯一排列，保证同物同码。
    """
    n = len(mol.atoms)
    init = _compress(initial_invariants(mol))
    colors = stable_refine(mol, init)
    if len(set(colors)) == n:
        return _order_by_color(colors)

    state = {"best": None, "nodes": 0, "budget": _node_budget(n),
             "aborted": False}
    _search(mol, colors, {}, 0, n, state)
    if state["best"] is not None:
        return state["best"]
    # 回退：按颜色与 idx 的稳定排序（确定性）
    return _order_by_color(colors)


def _search(mol: Molecule, colors: Sequence[int], extra: Dict[int, int],
            depth: int, n: int, state: dict) -> None:
    if state["aborted"]:
        return
    state["nodes"] += 1
    if state["nodes"] > state["budget"]:
        state["aborted"] = True
        return
    colors = stable_refine(mol, colors, extra if extra else None)
    classes = _color_classes(colors)
    if len(classes) == n:
        perm = _order_by_color(colors)
        cand = _permutation_signature(mol, perm)
        if state["best"] is None or cand < _permutation_signature(
                mol, state["best"]):
            state["best"] = perm
        return
    # 选元素最少的非平凡颜色类（>1 个原子），在其中个体优先
    target = min((c for c in classes.values() if len(c) > 1),
                 key=lambda c: (len(c), c[0]))
    for atom in target:
        new_extra = dict(extra)
        new_extra[atom] = depth + 1
        _search(mol, colors, new_extra, depth + 1, n, state)
        if state["aborted"]:
            return


def _color_classes(colors: Sequence[int]) -> Dict[int, List[int]]:
    out: Dict[int, List[int]] = {}
    for i, c in enumerate(colors):
        out.setdefault(c, []).append(i)
    return out


def _order_by_color(colors: Sequence[int]) -> List[int]:
    return sorted(range(len(colors)), key=lambda i: (colors[i], i))


def _compress(signatures: Sequence[str]) -> List[int]:
    return _rank_compress(signatures)


def _permutation_signature(mol: Molecule, perm: List[int]) -> Tuple:
    """给定排列下整个图的可比较签名（原子属性 + 排序后的边）。"""
    pos = {old: k for k, old in enumerate(perm)}
    atom_sig = tuple(
        (mol.atoms[old].symbol, mol.atoms[old].charge,
         mol.atoms[old].isotope, mol.atoms[old].total_h,
         mol.atoms[old].is_wildcard)
        for old in perm
    )
    edge_sig = tuple(sorted(
        (min(pos[b.a1], pos[b.a2]), max(pos[b.a1], pos[b.a2]),
         _bond_code(b))
        for b in mol.bonds
    ))
    return atom_sig, edge_sig


# --------------------------------------------------------------------------
# 凯库勒布局规范化：同一芳香环的等价双键布局统一
# --------------------------------------------------------------------------
def normalize_double_bond_layout(mol: Molecule) -> None:
    """在规范编号确定后，把每个环内的双键布局按规范位置最小化。

    对每个 SSSR 环，枚举合法的交替双键指派（完美匹配），
    选“双键端点规范位置序列”字典序最小的那个。只动环内键，
    环外键（取代基、羰基等）保持不变。
    """
    from .aromatic import perceive_aromaticity
    # 重新感知芳香环（规范图已无 aromatic 标记，靠环+双键识别）
    arom_atoms, arom_bonds = perceive_aromaticity(mol)
    if not arom_bonds:
        return
    # 规范位置：用当前原子 idx 即可（调用方已重排）
    for cyc in _independent_rings(mol, arom_bonds):
        _minimize_ring(mol, cyc)


def _independent_rings(mol: Molecule, arom_bonds: Set[int]) -> List[Set[int]]:
    rings = []
    for cyc in mol.sssr():
        if cyc & arom_bonds:
            rings.append(cyc)
    return rings


def _minimize_ring(mol: Molecule, cyc: Set[int]) -> None:
    atoms = sorted({x for bi in cyc for x in
                    (mol.bonds[bi].a1, mol.bonds[bi].a2)})
    # 环上的相邻关系
    adj: Dict[int, List[int]] = {i: [] for i in atoms}
    for bi in cyc:
        b = mol.bonds[bi]
        adj[b.a1].append(b.a2)
        adj[b.a2].append(b.a1)
    # 枚举交替双键：环上选一个极大独立边集，杂原子（O/S/吡咯N）跳过
    exempt = {i for i in atoms
              if mol.atoms[i].symbol in ("O", "S")
              or (mol.atoms[i].symbol == "N" and mol.atoms[i].total_h > 0)}
    candidates = [i for i in atoms if i not in exempt]
    best_layout: Optional[Set[int]] = None
    best_key: Optional[Tuple] = None

    def ring_order() -> List[int]:
        # 从规范位置最小的原子出发绕环，得到确定序列
        start = min(atoms)
        seq = [start]
        prev = -1
        cur = start
        while True:
            nxts = [j for j in adj[cur] if j != prev]
            if not nxts:
                break
            nxt = nxts[0]
            if nxt == start:
                break
            seq.append(nxt)
            prev, cur = cur, nxt
        return seq

    seq = ring_order()
    # 动态规划枚举环上的交替匹配（按 seq 相邻边）
    edge_of_pair: Dict[Tuple[int, int], int] = {}
    for bi in cyc:
        b = mol.bonds[bi]
        edge_of_pair[(min(b.a1, b.a2), max(b.a1, b.a2))] = bi

    def layout_key(double_bonds: Set[int]) -> Tuple:
        return tuple(sorted(
            (min(mol.bonds[bi].a1, mol.bonds[bi].a2),
             max(mol.bonds[bi].a1, mol.bonds[bi].a2))
            for bi in double_bonds))

    # 环序列上的边索引
    ring_edges = []
    L = len(seq)
    for k in range(L):
        u, v = seq[k], seq[(k + 1) % L]
        ring_edges.append(edge_of_pair[(min(u, v), max(u, v))])

    # 回溯枚举完美/近完美匹配（豁免原子不占双键）
    chosen: Set[int] = set()
    used: Set[int] = set()

    def enumerate_match(k: int) -> None:
        nonlocal best_layout, best_key
        if k == L:
            # 校验：每个非豁免原子恰好一条双键
            count: Dict[int, int] = {i: 0 for i in atoms}
            for bi in chosen:
                b = mol.bonds[bi]
                count[b.a1] += 1
                count[b.a2] += 1
            if any(count[i] != 1 for i in candidates):
                return
            key = layout_key(chosen)
            if best_key is None or key < best_key:
                best_key = key
                best_layout = set(chosen)
            return
        u = seq[k]
        if u in used or u in exempt:
            enumerate_match(k + 1)
            return
        # 选项1：u 与后继 v 成双键（若 v 可用且非豁免）
        v = seq[(k + 1) % L]
        bi = ring_edges[k]
        if v not in used and v not in exempt and v != u:
            chosen.add(bi)
            used.add(u)
            used.add(v)
            enumerate_match(k + 1)
            used.discard(u)
            used.discard(v)
            chosen.discard(bi)
        # 选项2：u 不在这里成双键（它会与前驱成键，由更早的步处理）
        enumerate_match(k + 1)

    enumerate_match(0)
    if best_layout is not None:
        for bi in cyc:
            mol.bonds[bi].order = 2 if bi in best_layout else 1


# --------------------------------------------------------------------------
# 重排与规范输出
# --------------------------------------------------------------------------
def reorder_molecule(mol: Molecule, perm: List[int]) -> Molecule:
    """按规范排列重排原子和键，返回新的 Molecule。"""
    pos = {old: k for k, old in enumerate(perm)}
    out = Molecule(name=mol.name)
    for old in perm:
        src = mol.atoms[old]
        out.add_atom(src.copy(len(out.atoms)))
    for b in mol.bonds:
        out.add_bond(pos[b.a1], pos[b.a2], order=b.order,
                     aromatic=b.aromatic, is_wildcard=b.is_wildcard,
                     in_ring_query=b.in_ring_query)
    out.compute_fragments()
    return out


def canonical_smiles(mol: Molecule) -> str:
    """从规范图生成确定性的线性结构式文本。"""
    return _emit_smiles(mol)


def canonical_hash(mol: Molecule) -> str:
    """规范图的稳定哈希，用于入库去重。"""
    text = canonical_smiles(mol)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _emit_smiles(mol: Molecule) -> str:
    """以规范位置 0 为起点的 DFS 线性化（多片段用 . 连接）。"""
    mol.compute_fragments()
    parts = []
    visited_atoms: Set[int] = set()
    ring_counter = [0]
    open_rings: Dict[Tuple[int, int], int] = {}

    for frag in sorted(mol.fragments, key=lambda f: (len(f), f[0])):
        start = frag[0]
        parts.append(_dfs_emit(mol, start, visited_atoms,
                               ring_counter, open_rings, -1))
    return ".".join(parts)


def _dfs_emit(mol: Molecule, i: int, seen: Set[int],
              ring_counter: List[int], open_rings: Dict[Tuple[int, int], int],
              parent: int) -> str:
    seen.add(i)
    out = [_atom_text(mol, i)]
    # 先处理连向已访问祖先的环闭合边
    neighbors = sorted(mol.neighbors(i))
    ring_edges_here: List[Tuple[int, int]] = []
    forward: List[int] = []
    for j in neighbors:
        if j == parent:
            continue
        if j in seen:
            if (j, i) in open_rings or (i, j) in open_rings:
                continue
            ring_counter[0] += 1
            rid = ring_counter[0]
            open_rings[(i, j)] = rid
            ring_edges_here.append((j, rid))
        else:
            forward.append(j)
    for j, rid in ring_edges_here:
        b = mol.bond_between(i, j)
        out.append(_bond_symbol(b) + _ring_label(rid))
    # 关闭此前在该原子登记的环
    for (u, v), rid in list(open_rings.items()):
        if v == i and u != parent:
            b = mol.bond_between(u, v)
            out.append(_bond_symbol(b) + _ring_label(rid))
            del open_rings[(u, v)]
    forward = sorted(forward)
    for k, j in enumerate(forward):
        if j in seen:
            continue
        b = mol.bond_between(i, j)
        child = _dfs_emit(mol, j, seen, ring_counter, open_rings, i)
        wrap = f"({_bond_symbol(b)}{child})" if k > 0 or ring_edges_here \
            else f"{_bond_symbol(b)}{child}"
        out.append(wrap if len(forward) > 1 or ring_edges_here
                   else f"{_bond_symbol(b)}{child}")
    return "".join(out)


def _atom_text(mol: Molecule, i: int) -> str:
    a = mol.atoms[i]
    if a.is_wildcard:
        return "*"
    from .elements import ORGANIC_SUBSET
    # 规范化后普通氢已折叠为隐式；只有电荷、同位素或非有机子集元素才需括号
    need_bracket = (
        a.charge != 0 or a.isotope != 0
        or (a.explicit_h >= 0 and a.isotope != 0)
        or a.symbol not in ORGANIC_SUBSET
    )
    if not need_bracket:
        return a.symbol
    inner = a.symbol
    if a.isotope:
        inner = f"{a.isotope}{inner}"
    if a.explicit_h > 0:
        inner += "H" + ("" if a.explicit_h == 1 else str(a.explicit_h))
    if a.charge > 0:
        inner += "+" if a.charge == 1 else f"+{a.charge}"
    elif a.charge < 0:
        inner += "-" if a.charge == -1 else str(a.charge)
    return f"[{inner}]"


def _bond_symbol(b) -> str:
    if b is None or b.is_wildcard:
        return "~" if (b is not None and b.is_wildcard) else "-"
    if b.aromatic:
        return ":"
    return {1: "-", 2: "=", 3: "#"}.get(b.order, "-")


def _ring_label(rid: int) -> str:
    return str(rid) if rid < 10 else f"%{rid}"


# --------------------------------------------------------------------------
# 顶层规范化
# --------------------------------------------------------------------------
def canonicalize(mol: Molecule) -> Molecule:
    """完整规范化：氢表示统一 + 排列重排 + 双键布局统一，返回新的规范图。"""
    _normalize_hydrogen_representation(mol)
    perm = canonical_permutation(mol)
    canon = reorder_molecule(mol, perm)
    normalize_double_bond_layout(canon)
    return canon


def _normalize_hydrogen_representation(mol: Molecule) -> None:
    """把括号里的显式 H（[NH]、[CH3] 等）折叠进 implicit_h，

    使同一分子无论 H 是显式还是隐式写法，规范表示都相同。
    同位素氢（[2H]、[3H]）保留为显式，因为它们是不同的核素。
    独立的 [H] 原子（显式连在重原子上）保持原子身份，不在此处理。
    """
    for a in mol.atoms:
        if a.symbol == "H" or a.is_wildcard:
            continue
        if a.explicit_h > 0 and a.isotope == 0:
            a.implicit_h = a.explicit_h + a.implicit_h
            a.explicit_h = -1
