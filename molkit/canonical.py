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
        # 只用稳定不变量。注意：SSSR 环尺寸集合依赖生成树选择，等价凯库勒
        # 布局下可能不同，因此不放入；环成员关系（是否在环内）是稳定的。
        inv.append("|".join([
            a.symbol,
            f"q{a.charge}",
            f"i{a.isotope}",
            f"h{a.total_h}",
            f"d{mol.degree(a.idx)}",
            f"r{1 if a.idx in ring_atoms else 0}",
            f"w{1 if a.is_wildcard else 0}",
        ]))
    return inv


def _rank_compress(values: Sequence) -> List[int]:
    """按内容字典序分配 0..k-1 的排名，与输入顺序无关。"""
    uniq = sorted(set(values))
    table = {v: k for k, v in enumerate(uniq)}
    return [table[v] for v in values]


def refine(mol: Molecule, colors: Sequence[int],
           extra: Optional[Dict[int, int]] = None,
           ignore_bond_order: bool = False,
           ) -> List[int]:
    """一轮 WL 细化。extra 是个体优先带来的附加区分标记。

    ignore_bond_order=True 时邻居键只区分 芳香/通配/普通，不区分单双三键——
    规范排列必须对等价的凯库勒双键布局保持不变，键级在布局规范化阶段再定。
    """
    sig: List[str] = []
    for i, a in enumerate(mol.atoms):
        neigh = sorted(
            (colors[j],
             _bond_code(mol.bond_between(i, j), ignore_bond_order))
            for j in mol.neighbors(i)
        )
        s = f"{colors[i]}<{neigh}"
        if extra and i in extra:
            s += f"!{extra[i]}"
        sig.append(s)
    # 按签名内容字典序压缩，颜色号与输入顺序无关
    return _rank_compress(sig)


def _bond_code(b, ignore_order: bool = False) -> str:
    if b is None:
        return "?"
    if b.is_wildcard:
        return "~"
    if b.aromatic:
        return ":"
    if ignore_order:
        return "b"   # 所有确定的整数键级（单/双/三）在排列阶段等价
    return str(b.order)


def stable_refine(mol: Molecule, colors: Sequence[int],
                  extra: Optional[Dict[int, int]] = None,
                  ignore_bond_order: bool = True
                  ) -> List[int]:
    """迭代 WL 细化到分区稳定。

    默认忽略键级：规范排列只依赖元素与连接拓扑，从而对等价的凯库勒
    双键布局不变。收敛判据用颜色分区（哪些原子同色）而非颜色数值。
    """
    prev = list(colors)
    prev_partition = _partition(prev)
    while True:
        nxt = refine(mol, prev, extra,
                     ignore_bond_order=ignore_bond_order)
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


def stable_refine_sigs(mol: Molecule, init_sigs: Sequence[str]
                       ) -> List[str]:
    """WL 细化，返回每个原子最终的完整签名字符串（内容稳定，不依赖
    原子输入顺序：相同结构语境必产生相同字符串）。"""
    current = list(init_sigs)
    prev_partition = None
    while True:
        nxt = []
        for i in range(len(mol.atoms)):
            neigh = sorted(
                (current[j], _bond_code(mol.bond_between(i, j),
                                         ignore_order=True))
                for j in mol.neighbors(i)
            )
            nxt.append(f"{current[i]}<{neigh}")
        part = _partition(_rank_compress(nxt))
        if part == prev_partition:
            return current
        prev_partition = part
        current = nxt


def canonical_permutation(mol: Molecule) -> List[int]:
    """返回规范排列 perm：perm[k] = 规范位置 k 上的原始原子 idx。

    以 WL 最终内容签名的字典序排名作为规范序：签名只由元素、连接拓扑与
    稳定属性决定，与原子输入顺序及等价凯库勒布局无关，因此同构分子
    （含不同写法）得到相同的规范排列。仍有同签名原子时再个体优先消歧。
    """
    n = len(mol.atoms)
    final_sigs = stable_refine_sigs(mol, initial_invariants(mol))
    colors = _rank_compress(final_sigs)
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
         _bond_code(b, ignore_order=True))
        for b in mol.bonds
    ))
    return atom_sig, edge_sig


# --------------------------------------------------------------------------
# 凯库勒布局规范化：同一芳香环的等价双键布局统一
# --------------------------------------------------------------------------
def normalize_double_bond_layout(mol: Molecule) -> None:
    """在规范编号确定后，对每个芳香连通体系做全局确定性双键指派。

    关键：稠环（如萘）的多个环共享键，逐环独立最小化会互相冲突、
    产生不稳定布局。这里把整个芳香子图作为一个体系，求一次完美匹配，
    并在所有合法匹配中选“双键边按规范端点排序后字典序最小”的那个，
    从而与输入写法无关、结果唯一。
    """
    from .aromatic import perceive_aromaticity
    arom_atoms, arom_bonds = perceive_aromaticity(mol)
    if not arom_bonds:
        return
    # 计算 DFS 环闭合边（回边）：这些边在输出时只能是裸数字单键，
    # 因此双键布局要尽量避开它们，把双键安排在树边上。
    closure_bond_ids: Set[int] = set()
    for frag in mol.compute_fragments():
        ring_no, _labels, _parent = _assign_ring_numbers(mol, frag[0])
        for (u, v) in ring_no:
            bi = _bond_index(mol, u, v)
            if bi is not None:
                closure_bond_ids.add(bi)
    # 按芳香键的连通性划分芳香体系
    components = _aromatic_components(mol, arom_atoms, arom_bonds)
    for comp_atoms in components:
        comp_bonds = {bi for bi in arom_bonds
                      if mol.bonds[bi].a1 in comp_atoms
                      and mol.bonds[bi].a2 in comp_atoms}
        _minimize_component(mol, comp_atoms, comp_bonds, closure_bond_ids)


def _aromatic_components(mol: Molecule, arom_atoms: Set[int],
                         arom_bonds: Set[int]) -> List[Set[int]]:
    adj: Dict[int, List[int]] = {i: [] for i in arom_atoms}
    for bi in arom_bonds:
        b = mol.bonds[bi]
        adj[b.a1].append(b.a2)
        adj[b.a2].append(b.a1)
    seen: Set[int] = set()
    out: List[Set[int]] = []
    for start in arom_atoms:
        if start in seen:
            continue
        stack = [start]
        comp: Set[int] = set()
        seen.add(start)
        while stack:
            x = stack.pop()
            comp.add(x)
            for y in adj[x]:
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
        out.append(comp)
    return out


def _bond_index(mol: Molecule, a1: int, a2: int) -> Optional[int]:
    for bi in mol.adj[a1]:
        if mol.bonds[bi].other(a1) == a2:
            return bi
    return None


def _minimize_component(mol: Molecule, comp_atoms: Set[int],
                        comp_bonds: Set[int],
                        closure_bond_ids: Optional[Set[int]] = None) -> None:
    """枚举芳香体系上所有合法双键匹配，取全局最优者。

    优化目标（按优先级）：
    1. 环闭合边（回边）尽量不承载双键——它们在规范 SMILES 里只能写成
       裸数字单键，双键放树边上才能被解析器无损读回；
    2. 在满足 1 的布局中，取双键边按规范端点排序后字典序最小者。
    纯由规范原子编号决定、与输入初始键级无关，因此幂等。
    """
    closure_bond_ids = closure_bond_ids or set()
    sub_adj: Dict[int, List[int]] = {i: [] for i in comp_atoms}
    for bi in comp_bonds:
        b = mol.bonds[bi]
        sub_adj[b.a1].append(b.a2)
        sub_adj[b.a2].append(b.a1)

    # 孤对电子豁免原子（呋喃 O、噻吩 S、吡咯 N 等）
    exempt = set()
    for i in comp_atoms:
        a = mol.atoms[i]
        if a.symbol in ("O", "S"):
            exempt.add(i)
        elif a.symbol == "N" and a.total_h > 0:
            exempt.add(i)
    for i in exempt:
        sub_adj.pop(i, None)
    for i in list(sub_adj):
        sub_adj[i] = [j for j in sub_adj[i] if j not in exempt]

    active = sorted(i for i in comp_atoms if i not in exempt)
    adj_set: Dict[int, Set[int]] = {i: set(sub_adj.get(i, []))
                                     for i in active}
    chosen: Set[int] = set()
    used: Set[int] = set()
    best_key: Optional[Tuple] = None
    best: Optional[Set[int]] = None

    def layout_key() -> Tuple:
        # 第一关键字：落在环闭合边上的双键数（越少越好，理想为 0）；
        # 第二关键字：双键边端点的规范字典序。
        closure_doubles = sum(1 for bi in chosen if bi in closure_bond_ids)
        edges = tuple(sorted(
            (min(mol.bonds[bi].a1, mol.bonds[bi].a2),
             max(mol.bonds[bi].a1, mol.bonds[bi].a2))
            for bi in chosen))
        return (closure_doubles, edges)

    def backtrack(k: int) -> None:
        nonlocal best_key, best
        if k == len(active):
            key = layout_key()
            if best_key is None or key < best_key:
                best_key = key
                best = set(chosen)
            return
        x = active[k]
        if x in used:
            backtrack(k + 1)
            return
        for y in sorted(adj_set.get(x, set())):
            if y in used:
                continue
            bi = _bond_index(mol, x, y)
            if bi is None or bi not in comp_bonds:
                continue
            chosen.add(bi)
            used.add(x)
            used.add(y)
            backtrack(k + 1)
            used.discard(x)
            used.discard(y)
            chosen.discard(bi)

    backtrack(0)
    if best is None:
        return  # 保留原布局，避免破坏
    for bi in comp_bonds:
        mol.bonds[bi].order = 2 if bi in best else 1


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
    """以规范位置 0 为起点的 DFS 线性化（多片段用 . 连接）。

    两遍：先按固定 DFS 顺序找出所有回边并分配环号，记录每个原子要输出的
    环标签；再发射。环标签紧贴原子（如 C1...C1），不进括号、不加额外连字符。
    """
    mol.compute_fragments()
    parts = []
    for frag in sorted(mol.fragments, key=lambda f: (len(f), f[0])):
        ring_no, labels_at, parent = _assign_ring_numbers(mol, frag[0])
        # 每个原子子树（沿 parent 向下）中出现的环号集合
        subtree_rings = _subtree_ring_numbers(mol, frag[0], parent,
                                              labels_at)
        seen: Set[int] = set()
        parts.append(_dfs_emit(mol, frag[0], seen, parent, labels_at,
                               subtree_rings))
    return ".".join(parts)


def _subtree_ring_numbers(mol: Molecule, root: int, parent: dict,
                          labels_at: Dict[int, List[Tuple[int, int, bool]]]
                          ) -> Dict[int, Set[int]]:
    """自底向上汇总每个原子子树内出现的所有环闭合标签号。"""
    result: Dict[int, Set[int]] = {i: set() for i in range(len(mol.atoms))}

    def walk(i: int) -> Set[int]:
        rings = {rid for _b, rid, _c in labels_at.get(i, [])}
        for j in mol.neighbors(i):
            if parent.get(j) == i:
                rings |= walk(j)
        result[i] = rings
        return rings

    walk(root)
    return result


def _assign_ring_numbers(mol: Molecule, start: int):
    """DFS 预先分配环号。

    返回 (回边集合, 原子->[(bond, 环号, 是否闭合端)], DFS 父节点表)。
    回边在 DFS 中后遇到的端点为“闭合端”：开口端写裸数字，
    闭合端写 键符号+数字（如 C1...C=1），既合法又保留键级。
    """
    ring_no: Dict[Tuple[int, int], int] = {}
    # labels_at: atom -> list of (bond, rid, is_closing)
    labels_at: Dict[int, List[Tuple[int, int, bool]]] = {}
    seen: Set[int] = {start}
    parent = {start: -1}
    counter = [0]
    stack = [(start, iter(sorted(mol.neighbors(start))))]
    while stack:
        node, it = stack[-1]
        advanced = False
        for j in it:
            edge = (min(node, j), max(node, j))
            if j == parent[node]:
                continue
            if j in seen:
                if edge in ring_no:
                    continue
                counter[0] += 1
                rid = counter[0]
                ring_no[edge] = rid
                b = mol.bond_between(node, j)
                # node 是当前 DFS 后遇到的闭合端；j 是更早的开口端。
                # 开口端裸数字，闭合端数字前写键符号（单键省略）。
                labels_at.setdefault(j, []).append((b, rid, False))
                labels_at.setdefault(node, []).append((b, rid, True))
            else:
                seen.add(j)
                parent[j] = node
                stack.append((j, iter(sorted(mol.neighbors(j)))))
                advanced = True
                break
        if not advanced:
            stack.pop()
    for i in labels_at:
        labels_at[i].sort(key=lambda x: x[1])
    return ring_no, labels_at, parent


def _dfs_emit(mol: Molecule, i: int, seen: Set[int], parent: dict,
              labels_at: Dict[int, List[Tuple[int, int, bool]]],
              subtree_rings: Optional[Dict[int, Set[int]]] = None) -> str:
    seen.add(i)
    out = [_atom_text(mol, i)]
    # 开口端裸数字；闭合端在数字前写键符号（单键省略），保留环内键级。
    for b, rid, is_closing in labels_at.get(i, []):
        out.append((_ring_bond_symbol(b) if is_closing else "")
                   + _ring_label(rid))
    # 只沿 DFS 树边向下；回边已由环标签表达，不再作为支链输出。
    children = [j for j in sorted(mol.neighbors(i))
                if parent.get(j) == i]
    if subtree_rings is not None and children:
        # 环骨架延续（子树含环号）排前面优先走主链，取代基（无环号）排后。
        children.sort(key=lambda j: (0 if subtree_rings.get(j) else 1, j))

    for k, j in enumerate(children):
        b = mol.bond_between(i, j)
        child_text = _dfs_emit(mol, j, seen, parent, labels_at,
                               subtree_rings)
        piece = f"{_bond_symbol(b)}{child_text}"
        # 标准 DFS-SMILES：第一个孩子直接延伸，其余孩子一律括号支链。
        out.append(piece if k == 0 else f"({piece})")
    return "".join(out)


def _ring_bond_symbol(b) -> str:
    """闭合端环键符号：单键省略，双/三/芳香显式。"""
    if b.is_wildcard:
        return "~"
    if b.aromatic:
        return ":"
    return {1: "", 2: "=", 3: "#"}.get(b.order, "")


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
