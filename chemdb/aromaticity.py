"""芳香性感知、芳香归一与凯库勒化。

两条入口路径：
  1. 输入本身是小写芳香写法：校验芳香原子必须落在满足 Hückel 规则的环内；
  2. 输入是凯库勒式（全大写 + 交替单双键）：收集 5~7 元、全可芳香化原子、
     仅含单双键的候选环，按共享原子并成芳香体系，再用“环内双键的完美匹配
     是否可行”整体判定（与 RDKit 等工具的凯库勒化判据同源），能正确处理
     萘等稠环——它们拆成最小环基后单个环并不独立满足 4n+2。

这样 "C1=CC=CC=C1" 与 "c1ccccc1"、苯并咪唑的凯库勒式与芳香式，
归一后都得到同一张芳香分子图，规范化阶段即产生同一个规范 SMILES。
"""
from __future__ import annotations

from typing import Dict, List, Set, Tuple

from .elements import AROMATIZABLE
from .molecule import Mol
from .rings import minimum_cycle_basis, ring_atom_set


class AromaticityError(Exception):
    pass


def _cycle_edges(mol: Mol, cyc: List[int]) -> Set[int]:
    edges: Set[int] = set()
    for k in range(len(cyc)):
        bi = mol.bond_index_between(cyc[k], cyc[(k + 1) % len(cyc)])
        if bi is None:
            raise AromaticityError("环数据不完整")
        edges.add(bi)
    return edges


def _is_huckel(e: int) -> bool:
    return e >= 2 and (e - 2) % 4 == 0


def _aromatic_pi_electrons(mol: Mol, cyc: List[int],
                           cyc_edges: Set[int]) -> int:
    """统计已用小写芳香写法的环的 pi 电子数。

    芳香碳/硼贡献 1；[nH]、芳香 O/S/Se 及中性三配位以下 N 按孤对贡献 2；
    带电荷原子按电荷修正。
    """
    total = 0
    for ai in cyc:
        a = mol.atoms[ai]
        el = a.element
        if el == "C":
            total += 1
        elif el == "B":
            total += 0 if a.charge == 0 else 1
        elif el in ("O", "S", "Se"):
            total += 2 if a.charge == 0 else 1
        elif el in ("N", "P", "As"):
            if a.explicit_h > 0 or a.charge < 0:
                total += 2  # 吡咯型 [nH] / [n-]
            elif a.charge > 0:
                total += 1
            else:
                # 环内度为 2 的中性 n 为吡啶型（贡献 1），否则吡咯型（2）
                in_ring_deg = sum(1 for bi in mol.adj[ai]
                                  if bi in cyc_edges)
                total += 1 if in_ring_deg == 2 else 2
        else:
            total += 1
    return total


def normalize_aromaticity(mol: Mol, allow_loose: bool = False) -> None:
    # ---- 路径 1：输入自带芳香标记，校验其合法 ----
    aromatic_atoms = {i for i, a in enumerate(mol.atoms) if a.aromatic}
    if aromatic_atoms:
        # 先用线性时间的环标记，避免为常见芳香输入计算完整最小环基
        ring_atoms = ring_atom_set(mol)
        for ai in aromatic_atoms:
            if ai not in ring_atoms and not allow_loose:
                raise AromaticityError(
                    "芳香原子 %d 不在任何环内，芳香写法非法" % ai)
        if not allow_loose:
            # 仅在需要逐环 Hückel 校验时才计算最小环基
            cycles = minimum_cycle_basis(mol)
            for cyc in cycles:
                if not any(mol.atoms[ai].aromatic for ai in cyc):
                    continue
                edges = _cycle_edges(mol, cyc)
                electrons = _aromatic_pi_electrons(mol, cyc, edges)
                if not _is_huckel(electrons):
                    raise AromaticityError(
                        "含 %d 个原子的芳香环 pi 电子数 %d 不满足 4n+2 规则"
                        % (len(cyc), electrons))
        # 芳香原子之间的默认键统一为芳香键
        for b in mol.bonds:
            if (mol.atoms[b.a1].aromatic and mol.atoms[b.a2].aromatic
                    and b.order in (1, 4)):
                b.order = 4
        return

    cycles = minimum_cycle_basis(mol)

    # ---- 路径 2：凯库勒式 -> 芳香式 ----
    # 稠环（如萘）拆成最小环基后单个环独立算 pi 电子往往不满足 4n+2，
    # 不能逐环判定。改为：收集结构上可能芳香的环 -> 按共享原子并成候选
    # 体系 -> 检验体系能否整体凯库勒化（环内单双键完美匹配自洽）。
    original_orders = [b.order for b in mol.bonds]
    candidate_cycles: List[Tuple[List[int], Set[int]]] = []
    for cyc in cycles:
        edges = _cycle_edges(mol, cyc)
        if _structurally_aromatizable(mol, cyc, edges, original_orders):
            candidate_cycles.append((cyc, edges))

    if not candidate_cycles:
        return

    systems = _group_cycles(candidate_cycles)
    for sys_atoms, sys_edges in systems:
        if _system_kekule_feasible(mol, sys_atoms, sys_edges,
                                   original_orders):
            for ai in sys_atoms:
                a = mol.atoms[ai]
                if a.element not in AROMATIZABLE:
                    continue
                # 吡咯型 N：环内全单键、中性且无显式氢，
                # 凯库勒式里它靠一个隐式 H 满价，芳香化时转成 [nH]。
                if (a.element == "N" and a.charge == 0
                        and a.explicit_h == 0
                        and all(original_orders[bi] == 1
                                for bi in mol.adj[ai] if bi in sys_edges)):
                    a.explicit_h = 1
                    a.bracket = True
                a.aromatic = True
                a.symbol = a.symbol.lower() if a.element != "Se" else "se"
            for bi in sys_edges:
                b = mol.bonds[bi]
                if mol.atoms[b.a1].aromatic and mol.atoms[b.a2].aromatic:
                    b.order = 4


def _structurally_aromatizable(mol: Mol, cyc: List[int],
                               cyc_edges: Set[int],
                               order_snapshot: List[int]) -> bool:
    if not (5 <= len(cyc) <= 7):
        return False
    for ai in cyc:
        a = mol.atoms[ai]
        if a.element not in AROMATIZABLE or a.aromatic:
            return False
    return all(order_snapshot[bi] in (1, 2) for bi in cyc_edges)


def _needs_double_bond(mol: Mol, ai: int,
                       edges: Set[int],
                       order_snapshot: List[int]) -> bool:
    """该芳香候选原子是否需要一条环内双键来满足价态。

    碳/硼通常需要；O/S/Se 以孤对贡献 2 电子，不需要；
    吡咯型 N（带显式 H 或负电，或环内全单键）不需要，吡啶型 N 需要。
    """
    a = mol.atoms[ai]
    el = a.element
    if el in ("O", "S", "Se"):
        return False
    if el in ("N", "P", "As"):
        if a.explicit_h > 0 or a.charge < 0:
            return False  # 吡咯型 [nH] / [n-]
        # 凯库勒式里裸 N 的氢是隐式的，改由键结构判断：
        # 环内若已有一条双键，是吡啶型（需要双键）；
        # 环内全是单键，则是吡咯型（以孤对贡献，不需要双键）。
        ring_double = any(
            order_snapshot[bi] == 2 for bi in mol.adj[ai]
            if bi in edges)
        return ring_double
    return True  # C、B 等默认需要


def _system_kekule_feasible(mol: Mol, atoms: Set[int], edges: Set[int],
                            order_snapshot: List[int]) -> bool:
    """候选体系环内单/双键能否构成自洽凯库勒结构（完美匹配）。"""
    # 饱和体系（无任何环内双键且无孤对贡献者）不含 pi 电子，直接排除，
    # 否则环己烷这类全单键环会被误判为芳香。
    has_double = any(order_snapshot[bi] == 2 for bi in edges)
    has_lone_pair = any(
        not _needs_double_bond(mol, ai, edges, order_snapshot)
        and mol.atoms[ai].element in ("N", "P", "As", "O", "S", "Se")
        for ai in atoms)
    if not has_double and not has_lone_pair:
        return False

    double_count: Dict[int, int] = {ai: 0 for ai in atoms}
    for bi in edges:
        if order_snapshot[bi] == 2:
            b = mol.bonds[bi]
            double_count[b.a1] += 1
            double_count[b.a2] += 1
    if any(v > 1 for v in double_count.values()):
        return False

    single_adj: Dict[int, List[int]] = {ai: [] for ai in atoms}
    for bi in edges:
        if order_snapshot[bi] != 1:
            continue
        b = mol.bonds[bi]
        if b.a1 in atoms and b.a2 in atoms:
            single_adj[b.a1].append(b.a2)
            single_adj[b.a2].append(b.a1)

    # 只有“需要靠双键满足价态”的原子才参与额外双键匹配；
    # O/S/Se 及吡咯型 N 这类以孤对贡献 2 电子的原子无需环内双键。
    need = [ai for ai in atoms
            if double_count[ai] == 0
            and _needs_double_bond(mol, ai, edges, order_snapshot)]
    used: Set[int] = set()

    def match(k: int) -> bool:
        if k == len(need):
            return True
        ai = need[k]
        if ai in used:
            return match(k + 1)
        for nbr in single_adj[ai]:
            if (nbr in used or double_count.get(nbr, 0) > 0
                    or nbr not in atoms
                    or not _needs_double_bond(mol, nbr, edges,
                                              order_snapshot)):
                continue
            used.add(ai)
            used.add(nbr)
            if match(k + 1):
                return True
            used.discard(ai)
            used.discard(nbr)
        return False

    return match(0)


def _group_cycles(aromatic_cycles):
    n = len(aromatic_cycles)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        parent[find(x)] = find(y)

    atom_owner: Dict[int, int] = {}
    for i, (cyc, edges) in enumerate(aromatic_cycles):
        for ai in cyc:
            if ai in atom_owner:
                union(i, atom_owner[ai])
            atom_owner[ai] = i
    groups: Dict[int, Tuple[Set[int], Set[int]]] = {}
    for i, (cyc, edges) in enumerate(aromatic_cycles):
        r = find(i)
        atoms, es = groups.setdefault(r, (set(), set()))
        atoms.update(cyc)
        es.update(edges)
    return list(groups.values())


# ------------------------------------------------------------------
# 凯库勒化：把芳香键（order=4）重新分配为单/双键，回溯求解
# ------------------------------------------------------------------
def kekulize(mol: Mol) -> Mol:
    """返回一份凯库勒式分子（芳香键 -> 单/双键），失败抛异常。"""
    out = mol.copy()
    aromatic_bonds = [i for i, b in enumerate(out.bonds) if b.order == 4]
    if not aromatic_bonds:
        return out
    # 每个芳香原子需要的“环内芳香双键”数量。
    # 凯库勒化把每条芳香键定成单键或双键，每个原子的双键需求由其
    # 化学角色决定（环外键与显式氢已经占用了相应价态）：
    #   芳香碳/硼/吡啶型 N：需要 1 条环内双键；
    #   O/S/Se 与吡咯型 [nH]/[n-]：以孤对贡献 2 电子，需要 0 条。
    demand: Dict[int, int] = {}
    for a in out.atoms:
        if not a.aromatic:
            continue
        el = a.element
        if el in ("O", "S", "Se"):
            need = 0 if a.charge == 0 else 1
        elif el in ("N", "P", "As"):
            if a.explicit_h > 0 or a.charge < 0:
                need = 0  # 吡咯型
            else:
                need = 1  # 吡啶型
        else:  # C、B 等
            need = 1
        demand[a.index] = need

    remaining = dict(demand)
    # 按约束最多的原子排序，回溯分配
    order = sorted(aromatic_bonds,
                   key=lambda bi: min(
                       remaining.get(out.bonds[bi].a1, 0),
                       remaining.get(out.bonds[bi].a2, 0)),
                   reverse=True)

    def assign(k: int) -> bool:
        if k == len(order):
            return all(v == 0 for v in remaining.values())
        bi = order[k]
        b = out.bonds[bi]
        u, v = b.a1, b.a2
        # 尝试双键
        if remaining.get(u, 0) > 0 and remaining.get(v, 0) > 0:
            remaining[u] -= 1
            remaining[v] -= 1
            b.order = 2
            if assign(k + 1):
                return True
            remaining[u] += 1
            remaining[v] += 1
        # 单键
        b.order = 1
        if assign(k + 1):
            return True
        b.order = 4
        return False

    if not assign(0):
        raise AromaticityError("芳香体系无法凯库勒化")
    for a in out.atoms:
        if a.aromatic:
            a.aromatic = False
            from .elements import AROMATIC_TO_ALIPHATIC
            a.symbol = AROMATIC_TO_ALIPHATIC.get(a.symbol, a.symbol)
    return out
