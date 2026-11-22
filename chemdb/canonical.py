"""规范 SMILES：Morgan 迭代精化 + 确定性 DFS 输出。

同一分子无论输入写法如何（凯库勒式/芳香式、支链顺序、环编号、原子遍历起点），
只要图同构就得到同一个规范字符串，作为入库去重的 key。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .molecule import Mol
from .aromaticity import kekulize


def _initial_invariants(mol: Mol) -> List[Tuple]:
    inv = []
    for a in mol.atoms:
        inv.append((
            a.number,
            a.charge,
            a.isotope if a.isotope is not None else -1,
            a.explicit_h,
            1 if a.aromatic else 0,
            mol.degree(a.index),
            sum(mol.bonds[bi].order for bi in mol.adj[a.index]),
            sum(1 for bi in mol.adj[a.index] if mol.bonds[bi].in_ring),
        ))
    return inv


def morgan_ranks(mol: Mol) -> List[int]:
    """迭代精化至类别不再分裂，返回每个原子的 rank（0..k-1）。"""
    invariants = _initial_invariants(mol)
    classes = _assign_classes(invariants)
    while True:
        sigs: List[Tuple] = []
        for ai, a in enumerate(mol.atoms):
            neigh = sorted(
                (classes[mol.other(bi, ai)], mol.bonds[bi].order)
                for bi in mol.adj[ai])
            sigs.append((classes[ai], tuple(neigh)))
        new_classes = _assign_classes(sigs)
        if new_classes == classes:
            break
        classes = new_classes
    return classes


def _assign_classes(values: List[Tuple]) -> List[int]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    classes = [0] * len(values)
    cls = 0
    prev = None
    for i in order:
        if prev is not None and values[i] != prev:
            cls += 1
        classes[i] = cls
        prev = values[i]
    return classes


def canonical_smiles(mol: Mol, aromatic: bool = True) -> str:
    """生成规范 SMILES。aromatic=True 输出芳香小写形式（默认规范形式）。"""
    if aromatic:
        # 输入可能是已清除芳香标记的凯库勒式（如 kekulize 的产物），
        # 重新做一次芳香归一，保证凯库勒式/芳香式收敛到同一规范形式。
        from .aromaticity import normalize_aromaticity
        from .rings import _flag_ring_bonds
        target = mol.copy()
        if not any(a.aromatic for a in target.atoms):
            _flag_ring_bonds(target)
            normalize_aromaticity(target)
    else:
        target = kekulize(mol)
    ranks = morgan_ranks(target)
    # 多片段：按每片段最小 rank 排序依次输出
    comps = target.components()
    comps.sort(key=lambda c: min(ranks[i] for i in c))
    parts = [_canonical_component(target, c, ranks) for c in comps]
    return ".".join(parts)


def _canonical_component(mol: Mol, comp: List[int],
                         ranks: List[int]) -> str:
    start = min(comp, key=lambda i: (ranks[i], i))
    return _dfs_recursive(mol, start, ranks)


def _ordered_neighbors(mol: Mol, ai: int, ranks: List[int]) -> List[int]:
    return sorted(mol.neighbors(ai),
                  key=lambda v: (ranks[v], v))


def _dfs_recursive(mol: Mol, start: int, ranks: List[int]) -> str:
    """确定性 DFS 生成 SMILES。

    环编号规则（与 Daylight 一致）：
      每条非树边（回边）分配一个环号；
      当 DFS 沿树边进入子节点 v 时，若 v 已带一条通向“更早访问节点”的
      回边，环号作为开环数字跟在父原子该键 token 后；
      当遍历到回边的另一端（较深节点）时，环号作为闭合数字跟在该节点
      全部子树输出之后。
    """
    parent: Dict[int, Optional[int]] = {start: None}
    children: Dict[int, List[int]] = {}
    back: Dict[int, List[int]] = {}
    ring_no: Dict[Tuple[int, int], int] = {}
    counter = [0]
    # 三色 DFS：0=白(未发现) 1=灰(在递归栈) 2=黑(已完成)
    color: Dict[int, int] = {start: 1}

    def rkey(u: int, v: int) -> Tuple[int, int]:
        return (u, v) if u < v else (v, u)

    # ---- 第一遍：无向图三色 DFS 建树 ----
    # 无向图只有树边与回边：非父邻居为灰=回边，为白=树边，
    # 为黑说明该边已从对端登记过，跳过。
    def build(ai: int) -> None:
        nbrs = _ordered_neighbors(mol, ai, ranks)
        kids: List[int] = []
        backs: List[int] = []
        for v in nbrs:
            if v == parent[ai]:
                continue
            cv = color.get(v, 0)
            if cv == 0:  # 白：树边
                parent[v] = ai
                color[v] = 1
                kids.append(v)
                build(v)
            elif cv == 1:  # 灰：回边（连向栈上祖先）
                backs.append(v)
                k = rkey(ai, v)
                if k not in ring_no:
                    counter[0] += 1
                    ring_no[k] = counter[0]
            # 黑色：对端已处理同一条边，忽略
        children[ai] = kids
        back[ai] = backs
        color[ai] = 2

    build(start)

    # ---- 第二遍：迭代 DFS 输出，保证环号位置严格符合 SMILES ----
    # open_at[浅端]：跟在浅端原子 token 后的 (键符号, 环号数字)
    # close_at[深端]：深端子树输出完毕后追加的环号数字
    open_at: Dict[int, List[Tuple[str, str]]] = {}
    close_at: Dict[int, List[str]] = {}
    for (u, v), no in ring_no.items():
        if _depth(parent, u) < _depth(parent, v):
            shallow, deep = u, v
        else:
            shallow, deep = v, u
        b = mol.bonds[mol.bond_index_between(u, v)]
        bond_tok = _bond_text(mol, u, v, b.order)
        digit = _ring_digit(no)
        open_at.setdefault(shallow, []).append((bond_tok, digit))
        close_at.setdefault(deep, []).append(digit)

    out: List[str] = []

    def emit_node_header(ai: int) -> None:
        """原子 token + 该节点持有的开环号与闭合号（均在子树之前）。"""
        out.append(_atom_text(mol, ai))
        for bond_tok, digit in open_at.get(ai, []):
            out.append(bond_tok + digit)
        for digit in close_at.get(ai, []):
            out.append(digit)

    # 栈帧 (ai, 待处理子节点列表, 该子树是否以括号包裹)
    stack: List[Tuple[int, List[int], bool]] = [
        (start, list(children[start]), False)]
    emit_node_header(start)

    while stack:
        ai, pending, bracketed = stack[-1]
        if not pending:
            stack.pop()
            if bracketed:
                out.append(")")
            continue
        v = pending.pop(0)
        b = mol.bonds[mol.bond_index_between(ai, v)]
        bond = _bond_text(mol, ai, v, b.order)
        # 多分支：除最后一个子树外，其余子树用括号包裹
        is_branch = bool(pending)
        if is_branch:
            out.append("(")
        out.append(bond)
        emit_node_header(v)
        stack.append((v, list(children[v]), is_branch))

    return "".join(out)


def _depth(parent: Dict[int, Optional[int]], x: int) -> int:
    d = 0
    while parent[x] is not None:
        x = parent[x]
        d += 1
    return d


def _ring_digit(rid: int) -> str:
    return "%%%d" % rid if rid >= 10 else str(rid)


def _bond_text(mol: Mol, u: int, v: int, order: int) -> str:
    a1, a2 = mol.atoms[u], mol.atoms[v]
    if order == 4:
        return ""  # 芳香-芳香键省略
    if order == 1:
        # 芳香与脂肪之间的单键必须显式写出
        if a1.aromatic != a2.aromatic:
            return "-"
        return "-" if (a1.bracket or a2.bracket) else ""
    if order == 2:
        return "="
    if order == 3:
        return "#"
    return "-"


def _atom_text(mol: Mol, ai: int) -> str:
    a = mol.atoms[ai]
    if a.wildcard:
        return "*"
    if a.aromatic_query:
        return "a"
    if a.aliphatic_query:
        return "A"
    if a.aromatic:
        # 芳香小写：[nH] 需要括号，其余裸写
        if a.element == "N" and (a.explicit_h or a.charge):
            return _bracket_text(a, lower=True)
        if a.element in ("c", "C", "n", "N", "o", "O", "s", "S",
                         "p", "P", "b", "B", "se", "Se"):
            if not a.bracket and a.charge == 0 and a.isotope is None \
                    and a.explicit_h == 0:
                return a.symbol
        return _bracket_text(a, lower=True)
    # 脂肪原子
    from .elements import ORGANIC_SUBSET
    if (not a.bracket and a.isotope is None and a.charge == 0
            and a.explicit_h == 0 and a.symbol in ORGANIC_SUBSET):
        return a.symbol
    return _bracket_text(a, lower=False)


def _bracket_text(a, lower: bool) -> str:
    sym = a.symbol if lower else a.element
    s = "["
    if a.isotope is not None:
        s += str(a.isotope)
    s += sym
    if a.explicit_h:
        s += "H" + (str(a.explicit_h) if a.explicit_h > 1 else "")
    if a.charge:
        if abs(a.charge) == 1:
            s += "+" if a.charge > 0 else "-"
        else:
            s += ("+" if a.charge > 0 else "-") + str(abs(a.charge))
    return s + "]"
