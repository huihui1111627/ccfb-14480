# -*- coding: utf-8 -*-
"""自研线性结构式（SMILES 方言）解析器。

支持：
  - 裸原子有机子集 B C N O P S F Cl Br I 与芳香小写 b c n o p s
  - 括号原子 [同位素 元素 H数 电荷 类号]，如 [13CH3-]、[NH4+]、[nH]、[OH-]
  - 键级 - = # : ，立体键 / \\（按单键接受），查询键 ~（任意） @（环内） @!（环外）
  - 支链 ( )、点断片段 .、环闭合 1..9 与 %10..%99（环闭合可带键级，如 C=1CCCCC1）
  - 任意原子 * 、括号内原子序数 [#6]、查询标记 R / !R / r<n> / !r<n>
  - 电荷 + - ++ -- +2 -2、同位素前缀、显式氢 [CH2] 与独立 [H]
不支持并明确报错：立体化学 @@（会被吃掉但不参与图结构，仅保留兼容）。

所有语法错误抛 ParseError，带 0 基位置 pos 与 1 基位置 column。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .elements import (AROMATIC_ORGANIC, ORGANIC_SUBSET, SYMBOL_TO_Z)
from .graph import Atom, Bond, Molecule


class ParseError(Exception):
    def __init__(self, message: str, pos: int):
        self.pos = pos          # 0 基字符下标
        self.column = pos + 1   # 1 基，人类阅读
        super().__init__(f"结构式不合法（第 {self.column} 个字符）：{message}")


# 两字符元素优先匹配
_TWO_CHAR = {
    "Li", "Be", "Ne", "Na", "Mg", "Al", "Si", "Cl", "Ar", "Ca", "Sc", "Ti",
    "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se",
    "Br", "Kr", "Rb", "Sr", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag",
    "Cd", "In", "Sn", "Sb", "Te", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf",
    "Ta", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At",
    "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "Np", "Pu",
}
_ONE_CHAR = set("BCNOPSFIHKWUVY")


class Parser:
    def __init__(self, text: str):
        self.s = text
        self.n = len(text)
        self.i = 0
        self.mol = Molecule()
        # 环号 -> (原子 idx, 键描述 or None, 环号字符位置)
        self.rings: Dict[int, Tuple[int, Optional[dict], int]] = {}

    # ---------- 基础游标 ----------
    def _peek(self, off: int = 0) -> str:
        j = self.i + off
        return self.s[j] if j < self.n else ""

    def _err(self, msg: str, pos: Optional[int] = None) -> ParseError:
        return ParseError(msg, self.i if pos is None else pos)

    def parse(self) -> Molecule:
        if self.n == 0:
            raise self._err("空字符串")
        self._parse_sequence(allow_dot=True)
        if self.i != self.n:
            raise self._err(f"多余的字符 {self.s[self.i:self.i + 1]!r}")
        if not self.mol.atoms:
            raise self._err("没有解析到任何原子")
        if self.rings:
            nums = sorted(self.rings)
            ring_pos = self.rings[nums[0]][2]
            raise ParseError(
                f"环闭合编号 {nums[0]} 没有配对的闭合标记", ring_pos)
        self.mol.compute_fragments()
        return self.mol

    # ---------- 序列/支链 ----------
    def _parse_sequence(self, allow_dot: bool,
                        stop_at: Optional[str] = None) -> None:
        prev: Optional[int] = None
        pending_bond: Optional[dict] = None
        while self.i < self.n:
            ch = self.s[self.i]
            if stop_at is not None and ch == stop_at:
                return
            if ch == "(":
                if prev is None:
                    raise self._err("支链不能作为序列开头")
                self.i += 1
                self._parse_branch(prev)
                continue
            if ch == ")":
                if stop_at is None:
                    raise self._err("缺少与之配对的左括号 (")
                return
            if ch == ".":
                if not allow_dot or prev is None:
                    raise self._err("点号 . 使用位置错误")
                self.i += 1
                prev = None
                pending_bond = None
                continue
            if ch in "-=#:/\\~@":
                pending_bond = self._read_bond()
                continue
            if ch.isdigit() or ch == "%":
                if prev is None:
                    raise self._err("环闭合编号必须跟在原子后面")
                self._read_ring_closure(prev, pending_bond)
                pending_bond = None
                continue
            # 否则应当是原子
            atom = self._parse_atom()
            if prev is not None:
                self._connect(prev, atom.idx, pending_bond)
            pending_bond = None
            prev = atom.idx
        if pending_bond is not None:
            raise self._err("键符号后缺少原子")
        if stop_at is not None:
            raise self._err(f"缺少右括号 {stop_at}")

    def _parse_branch(self, root: int) -> None:
        # 进入时 self.i 指向 '(' 之后
        prev = root
        pending_bond: Optional[dict] = None
        while self.i < self.n:
            ch = self.s[self.i]
            if ch == ")":
                if pending_bond is not None:
                    raise self._err("支链中的键符号后缺少原子")
                self.i += 1
                return
            if ch == "(":
                self.i += 1
                self._parse_branch(prev)
                continue
            if ch == ".":
                raise self._err("支链内部不允许点断 .")
            if ch in "-=#:/\\~@":
                pending_bond = self._read_bond()
                continue
            if ch.isdigit() or ch == "%":
                self._read_ring_closure(prev, pending_bond)
                pending_bond = None
                continue
            atom = self._parse_atom()
            self._connect(prev, atom.idx, pending_bond)
            pending_bond = None
            prev = atom.idx
        raise self._err("支链缺少右括号 )")

    # ---------- 键 ----------
    def _read_bond(self) -> dict:
        ch = self.s[self.i]
        pos = self.i
        if ch == "@":
            self.i += 1
            if self._peek() == "!":
                self.i += 1
                return {"in_ring": False}
            return {"in_ring": True}
        if ch == "~":
            self.i += 1
            return {"wild": True}
        if ch in "-=#:":
            self.i += 1
            return {"order": {"-": 1, "=": 2, "#": 3, ":": 1.5}[ch],
                    "aromatic": ch == ":"}
        if ch in "/\\":
            self.i += 1
            return {"order": 1, "stereo": ch}
        raise self._err(f"无法识别的键符号 {ch!r}", pos)

    def _connect(self, a1: int, a2: int,
                 bond_desc: Optional[dict]) -> None:
        aromatic = False
        order = 1
        wild = False
        in_ring_q: Optional[bool] = None
        if bond_desc is not None:
            if "wild" in bond_desc:
                wild = True
            if "in_ring" in bond_desc:
                in_ring_q = bond_desc["in_ring"]
            if "order" in bond_desc:
                if bond_desc["order"] == 1.5:
                    aromatic = True
                else:
                    order = int(bond_desc["order"])
        else:
            # 未写键：两个芳香原子之间默认芳香键，否则单键
            if self.mol.atoms[a1].aromatic and self.mol.atoms[a2].aromatic:
                aromatic = True
        self.mol.add_bond(a1, a2, order=order, aromatic=aromatic,
                          is_wildcard=wild, in_ring_query=in_ring_q)

    # ---------- 环闭合 ----------
    def _read_ring_closure(self, atom_idx: int,
                           bond_desc: Optional[dict]) -> None:
        ring_pos = self.i
        if self._peek() == "%":
            if self.i + 2 >= self.n or not (self.s[self.i + 1].isdigit()
                                            and self.s[self.i + 2].isdigit()):
                raise self._err("%% 环号必须是两位数字，如 %10")
            digit = int(self.s[self.i + 1:self.i + 3])
            self.i += 3
        else:
            digit = int(self.s[self.i])
            self.i += 1
        if digit == 0:
            raise self._err("环闭合编号 0 不合法，范围是 1..99")
        if digit in self.rings:
            other, first_desc, _ = self.rings.pop(digit)
            if other == atom_idx:
                raise self._err("环闭合不能连到原子自身")
            desc = bond_desc or first_desc
            if bond_desc and first_desc and \
                    bond_desc.get("order") != first_desc.get("order"):
                raise self._err("环闭合两端键级冲突")
            self._connect(other, atom_idx, desc)
        else:
            self.rings[digit] = (atom_idx, bond_desc, ring_pos)

    # ---------- 原子 ----------
    def _parse_atom(self) -> Atom:
        ch = self.s[self.i]
        if ch == "[":
            return self._parse_bracket()
        if ch == "*":
            self.i += 1
            a = Atom(idx=-1, symbol="*", is_wildcard=True)
            self.mol.add_atom(a)
            return a
        # 芳香小写
        if ch in AROMATIC_ORGANIC:
            self.i += 1
            a = Atom(idx=-1, symbol=ch, aromatic=True, explicit_h=-1)
            self.mol.add_atom(a)
            return a
        # 裸有机子集：两字符卤素 Cl/Br 优先（要求第二字符确实在串内），
        # 否则单字符 B C N O P S F I 等，避免把结尾的 O 误判成 Os。
        two = self.s[self.i:self.i + 2]
        if len(two) == 2 and two in ORGANIC_SUBSET:
            self.i += 2
            a = Atom(idx=-1, symbol=two, explicit_h=-1)
            self.mol.add_atom(a)
            return a
        if ch in ORGANIC_SUBSET:
            self.i += 1
            a = Atom(idx=-1, symbol=ch, explicit_h=-1)
            self.mol.add_atom(a)
            return a
        if ch.isalpha():
            raise self._err(
                f"元素 {two!r} 不在裸写有机子集内，请使用方括号 [{two}]")
        raise self._err(f"此处应为原子，却遇到 {ch!r}")

    def _parse_bracket(self) -> Atom:
        start = self.i
        self.i += 1  # 吃掉 [
        isotope = 0
        digits = ""
        while self._peek().isdigit():
            digits += self.s[self.i]
            self.i += 1
        if digits:
            isotope = int(digits)
            if not (1 <= isotope <= 999):
                raise self._err("同位素质量数范围 1..999", start)

        wildcard = False
        symbol = ""
        aromatic = False
        if self._peek() == "#":
            self.i += 1
            num = ""
            if not self._peek().isdigit():
                raise self._err("[# 后面必须跟原子序数")
            while self._peek().isdigit():
                num += self.s[self.i]
                self.i += 1
            z = int(num)
            if z <= 0 or z >= 119:
                raise self._err(f"原子序数 {z} 超出周期表范围")
            from .elements import SYMBOLS
            symbol = SYMBOLS[z]
        elif self._peek() == "*":
            self.i += 1
            wildcard = True
            symbol = "*"
        else:
            if self.i >= self.n:
                raise self._err("方括号内缺少元素符号")
            # 查询语境前瞻：单字符有机元素后紧跟 r<数字> 是环大小查询，
            # 不能让 Cr/Pr 等两字符元素符号抢先（如 [Cr6] = 碳且在六元环）。
            ch0 = self.s[self.i]
            after = self.s[self.i + 1:self.i + 3]
            ring_query_here = (
                ch0 in "BCNOPSFIbcnops"
                and len(after) >= 2 and after[0] == "r"
                and after[1].isdigit())
            if ring_query_here:
                symbol = ch0
                if ch0 in "bcnops":
                    aromatic = True
                self.i += 1
            else:
                two = self.s[self.i:self.i + 2]
                if two in _TWO_CHAR:
                    symbol = two
                    self.i += 2
                elif self._peek() in _ONE_CHAR:
                    symbol = self.s[self.i]
                    self.i += 1
                elif self._peek() in "bcnops":
                    symbol = self.s[self.i]
                    aromatic = True
                    self.i += 1
                else:
                    raise self._err("方括号内缺少元素符号")
            if not aromatic and symbol not in SYMBOL_TO_Z and symbol != "*":
                raise self._err(f"未知元素 {symbol!r}")

        h_count: Optional[int] = None   # None=未写（括号语义=0 氢）
        if self._peek() == "H":
            self.i += 1
            h_count = 1
            if self._peek().isdigit():
                h_count = int(self._read_int())

        # 立体标记（接受但不实现立体化学）
        if self._peek() == "@":
            self.i += 1
            if self._peek() == "@":
                self.i += 1

        charge = 0
        if self._peek() in "+-":
            if self.i >= self.n:
                raise self._err("电荷符号后内容缺失")
            sign_ch = self.s[self.i]
            self.i += 1
            run = 1
            while self._peek() == sign_ch:
                self.i += 1
                run += 1
            if self._peek().isdigit():
                charge = int(self._read_int()) * (1 if sign_ch == "+" else -1)
            else:
                charge = run * (1 if sign_ch == "+" else -1)
            if abs(charge) > 9:
                raise self._err("电荷绝对值不能超过 9")

        # 查询标记 R / !R / r6 / !r5
        in_ring_q: Optional[bool] = None
        ring_size_q: Optional[int] = None
        while self._peek() in "Rr!":
            if self.s[self.i:self.i + 2] == "!R":
                in_ring_q = False
                self.i += 2
            elif self.s[self.i:self.i + 2] == "!r":
                self.i += 2
                size = int(self._read_int())
                ring_size_q = -size  # 负数编码“不等于”
            elif self._peek() == "R":
                in_ring_q = True
                self.i += 1
            elif self._peek() == "r":
                self.i += 1
                ring_size_q = int(self._read_int())
            else:
                raise self._err("方括号内查询标记无法识别")

        class_label = None
        if self._peek() == ":":
            self.i += 1
            class_label = int(self._read_int())

        if self._peek() != "]":
            raise self._err(
                f"方括号没有以 ] 结束，遇到 {self._peek() or '结尾'}")
        self.i += 1

        a = Atom(
            idx=-1, symbol=symbol, charge=charge, isotope=isotope,
            explicit_h=0 if h_count is None else h_count,
            aromatic=aromatic, is_wildcard=wildcard,
            in_ring_query=in_ring_q,
            ring_size_query=ring_size_q,
            h_query=h_count, class_label=class_label,
        )
        self.mol.add_atom(a)
        return a

    def _read_int(self) -> str:
        out = ""
        while self._peek().isdigit():
            out += self.s[self.i]
            self.i += 1
        if not out:
            raise self._err("此处应为数字")
        return out


def parse_smiles(text: str) -> Molecule:
    """解析入口。返回带原始图的 Molecule（尚未做芳香性/隐氢规范化）。"""
    if not isinstance(text, str):
        raise ParseError("输入必须是字符串", 0)
    return Parser(text.strip()).parse()
