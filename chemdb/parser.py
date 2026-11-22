"""线性结构式（SMILES 方言）递归下降解析器。

支持：
  原子（有机子集裸原子、方括号原子）、键级 - = # : / \\、支链 ()、
  环闭合（数字与 %nn）、电荷、同位素、芳香小写原子、显式氢 [nH]、
  隐式氢（由价键规则补全）、断开的多片段 '.'、通配原子 * / a / A。

查询扩展语法（仅片段查询使用，普通分子也能解析但会带查询标记）：
  键符号后紧跟 r 表示“环内键”，c 表示“链键”，如 c1ccccc1-C~rC
  ~ 表示任意键；* 任意原子；a 任意芳香原子；A 任意脂肪原子。

解析失败抛 ParseError，其中 position 为从 1 开始的字符位置。
不支持立体化学（@ / @@），遇到时明确报错，避免静默忽略语义。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .elements import (AROMATIC_SYMBOLS, AROMATIC_TO_ALIPHATIC,
                       ORGANIC_SUBSET)
from .molecule import Atom, Mol, SYMBOL_BOND
from .aromaticity import normalize_aromaticity, AromaticityError
from .valence import compute_implicit_h


class ParseError(Exception):
    def __init__(self, message: str, position: int):
        # position 为 1-based
        self.position = position
        super().__init__("第 %d 个字符: %s" % (position, message))


# 两字符的裸原子符号（顺序敏感，先长后短）
_TWO_CHAR = ("Cl", "Br", "se")


class SmilesParser:
    def __init__(self, text: str, query: bool = False):
        self.text = text
        self.n = len(text)
        self.pos = 0
        self.query = query  # 查询片段模式：放宽芳香成环等整分子约束
        self.mol = Mol()
        # ring number -> (atom idx, bond order or None, ring_q, chain_q)
        self.ring_open: Dict[int, Tuple[int, Optional[int], bool, bool]] = {}

    # ---------- 基础工具 ----------
    def peek(self, off: int = 0) -> str:
        j = self.pos + off
        return self.text[j] if j < self.n else ""

    def error(self, msg: str) -> ParseError:
        return ParseError(msg, self.pos + 1)

    def parse(self) -> Mol:
        if self.n == 0:
            raise ParseError("空结构式", 1)
        if self.text[0] == "(":
            raise ParseError("结构式不能以分支 '(' 开头", 1)
        self._parse_chain(head=True)
        while self.peek() == ".":
            self.pos += 1
            if self.peek() in (")", ".", ""):
                raise self.error("'.' 后缺少片段")
            self._parse_chain(head=True)
        if self.pos != self.n:
            raise self.error("多余的字符 '%s'" % self.peek())
        if self.ring_open and not self.query:
            num = next(iter(self.ring_open))
            raise ParseError("环编号 %d 没有闭合" % num, self.n)
        # 查询模式下未闭合的环号只是局部模式（如 c1ccc），无需补键。
        # 只需线性时间的环内键标记；最小环基在真正需要时才惰性计算。
        from .rings import _flag_ring_bonds
        _flag_ring_bonds(self.mol)
        try:
            normalize_aromaticity(self.mol, allow_loose=self.query)
        except AromaticityError as exc:
            raise ParseError(str(exc), self.n)
        compute_implicit_h(self.mol)
        return self.mol

    # ---------- 链 / 支链 ----------
    def _parse_chain(self, head: bool) -> None:
        current = self._parse_atom()
        self._parse_chain_tail(current)

    def _parse_chain_tail(self, current: int) -> None:
        while True:
            ch = self.peek()
            if ch in ("", ")", "."):
                return
            if ch == "(":
                self.pos += 1
                # 分支可以键符号开头，如 CC(=O)O：先读出键前缀
                order, ring_q, chain_q, any_bond = self._parse_bond_prefix()
                branch_atom = self._parse_atom()
                self._connect(current, branch_atom, order, ring_q,
                              chain_q, any_bond)
                self._consume_ring_numbers(branch_atom)
                self._parse_chain_tail(branch_atom)
                if self.peek() != ")":
                    raise self.error("缺少右括号 ')'")
                self.pos += 1
                continue
            order, ring_q, chain_q, any_bond = self._parse_bond_prefix()
            # 环号若直接跟在键符号后（无新原子），挂在 current 上；
            # 否则先读原子，环号作为该原子的修饰符随后消费。
            if self._is_ring_digit_ahead():
                self._consume_ring_numbers(
                    current, order, ring_q, chain_q, any_bond)
                continue
            nxt = self._parse_atom()
            self._connect(current, nxt, order, ring_q, chain_q, any_bond)
            self._consume_ring_numbers(nxt)
            current = nxt

    def _consume_ring_numbers(self, atom: int,
                              bond_order=None, ring_q: bool = False,
                              chain_q: bool = False,
                              any_bond: bool = False) -> None:
        """消费原子后面连续的一个或多个环编号。"""
        while self._is_ring_digit_ahead():
            self._handle_ring(atom, bond_order, ring_q, chain_q, any_bond)
            bond_order = None  # 仅第一个环号可携带键符号
            ring_q = chain_q = False
            any_bond = False

    def _connect(self, a1: int, a2: int,
                 order: Optional[int], ring_q: bool, chain_q: bool,
                 any_bond: bool) -> None:
        if any_bond:
            real_order = 1  # 查询用 ~，存储时按单键承载匹配标记
        elif order is not None:
            real_order = order
        else:
            # 默认键：两个芳香原子之间为芳香键，否则单键
            if (self.mol.atoms[a1].aromatic
                    and self.mol.atoms[a2].aromatic):
                real_order = 4
            else:
                real_order = 1
        self.mol.add_bond(a1, a2, real_order,
                          is_ring_query=ring_q,
                          is_chain_query=chain_q)
        if any_bond:
            # order=0 约定为“任意键”查询
            self.mol.bonds[-1].order = 0

    # ---------- 键前缀 ----------
    def _parse_bond_prefix(self) -> Tuple[Optional[int], bool, bool, bool]:
        ch = self.peek()
        order: Optional[int] = None
        any_bond = False
        if ch == "~":
            self.pos += 1
            any_bond = True
            ch = self.peek()
        elif ch in SYMBOL_BOND:
            if ch in ("/", "\\"):
                raise self.error("本系统不支持立体化学键 '/' '\\\\'")
            order = SYMBOL_BOND[ch]
            self.pos += 1
            ch = self.peek()
        ring_q = chain_q = False
        if ch == "r":
            ring_q = True
            self.pos += 1
        elif ch == "k" and (order is not None or any_bond):
            # 链键查询标记用 k（避免与芳香碳 c 冲突），如 -k、=k、~k
            chain_q = True
            self.pos += 1
        return order, ring_q, chain_q, any_bond

    def _is_ring_digit_ahead(self) -> bool:
        ch = self.peek()
        return ch.isdigit() or ch == "%"

    def _handle_ring(self, atom: int, order: Optional[int],
                     ring_q: bool, chain_q: bool, any_bond: bool) -> None:
        num = self._read_ring_number()
        if num in self.ring_open:
            first, forder, frq, fcq = self.ring_open.pop(num)
            if any_bond:
                real_order = 0
            elif order is not None:
                real_order = order
            elif forder is not None:
                real_order = forder
            elif (self.mol.atoms[first].aromatic
                  and self.mol.atoms[atom].aromatic):
                real_order = 4
            else:
                real_order = 1
            self.mol.add_bond(first, atom, real_order,
                              is_ring_query=ring_q or frq,
                              is_chain_query=chain_q or fcq)
            if any_bond:
                self.mol.bonds[-1].order = 0
        else:
            self.ring_open[num] = (atom, order, ring_q, chain_q)

    def _read_ring_number(self) -> int:
        if self.peek() == "%":
            self.pos += 1
            if not (self.peek().isdigit() and self.peek(1).isdigit()):
                raise self.error("'%%' 后需要两位环编号")
            num = int(self.text[self.pos:self.pos + 2])
            self.pos += 2
            return num
        num = int(self.peek())
        self.pos += 1
        return num

    # ---------- 原子 ----------
    def _parse_atom(self) -> int:
        ch = self.peek()
        if ch == "":
            raise self.error("缺少原子")
        if ch == "[":
            return self._parse_bracket()
        if ch == "*":
            self.pos += 1
            return self.mol.add_atom(Atom(0, "*", wildcard=True))
        if ch == "a":
            self.pos += 1
            return self.mol.add_atom(Atom(0, "*", aromatic_query=True))
        if ch == "A":
            self.pos += 1
            return self.mol.add_atom(Atom(0, "*", aliphatic_query=True))
        two = self.text[self.pos:self.pos + 2]
        if two in _TWO_CHAR:
            self.pos += 2
            if two == "se":
                return self.mol.add_atom(Atom(0, "se", aromatic=True))
            return self.mol.add_atom(Atom(0, two))
        if ch in ORGANIC_SUBSET:
            self.pos += 1
            return self.mol.add_atom(Atom(0, ch))
        if ch in AROMATIC_SYMBOLS:
            self.pos += 1
            return self.mol.add_atom(Atom(0, ch, aromatic=True))
        if ch.isalpha():
            raise self.error("元素 '%s' 不在有机子集内，需写方括号" % ch)
        raise self.error("无法识别的原子写法 '%s'" % ch)

    def _parse_bracket(self) -> int:
        start = self.pos
        self.pos += 1  # 吃掉 [
        isotope = None
        m = 0
        while self.peek().isdigit():
            m = m * 10 + int(self.peek())
            self.pos += 1
        if m:
            isotope = m
        symbol, aromatic = self._read_bracket_symbol()
        explicit_h = 0
        if self.peek() == "H":
            self.pos += 1
            hc = 0
            while self.peek().isdigit():
                hc = hc * 10 + int(self.peek())
                self.pos += 1
            explicit_h = hc if hc else 1
        charge = self._read_charge()
        if self.peek() == ":":  # 原子类，跳过
            self.pos += 1
            while self.peek().isdigit():
                self.pos += 1
        if self.peek() != "]":
            raise self.error("方括号原子缺少 ']'")
        self.pos += 1
        atom = Atom(0, symbol, charge=charge, isotope=isotope,
                    explicit_h=explicit_h, bracket=True, aromatic=aromatic)
        return self.mol.add_atom(atom)

    def _read_bracket_symbol(self) -> Tuple[str, bool]:
        ch = self.peek()
        if ch == "":
            raise self.error("'[' 后缺少元素符号")
        two = self.text[self.pos:self.pos + 2]
        if two == "se":
            self.pos += 2
            return "se", True
        if ch.islower() and ch in AROMATIC_SYMBOLS:
            self.pos += 1
            return ch, True
        if ch.isupper():
            self.pos += 1
            if self.peek().islower() and self.peek().isalpha():
                ch2 = self.peek()
                self.pos += 1
                return ch + ch2, False
            return ch, False
        raise self.error("方括号内元素符号非法 '%s'" % ch)

    def _read_charge(self) -> int:
        ch = self.peek()
        if ch not in ("+", "-"):
            return 0
        sign = 1 if ch == "+" else -1
        self.pos += 1
        if self.peek() in ("+", "-"):
            # ++ / -- 形式
            if self.peek() != ch:
                raise self.error("电荷写法不一致")
            self.pos += 1
            return sign * 2
        num = 0
        while self.peek().isdigit():
            num = num * 10 + int(self.peek())
            self.pos += 1
        return sign * (num if num else 1)


def parse_smiles(text: str) -> Mol:
    """解析入口：文本 -> 分子图（已完成环标记、芳香归一、隐式氢）。"""
    if not isinstance(text, str):
        raise ParseError("输入必须是字符串", 1)
    text = text.strip()
    return SmilesParser(text).parse()


def parse_query(text: str) -> Mol:
    """解析查询片段：支持 * / a / A / ~ / -r / -k 等查询语法，
    并放宽芳香原子必须成环等整分子约束。"""
    if not isinstance(text, str):
        raise ParseError("输入必须是字符串", 1)
    text = text.strip()
    return SmilesParser(text, query=True).parse()
