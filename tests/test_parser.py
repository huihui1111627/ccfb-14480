# -*- coding: utf-8 -*-
"""解析器单元测试：覆盖需求 1 的全部语法点与错误定位。"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from molkit.parser import ParseError, parse_smiles


class TestParser(unittest.TestCase):
    def test_basic_atoms_bonds(self):
        m = parse_smiles("CCO")
        self.assertEqual(len(m.atoms), 3)
        self.assertEqual(len(m.bonds), 2)
        self.assertEqual([b.order for b in m.bonds], [1, 1])

    def test_bond_orders(self):
        self.assertEqual(parse_smiles("C=O").bonds[0].order, 2)
        self.assertEqual(parse_smiles("C#N").bonds[0].order, 3)
        self.assertEqual(parse_smiles("C-C").bonds[0].order, 1)

    def test_branches(self):
        m = parse_smiles("CC(C)(C)C")
        self.assertEqual(len(m.atoms), 5)
        # 中心碳度为 4
        center = max(range(5), key=lambda i: m.degree(i))
        self.assertEqual(m.degree(center), 4)

    def test_ring_closure_single_digit(self):
        m = parse_smiles("C1CCCCC1")
        self.assertEqual(len(m.atoms), 6)
        self.assertEqual(len(m.bonds), 6)
        self.assertTrue(m.ring_bonds())

    def test_ring_closure_percent(self):
        m = parse_smiles("C%10CCCCC%10")
        self.assertEqual(len(m.atoms), 6)
        self.assertEqual(len(m.ring_bonds()), 6)

    def test_charge(self):
        m = parse_smiles("[NH4+]")
        self.assertEqual(m.atoms[0].charge, 1)
        m2 = parse_smiles("[O-]")
        self.assertEqual(m2.atoms[0].charge, -1)
        m3 = parse_smiles("[Fe+3]")
        self.assertEqual(m3.atoms[0].charge, 3)

    def test_isotope(self):
        m = parse_smiles("[13CH3-]")
        self.assertEqual(m.atoms[0].isotope, 13)
        self.assertEqual(m.atoms[0].charge, -1)
        m2 = parse_smiles("[2H]O[H]")
        self.assertEqual(m2.atoms[0].isotope, 2)

    def test_aromatic_lowercase(self):
        m = parse_smiles("c1ccccc1")
        self.assertTrue(all(a.aromatic for a in m.atoms))
        self.assertTrue(all(b.aromatic for b in m.bonds))

    def test_explicit_hydrogen(self):
        m = parse_smiles("[nH]1cccc1")
        n = next(a for a in m.atoms if a.symbol == "n")
        self.assertEqual(n.explicit_h, 1)

    def test_implicit_vs_explicit(self):
        bare = parse_smiles("N")
        bracketed = parse_smiles("[NH3]")
        self.assertEqual(bare.atoms[0].explicit_h, -1)
        self.assertEqual(bracketed.atoms[0].explicit_h, 3)

    def test_disconnected(self):
        m = parse_smiles("CC(=O)[O-].[Na+]")
        self.assertEqual(len(m.compute_fragments()), 2)

    def test_wildcard_atom(self):
        m = parse_smiles("*CCO")
        self.assertTrue(m.atoms[0].is_wildcard)

    def test_atomic_number_bracket(self):
        m = parse_smiles("[#6]1ccccc1")
        self.assertEqual(m.atoms[0].symbol, "C")

    def test_query_wild_bond(self):
        m = parse_smiles("C~C")
        self.assertTrue(m.bonds[0].is_wildcard)

    def test_query_ring_bond(self):
        m = parse_smiles("C@C")
        self.assertEqual(m.bonds[0].in_ring_query, True)

    def test_ring_atom_query(self):
        m = parse_smiles("[CR]")
        self.assertEqual(m.atoms[0].in_ring_query, True)
        m2 = parse_smiles("[C!R]")
        self.assertEqual(m2.atoms[0].in_ring_query, False)

    def test_ring_size_query_disambiguation(self):
        # [Cr6] 是“碳 + 六元环查询”，不是铬元素
        m = parse_smiles("[Cr6]")
        self.assertEqual(m.atoms[0].symbol, "C")
        self.assertEqual(m.atoms[0].ring_size_query, 6)
        # 真正的铬
        self.assertEqual(parse_smiles("[Cr]").atoms[0].symbol, "Cr")

    # ---- 错误定位 ----
    def assert_error_column(self, text, column):
        with self.assertRaises(ParseError) as ctx:
            parse_smiles(text)
        self.assertEqual(ctx.exception.column, column)

    def test_error_unclosed_branch(self):
        self.assert_error_column("C(C", 4)

    def test_error_unclosed_ring(self):
        self.assert_error_column("CC1", 3)

    def test_error_bad_bracket(self):
        self.assert_error_column("[CH", 4)

    def test_error_percent_ring(self):
        self.assert_error_column("C%1", 2)

    def test_error_dangling_bond(self):
        self.assert_error_column("C#", 3)

    def test_error_ring_at_start(self):
        self.assert_error_column("1CC", 1)

    def test_error_double_dot(self):
        self.assert_error_column("C..O", 3)

    def test_error_empty(self):
        self.assert_error_column("", 1)

    def test_error_isotope_range(self):
        self.assert_error_column("[1234C]", 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
