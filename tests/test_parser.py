"""解析器单元测试：覆盖原子、键级、支链、环、电荷、同位素、芳香、氢、错误定位。"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chemdb import parse_smiles, canonical_smiles, ParseError
from chemdb.parser import parse_query


class TestParser(unittest.TestCase):

    def test_basic_atoms_bonds(self):
        m = parse_smiles("CCO")
        self.assertEqual(m.n_atoms(), 3)
        self.assertEqual(m.n_bonds(), 2)
        self.assertEqual([b.order for b in m.bonds], [1, 1])
        self.assertEqual(m.formula(), "C2H6O")

    def test_bond_orders(self):
        self.assertEqual([b.order for b in parse_smiles("CC#N").bonds],
                         [1, 3])
        self.assertEqual([b.order for b in parse_smiles("C=O").bonds], [2])

    def test_branches(self):
        m = parse_smiles("CC(C)C")
        self.assertEqual(m.n_atoms(), 4)
        # 中心碳度为 3
        center = max(range(4), key=lambda i: m.degree(i))
        self.assertEqual(m.degree(center), 3)
        # 分支以键符号开头
        self.assertEqual(parse_smiles("CC(=O)O").n_atoms(), 4)

    def test_ring_closure(self):
        m = parse_smiles("C1CCCCC1")
        self.assertEqual(m.n_atoms(), 6)
        self.assertEqual(m.n_bonds(), 6)
        self.assertTrue(all(b.in_ring for b in m.bonds))

    def test_percent_ring_number(self):
        m = parse_smiles("C%10CCCCC%10")
        self.assertEqual(m.n_bonds(), m.n_atoms())

    def test_isotope(self):
        m = parse_smiles("[13CH4]")
        self.assertEqual(m.atoms[0].isotope, 13)
        self.assertEqual(m.total_h(0), 4)

    def test_charge(self):
        m = parse_smiles("[NH4+]")
        self.assertEqual(m.atoms[0].charge, 1)
        m2 = parse_smiles("[O-]C")
        self.assertEqual(m2.atoms[0].charge, -1)
        m3 = parse_smiles("[Fe+3]")
        self.assertEqual(m3.atoms[0].charge, 3)

    def test_explicit_h(self):
        m = parse_smiles("[nH]1cccc1")
        self.assertEqual(m.atoms[0].explicit_h, 1)
        self.assertTrue(m.atoms[0].aromatic)

    def test_implicit_h(self):
        m = parse_smiles("C")
        self.assertEqual(m.total_h(0), 4)
        m2 = parse_smiles("O")
        self.assertEqual(m2.total_h(0), 2)

    def test_aromatic_lowercase(self):
        m = parse_smiles("c1ccccc1")
        self.assertTrue(all(a.aromatic for a in m.atoms))
        self.assertTrue(all(b.order == 4 for b in m.bonds))

    def test_multi_fragment(self):
        m = parse_smiles("CCO.[Na+]")
        self.assertEqual(len(m.components()), 2)

    def test_query_wildcards(self):
        self.assertEqual(parse_query("*").atoms[0].wildcard, True)
        self.assertTrue(parse_query("a").atoms[0].aromatic_query)
        self.assertTrue(parse_query("A").atoms[0].aliphatic_query)
        q = parse_query("c~rc")
        self.assertTrue(any(b.is_ring_query for b in q.bonds))
        q2 = parse_query("C-kC")
        self.assertTrue(any(b.is_chain_query for b in q2.bonds))
        q3 = parse_query("C~O")
        self.assertTrue(any(b.order == 0 for b in q3.bonds))

    # ---- 错误定位 ----
    def assertErrorPos(self, text, pos):
        with self.assertRaises(ParseError) as ctx:
            parse_smiles(text)
        self.assertEqual(ctx.exception.position, pos)

    def test_error_positions(self):
        self.assertErrorPos("", 1)
        self.assertErrorPos("CC(", 4)
        self.assertErrorPos("C1CC", 4)
        self.assertErrorPos("(CC)", 1)
        self.assertErrorPos("CC))", 3)
        self.assertErrorPos("[C", 3)
        self.assertErrorPos("1CC", 1)
        self.assertErrorPos("CC?", 3)

    def test_invalid_aromatic_ring(self):
        with self.assertRaises(ParseError):
            parse_smiles("c1cccc1")  # 5 个芳香碳，不满足 4n+2


class TestCanonical(unittest.TestCase):

    def test_idempotent(self):
        for t in ["c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O", "c1ccc(CC)cc1",
                  "c1ccc2ccccc2c1", "CC(C)(C)c1ccccc1", "C12(CCCCC1)CCCCC2"]:
            c1 = canonical_smiles(parse_smiles(t))
            c2 = canonical_smiles(parse_smiles(c1))
            self.assertEqual(c1, c2, t)

    def test_aromatic_kekule_unify(self):
        pairs = [
            ("C1=CC=CC=C1", "c1ccccc1"),
            ("C1=CC=NC=C1", "c1ccncc1"),
            ("C1=CC=CO1", "c1ccoc1"),
            ("C1=CC2=CC=CC=C2C=C1", "c1ccc2ccccc2c1"),
            ("C1=CC=C2C=CC=CC2=C1", "c1ccc2ccccc2c1"),
            ("C1=CC=CS1", "c1ccsc1"),
        ]
        for a, b in pairs:
            self.assertEqual(canonical_smiles(parse_smiles(a)),
                             canonical_smiles(parse_smiles(b)),
                             "%s vs %s" % (a, b))

    def test_writing_invariance(self):
        # 同一分子不同书写顺序得到同一规范形式
        variants = ["c1ccccc1", "c1ccccc1", "C1=CC=CC=C1"]
        cans = {canonical_smiles(parse_smiles(v)) for v in variants}
        self.assertEqual(len(cans), 1)

    def test_dedup_by_canonical(self):
        from chemdb import ChemDB
        db = ChemDB()
        _, new1 = db.add_smiles("c1ccccc1")
        _, new2 = db.add_smiles("C1=CC=CC=C1")
        self.assertTrue(new1)
        self.assertFalse(new2)
        self.assertEqual(len(db), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
