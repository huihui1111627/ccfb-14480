# -*- coding: utf-8 -*-
"""化学语义、规范化、指纹与匹配的集成测试。"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from molkit.aromatic import standardize
from molkit.canonical import canonical_hash, canonicalize, canonical_smiles
from molkit.database import MoleculeDatabase
from molkit.fingerprint import (fingerprint_bits, is_subset_fp, tanimoto,
                                 tanimoto_bounds)
from molkit.matcher import has_substructure, prepare_query
from molkit.parser import parse_smiles
from molkit.scaffold import scaffold_key


def T(s):
    return standardize(parse_smiles(s))


def Q(s):
    return prepare_query(parse_smiles(s))


class TestAromatic(unittest.TestCase):
    def _canon_hash(self, s):
        # 芳香/凯库勒写法的等价性最终体现在规范哈希上（原子顺序无关）
        return canonical_hash(canonicalize(T(s)))

    def test_benzene_unified(self):
        self.assertEqual(self._canon_hash("c1ccccc1"),
                         self._canon_hash("C1=CC=CC=C1"))

    def test_pyridine_unified(self):
        self.assertEqual(self._canon_hash("c1ccncc1"),
                         self._canon_hash("C1=CC=NC=C1"))

    def test_furan_unified(self):
        self.assertEqual(self._canon_hash("c1ccoc1"),
                         self._canon_hash("C1=COC=C1"))

    def test_pyrrole_unified(self):
        self.assertEqual(self._canon_hash("c1cc[nH]c1"),
                         self._canon_hash("C1=CNC=C1"))

    def test_thiophene_unified(self):
        self.assertEqual(self._canon_hash("c1ccsc1"),
                         self._canon_hash("C1=CSC=C1"))

    def test_no_aromatic_mark_after(self):
        m = T("c1ccccc1")
        self.assertFalse(any(a.aromatic for a in m.atoms))


class TestCanonical(unittest.TestCase):
    def _hash(self, s):
        return canonical_hash(canonicalize(T(s)))

    def test_equivalent_writings_same_hash(self):
        for group in [
            ["CCO", "OCC", "C(O)C"],
            ["c1ccccc1", "C1=CC=CC=C1"],
            ["CC(=O)O", "CC(O)=O", "O=C(O)C"],
            ["c1cc[nH]c1", "C1=CNC=C1"],
            ["c1ccc([Cl])cc1", "Clc1ccccc1"],
        ]:
            self.assertEqual(len({self._hash(s) for s in group}), 1, group)

    def test_different_molecules_differ(self):
        self.assertNotEqual(self._hash("CCO"), self._hash("CCN"))

    def test_deterministic(self):
        self.assertEqual(self._hash("c1ccccc1C"), self._hash("c1ccccc1C"))


class TestMatcher(unittest.TestCase):
    def test_aromatic_vs_aliphatic_strict(self):
        self.assertTrue(has_substructure(Q("c1ccccc1"), T("Cc1ccccc1")))
        self.assertFalse(has_substructure(Q("C1CCCCC1"), T("c1ccccc1")))
        self.assertFalse(has_substructure(Q("c1ccccc1"), T("C1CCCCC1")))

    def test_heteroatom_distinction(self):
        self.assertTrue(has_substructure(Q("c1ccncc1"), T("c1ccncc1C")))
        self.assertFalse(has_substructure(Q("c1ccncc1"), T("c1ccccc1")))

    def test_wildcards(self):
        self.assertTrue(has_substructure(Q("*O"), T("CCO")))
        self.assertTrue(has_substructure(Q("C~C"), T("CCN")))

    def test_ring_queries(self):
        self.assertTrue(has_substructure(Q("[CR]"), T("C1CCCCC1")))
        self.assertFalse(has_substructure(Q("[C!R]"), T("C1CCCCC1")))
        self.assertTrue(has_substructure(Q("[cr6]"), T("c1ccccc1")))
        self.assertFalse(has_substructure(Q("[cr5]"), T("c1ccccc1")))

    def test_bond_order_sensitive(self):
        self.assertTrue(has_substructure(Q("C#N"), T("CC#N")))
        self.assertFalse(has_substructure(Q("C#N"), T("CCN")))


class TestFingerprint(unittest.TestCase):
    def test_subset_no_false_negative(self):
        for q, t in [("c1ccccc1", "Cc1ccccc1"), ("CCO", "CCCO"),
                     ("C=O", "CC(=O)O"), ("c1ccncc1", "c1ccncc1C"),
                     ("Cl", "Clc1ccccc1")]:
            qbits = fingerprint_bits(Q(q))
            tbits = fingerprint_bits(T(t))
            self.assertTrue(is_subset_fp(qbits, tbits), q)

    def test_tanimoto_range_and_self(self):
        bits = fingerprint_bits(T("CCO"))
        self.assertAlmostEqual(tanimoto(bits, bits), 1.0)
        self.assertGreaterEqual(tanimoto(bits, fingerprint_bits(T("CCN"))),
                                0.0)

    def test_tanimoto_bounds(self):
        lo, hi = tanimoto_bounds(5, 8)
        self.assertTrue(0.0 <= lo <= hi <= 1.0)
        # 相同位数的上界
        _, hi_same = tanimoto_bounds(10, 10)
        self.assertAlmostEqual(hi_same, 1.0)


class TestDatabase(unittest.TestCase):
    def setUp(self):
        self.db = MoleculeDatabase()
        for s in ["CCO", "OCC", "c1ccccc1", "Cc1ccccc1", "CC(=O)O",
                  "c1ccncc1", "C1CCCCC1"]:
            self.db.add_smiles(s)

    def test_dedup(self):
        self.assertEqual(len(self.db), 6)  # CCO 与 OCC 同物
        _, is_new = self.db.add_smiles("OCC")
        self.assertFalse(is_new)

    def test_substructure_matches_brute_force(self):
        from molkit.matcher import substructure_matches
        for q in ["c1ccccc1", "C1CCCCC1", "C=O", "C~C", "*O"]:
            qmol = Q(q)
            hits, _ = self.db.substructure_search(q)
            indexed = {h["mol_id"] for h in hits}
            brute = {r.mol_id for r in self.db.records
                     if substructure_matches(qmol, r.canon_mol, 1)}
            self.assertEqual(indexed, brute, q)

    def test_similarity_threshold(self):
        hits, stats = self.db.similarity_search("CCO", 0.3)
        self.assertTrue(stats.filtered >= 0)
        for h in hits:
            self.assertGreaterEqual(h["tanimoto"], 0.3 - 1e-9)

    def test_scaffold_grouping(self):
        counts = dict(self.db.scaffold_counts())
        # 苯与甲苯共享苯环骨架
        self.assertGreaterEqual(counts.get(
            canonical_smiles(canonicalize(T("c1ccccc1"))), 0), 1)


class TestScaffold(unittest.TestCase):
    def test_acyclic_no_scaffold(self):
        self.assertEqual(scaffold_key(T("CCO")), "")

    def test_ring_preserved(self):
        key = scaffold_key(T("Cc1ccccc1"))
        self.assertIn("C", key)
        self.assertTrue(key)  # 非空骨架


if __name__ == "__main__":
    unittest.main(verbosity=2)
