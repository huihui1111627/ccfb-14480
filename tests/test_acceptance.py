"""验收测试：指纹筛选零漏报、子结构查询与暴力匹配完全一致、相似性正确。"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chemdb import ChemDB, parse_smiles, fingerprint, has_substructure, \
    canonical_smiles
from chemdb.parser import parse_query
from chemdb.fingerprint import tanimoto
from chemdb.generator import generate_dataset


# 一组手写的、覆盖多种骨架与官能团的分子
SEED_MOLECULES = [
    "c1ccccc1", "CCc1ccccc1", "c1ccc(CC)cc1", "c1ccccc1C(=O)O",
    "CC(=O)Oc1ccccc1C(=O)O", "c1ccc2ccccc2c1", "c1ccncc1", "c1ccoc1",
    "c1ccsc1", "c1c[nH]cn1", "CCO", "CCCO", "CC(C)O", "CC(=O)O",
    "CC(=O)N", "CC#N", "C1CCCCC1", "C1CCCC1", "C1CCNCC1", "CC(C)C",
    "c1ccc(Cl)cc1", "c1ccc(O)c(C(=O)O)c1", "CCN(CC)CC", "O=C1CCCCC1",
    "c1ccccc1-c1ccccc1", "c1ccccc1Cc1ccccc1", "CC(C)c1ccccc1",
    "c1ccc2[nH]ccc2c1", "c1cnc2ccccc2n1", "CCSC", "OC(=O)C(O)C(=O)O",
]

# 一组查询片段（含通配、任意键、环内键、链键）
QUERIES = [
    "c1ccccc1", "CCO", "C(=O)O", "c1ccncc1", "c1ccoc1", "C1CCCCC1",
    "Cl", "O", "N", "C#N", "*", "a", "A", "c~rc", "C-kC", "*~*",
    "c1ccc", "CC", "C(=O)N", "c1ccsc1", "c~c", "C~O",
]


class TestFingerprintSubset(unittest.TestCase):

    def setUp(self):
        self.db = ChemDB()
        for s in SEED_MOLECULES:
            self.db.add_smiles(s)

    def test_subset_never_misses(self):
        """指纹位子集筛选的候选必须包含所有真实命中（零漏报）。"""
        for q in QUERIES:
            qmol = parse_query(q)
            qfp = fingerprint(qmol)
            candidates = set(self.db.fingerprint_candidates(qfp))
            # 暴力真值
            truth = set()
            for rid, rec in enumerate(self.db.records):
                tmol = self.db.get_mol(rid)
                if has_substructure(qmol, tmol):
                    truth.add(rid)
            with self.subTest(q=q):
                self.assertTrue(
                    truth.issubset(candidates),
                    "片段 %s 指纹筛选漏报 %s" % (q, truth - candidates))

    def test_fingerprint_filters_something(self):
        """指纹筛选确实能淘汰一部分分子（有筛选效果）。"""
        qfp = fingerprint(parse_query("c1ccncc1"))
        candidates = self.db.fingerprint_candidates(qfp)
        self.assertLess(len(candidates), len(self.db))


class TestSubstructureVsBruteForce(unittest.TestCase):

    def setUp(self):
        self.db = ChemDB()
        for s in SEED_MOLECULES:
            self.db.add_smiles(s)

    def test_exact_consistency(self):
        """数据库子结构查询结果与逐个暴力匹配完全一致。"""
        for q in QUERIES:
            db_hits = {h["id"] for h in self.db.substructure_search(q)}
            qmol = parse_query(q)
            brute = set()
            for rid in range(len(self.db)):
                if has_substructure(qmol, self.db.get_mol(rid)):
                    brute.add(rid)
            with self.subTest(q=q):
                self.assertEqual(db_hits, brute)

    def test_aromatic_vs_aliphatic(self):
        """芳香碳不能匹配脂肪碳，反之亦然。"""
        self.assertFalse(has_substructure(
            parse_smiles("c1ccccc1"), parse_smiles("C1CCCCC1")))
        self.assertFalse(has_substructure(
            parse_smiles("C1CCCCC1"), parse_smiles("c1ccccc1")))

    def test_bracket_explicit_h_query(self):
        """含 [nH] 的方括号氢查询不得报错，且能命中吡咯型 N。"""
        # 苯并咪唑骨架含 [nH]，能匹配自身及衍生物
        core = parse_smiles("c1ccc2[nH]cnc2c1")
        self.assertTrue(has_substructure(core, parse_smiles(
            "c1ccc2[nH]cnc2c1")))
        self.assertTrue(has_substructure(core, parse_smiles(
            "Cc1ccc2[nH]cnc2c1")))
        # [nH] 的氢是硬约束：目标 N 无氢（吡啶型）时不匹配
        self.assertFalse(has_substructure(
            parse_smiles("c1cc[nH]c1"), parse_smiles("c1ccncc1")))


class TestSimilarity(unittest.TestCase):

    def setUp(self):
        self.db = ChemDB()
        for s in SEED_MOLECULES:
            self.db.add_smiles(s)

    def test_tanimoto_bounds(self):
        for rec in self.db.records:
            self.assertGreaterEqual(rec.fp, 0)

    def test_similarity_matches_bruteforce(self):
        q = "c1ccccc1"
        threshold = 0.3
        results = self.db.similarity_search(q, threshold)
        qfp = fingerprint(parse_smiles(q))
        # 暴力计算真值
        truth = []
        for rid, rec in enumerate(self.db.records):
            score = tanimoto(qfp, rec.fp)
            if score >= threshold:
                truth.append((rid, round(score, 9)))
        got = [(r["id"], round(r["score"], 9)) for r in results]
        self.assertEqual(sorted(got), sorted(truth))

    def test_bound_pruning_keeps_results(self):
        """启用上下界剪枝不得改变结果集，只允许跳过计算。"""
        q = "CC(=O)O"
        pruned = {r["id"] for r in
                  self.db.similarity_search(q, 0.3, use_bound=True)}
        full = {r["id"] for r in
                self.db.similarity_search(q, 0.3, use_bound=False)}
        self.assertEqual(pruned, full)

    def test_self_similarity_is_one(self):
        rid, _ = self.db.add_smiles("c1ccncc1")
        res = self.db.similarity_search("c1ccncc1", 0.99)
        self.assertIn(rid, [r["id"] for r in res])
        top = max(res, key=lambda r: r["score"])
        self.assertAlmostEqual(top["score"], 1.0, places=9)


class TestScaffold(unittest.TestCase):

    def test_scaffold_frequency(self):
        db = ChemDB()
        for s in ["CCc1ccccc1", "Oc1ccccc1", "c1ccccc1C(=O)O",
                  "CCCO", "CCC"]:
            db.add_smiles(s)
        freq = db.scaffold_frequency(top=3)
        # 三个含苯环分子应归到同一苯骨架
        top_scaffold, count = freq[0]
        self.assertEqual(count, 3)
        self.assertIn("c1ccccc1", top_scaffold)

    def test_acyclic_no_scaffold(self):
        from chemdb import scaffold_key
        self.assertEqual(scaffold_key(parse_smiles("CCC")), "")


class TestRandomDataset(unittest.TestCase):

    def test_generated_are_valid_and_canonical(self):
        data = generate_dataset(200, seed=7)
        self.assertGreaterEqual(len(data), 150)
        for smi in data:
            m = parse_smiles(smi)
            # 规范形式幂等
            c1 = canonical_smiles(m)
            c2 = canonical_smiles(parse_smiles(c1))
            self.assertEqual(c1, c2, smi)


if __name__ == "__main__":
    unittest.main(verbosity=2)
