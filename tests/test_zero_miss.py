"""零漏报专项：随机库上对多种片段验证指纹筛选 + 图匹配与暴力完全一致。

控制规模以保证可在约一分钟内跑完；统计意义由多样化查询保证。
"""
import os
import sys
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chemdb import ChemDB, fingerprint
from chemdb.parser import parse_query
from chemdb.matcher import has_substructure
from chemdb.generator import generate_dataset

# 覆盖原子、官能团、芳香/脂肪环、查询扩展的多样化片段
QUERIES = [
    "c1ccccc1", "c1ccncc1", "c1ccoc1", "c1ccsc1", "c1c[nH]cn1",
    "C(=O)O", "C(=O)N", "CCO", "CC", "Cl", "N", "O", "C#N",
    "C1CCCCC1", "C1CCCC1", "c~rc", "C-kC", "*~*", "a", "A",
    "c1ccc", "C~O", "c~c",
]


def main(n_db=600, seed=123):
    data = generate_dataset(n_db, seed=seed)
    db = ChemDB()
    for s in data:
        db.add_smiles(s)

    total_miss = 0
    total_mismatch = 0
    for q in QUERIES:
        qmol = parse_query(q)
        candidates = set(db.fingerprint_candidates(fingerprint(qmol)))
        brute = {rid for rid in range(len(db))
                 if has_substructure(qmol, db.get_mol(rid))}
        refined = {rid for rid in candidates
                   if has_substructure(qmol, db.get_mol(rid))}
        total_miss += len(brute - candidates)   # 指纹漏报，必须为 0
        if refined != brute:
            total_mismatch += 1

    print("库大小 %d，查询片段 %d 个" % (len(db), len(QUERIES)))
    print("指纹漏报总数: %d（必须为 0）" % total_miss)
    print("图匹配与暴力不一致查询数: %d（必须为 0）" % total_mismatch)
    assert total_miss == 0 and total_mismatch == 0
    print("零漏报 & 完全一致 验证通过")


if __name__ == "__main__":
    main()
