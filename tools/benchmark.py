# -*- coding: utf-8 -*-
"""验收基准：覆盖规范一致性、子结构正确性、相似性、指纹筛选率与耗时曲线。

运行：
  python3 -I tools/benchmark.py --n 50000
全部检查：
  1. 规范形式一致性：等价写法必须同哈希；随机分子去重后重复入库不新增。
  2. 子结构查询：指纹预筛 + 精确匹配的结果，与逐分子暴力匹配完全一致。
  3. 相似性查询：结果与直接全量 Tanimoto 一致，报告位数上界剪枝比例。
  4. 指纹筛选比例与查询耗时随库大小的变化（多个规模采样）。
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from molkit.aromatic import standardize          # noqa: E402
from molkit.canonical import canonicalize, canonical_hash  # noqa: E402
from molkit.database import MoleculeDatabase     # noqa: E402
from molkit.fingerprint import (fingerprint_bits, tanimoto,  # noqa: E402
                                 could_meet_threshold)
from molkit.generator import generate_distinct   # noqa: E402
from molkit.matcher import prepare_query, substructure_matches  # noqa: E402
from molkit.parser import parse_smiles           # noqa: E402

# 覆盖各种写法的固定用例（等价写法分组）
EQUIVALENCE_GROUPS = [
    ["CCO", "OCC", "C(O)C"],
    ["c1ccccc1", "C1=CC=CC=C1"],
    ["c1ccncc1", "C1=CC=NC=C1"],
    ["c1ccoc1", "C1=COC=C1"],
    ["c1cc[nH]c1", "C1=CNC=C1"],
    ["c1ccsc1", "C1=CSC=C1"],
    ["CC(=O)O", "CC(O)=O", "O=C(O)C"],
    ["c1ccc2ccccc2c1", "C1=CC=C2C=CC=CC2=C1"],
    ["c1ccccc1C", "Cc1ccccc1", "C1=CC=CC=C1C"],
    ["c1ccc([Cl])cc1", "Clc1ccccc1"],
    ["CC(=O)[O-]", "CC([O-])=O"],
    ["[NH4+]", "[NH4+]"],
]

# 子结构查询用例（片段, 描述）
SUBSTRUCTURE_QUERIES = [
    ("c1ccccc1", "苯环（芳香）"),
    ("C1CCCCC1", "环己烷（脂肪环，不应匹配苯）"),
    ("c1ccncc1", "吡啶环"),
    ("C=O", "羰基"),
    ("C#N", "腈基"),
    ("CCO", "乙醇骨架"),
    ("Cl", "氯原子"),
    ("c1ccoc1", "呋喃环"),
    ("[CR]", "环内碳"),
    ("C~C", "任意键连接的两个碳"),
    ("*O", "连在任意原子上的氧"),
]


def check_equivalence() -> bool:
    print("\n[1] 规范形式一致性（等价写法同哈希）")
    ok = True
    for group in EQUIVALENCE_GROUPS:
        hashes, texts = set(), set()
        for s in group:
            mol = canonicalize(standardize(parse_smiles(s)))
            hashes.add(canonical_hash(mol))
            from molkit.canonical import canonical_smiles
            texts.add(canonical_smiles(mol))
        good = len(hashes) == 1 and len(texts) == 1
        ok &= good
        print(f"  {'OK ' if good else 'FAIL'} {group}")
    return ok


def brute_force_substructure(qmol, db: MoleculeDatabase):
    """不使用指纹，逐分子精确匹配的参考实现。"""
    hits = set()
    for rec in db.records:
        if substructure_matches(qmol, rec.canon_mol, max_matches=1):
            hits.add(rec.mol_id)
    return hits


def check_substructure(db: MoleculeDatabase) -> bool:
    print("\n[2] 子结构查询：指纹预筛结果 vs 逐分子暴力匹配")
    ok = True
    for q, desc in SUBSTRUCTURE_QUERIES:
        qmol = prepare_query(parse_smiles(q))
        hits, stats = db.substructure_search(q, max_matches_per_mol=1)
        indexed = {h["mol_id"] for h in hits}
        brute = brute_force_substructure(qmol, db)
        good = indexed == brute
        ok &= good
        print(f"  {'OK ' if good else 'FAIL'} {q:12s} {desc:28s} "
              f"命中 {len(indexed):4d} 候选 {stats.candidates:5d} "
              f"过滤率 {stats.filtered_ratio:6.1%}")
        if not good:
            print(f"      仅索引命中: {sorted(indexed - brute)}")
            print(f"      漏检: {sorted(brute - indexed)}")
    return ok


def check_similarity(db: MoleculeDatabase, threshold: float = 0.5) -> bool:
    print(f"\n[3] 相似性查询（Tanimoto 阈值 {threshold}）与剪枝比例")
    probe_smiles = ["CCO", "c1ccccc1", "CC(=O)O", "c1ccncc1"]
    ok = True
    for probe in probe_smiles:
        qmol = canonicalize(standardize(parse_smiles(probe)))
        qbits = fingerprint_bits(qmol)
        # 参考：全量直接算 Tanimoto
        brute_hits = {}
        for rec in db.records:
            coef = tanimoto(qbits, rec.fp_bits)
            if coef >= threshold - 1e-12:
                brute_hits[rec.mol_id] = round(coef, 6)
        # 索引查询（带上界剪枝）
        hits, stats = db.similarity_search(probe, threshold)
        indexed = {h["mol_id"]: h["tanimoto"] for h in hits}
        good = indexed == brute_hits
        ok &= good
        print(f"  {'OK ' if good else 'FAIL'} 探针 {probe:10s} "
              f"命中 {len(indexed):4d} 位数上界剪枝 {stats.filtered:5d} "
              f"({stats.filtered_ratio:6.1%})")
    return ok


def check_scaling(smiles_pool, sizes) -> dict:
    print("\n[4] 查询耗时与指纹筛选率随库大小变化")
    print(f"  {'库大小':>8} | {'入库(s)':>9} | {'子结构查询(s)':>12} | "
          f"{'过滤率':>7} | {'相似查询(s)':>11}")
    rows = []
    db = MoleculeDatabase()
    inserted = 0
    for size in sizes:
        while len(db) < size and inserted < len(smiles_pool):
            db.add_smiles(smiles_pool[inserted])
            inserted += 1
        if len(db) < size:
            break
        t0 = time.perf_counter()
        _, sub_stats = db.substructure_search("c1ccccc1")
        t_sub = time.perf_counter() - t0
        t1 = time.perf_counter()
        _, sim_stats = db.similarity_search("c1ccccc1", 0.5)
        t_sim = time.perf_counter() - t1
        rows.append((size, t_sub, sub_stats.filtered_ratio, t_sim))
        print(f"  {size:8d} | {'':>9} | {t_sub:12.4f} | "
              f"{sub_stats.filtered_ratio:7.1%} | {t_sim:11.4f}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000,
                    help="随机分子库目标规模")
    ap.add_argument("--seed", type=int, default=20261123)
    ap.add_argument("--scaling", default="1000,5000,10000,20000")
    args = ap.parse_args()

    t_start = time.perf_counter()
    print(f"生成 {args.n} 个随机合理分子（seed={args.seed}）…")
    rng = random.Random(args.seed)
    pool, dup, invalid = generate_distinct(rng, args.n)
    print(f"  得到 {len(pool)} 个唯一分子，重复 {dup}，非法 {invalid}")

    results = {}
    results["equivalence"] = check_equivalence()

    print("\n[入库] 构建数据库并验证去重…")
    db = MoleculeDatabase()
    new_count = 0
    t0 = time.perf_counter()
    for s in pool:
        _, is_new = db.add_smiles(s)
        new_count += int(is_new)
    build_seconds = time.perf_counter() - t0
    # 再全部入一遍，应零新增
    second_new = sum(int(db.add_smiles(s)[1]) for s in pool)
    dedup_ok = second_new == 0 and new_count == len(db)
    print(f"  库大小 {len(db)}，首次入库新增 {new_count}，"
          f"二次入库新增 {second_new}，耗时 {build_seconds:.2f}s "
              f"({len(db)/max(build_seconds,1e-9):.0f} 分子/秒)")
    results["dedup"] = dedup_ok

    results["substructure"] = check_substructure(db)
    results["similarity"] = check_similarity(db)

    sizes = [int(x) for x in args.scaling.split(",") if int(x) <= len(pool)]
    check_scaling(pool, sizes)

    print("\n[5] 骨架归类 top10")
    for key, count in db.scaffold_counts(top=10):
        print(f"  {count:5d}x  {key or '(无环)'}")

    print("\n================ 验收汇总 ================")
    for name, ok in results.items():
        print(f"  {name:14s}: {'通过' if ok else '失败'}")
    all_pass = all(results.values())
    print(f"  总耗时 {time.perf_counter()-t_start:.1f}s")
    print("  结论：", "全部通过 ✅" if all_pass else "存在失败 ❌")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
