"""大规模压测：随机分子入库、规范一致性、指纹筛选比例、查询耗时随库大小变化。

用法：
  python3 -I bench/benchmark.py --n 100000
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chemdb import ChemDB, parse_smiles, canonical_smiles, fingerprint
from chemdb.parser import parse_query
from chemdb.fingerprint import tanimoto
from chemdb.generator import generate_dataset, random_writings
from chemdb.matcher import has_substructure
import random


def fmt_time(sec: float) -> str:
    if sec < 1e-3:
        return "%.2f ms" % (sec * 1000)
    if sec < 1:
        return "%.1f ms" % (sec * 1000)
    return "%.2f s" % sec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--brute-sample", type=int, default=400,
                    help="暴力校验抽样的库大小上限")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    print("生成 %d 个随机合理分子（seed=%d）..." % (args.n, args.seed))
    t0 = time.perf_counter()
    dataset = generate_dataset(args.n, seed=args.seed)
    print("  实际生成 %d 个，耗时 %s（%.0f 分子/秒）"
          % (len(dataset), fmt_time(time.perf_counter() - t0),
             len(dataset) / max(1e-9, time.perf_counter() - t0)))

    db = ChemDB(n_bits=1024, max_path=7)

    # ---- [1] 入库 + 规范一致性 ----
    print("\n[1] 入库与规范形式一致性")
    t0 = time.perf_counter()
    n_dup = consistency_fail = 0
    for smi in dataset:
        rid, is_new = db.add_smiles(smi)
        if not is_new:
            n_dup += 1
        rec = db.records[rid]
        if canonical_smiles(parse_smiles(rec.canonical)) != rec.canonical:
            consistency_fail += 1
    insert_time = time.perf_counter() - t0
    print("  库大小 %d（重复丢弃 %d），规范不一致 %d"
          % (len(db), n_dup, consistency_fail))
    print("  入库耗时 %s，平均 %.2f ms/分子（%.0f 分子/秒）"
          % (fmt_time(insert_time), insert_time * 1000 / len(dataset),
             len(dataset) / insert_time))

    # ---- [2] 多写法归一 ----
    print("\n[2] 同一分子多种写法归一（凯库勒式/随机遍历顺序）")
    sample = rng.sample(dataset, min(100, len(dataset)))
    writing_fail = checked = 0
    for canon in sample:
        base = canonical_smiles(parse_smiles(canon))
        for w in random_writings(canon, rng):
            checked += 1
            if canonical_smiles(parse_smiles(w)) != base:
                writing_fail += 1
    print("  检查 %d 个等价写法，归一失败 %d" % (checked, writing_fail))

    # ---- [3] 子结构查询随库大小变化（暴力校验只在小库做全量）----
    print("\n[3] 子结构查询：指纹筛选比例与耗时随库大小变化")
    query_list = ["c1ccccc1", "c1ccncc1", "C(=O)O", "CCO", "c1ccoc1",
                  "Cl", "C1CCCCC1", "c~rc"]
    sizes = sorted({s for s in [1000, 5000, 10000, 25000, 50000, len(db)]
                    if s <= len(db)})
    print("  %-8s %-10s %-10s %-10s %-10s %s" %
          ("库大小", "指纹候选", "淘汰率", "查询耗时", "命中数", "暴力一致"))
    for size in sizes:
        sub = _sub_db(db, size)
        q = query_list[size % len(query_list)]
        qmol = parse_query(q)
        t1 = time.perf_counter()
        cands = sub.fingerprint_candidates(fingerprint(qmol))
        fp_time = time.perf_counter() - t1
        t2 = time.perf_counter()
        hits = sub.substructure_search(q)
        total_time = time.perf_counter() - t2
        reject = 1 - len(cands) / max(1, size)
        consistent = "-"
        if size <= args.brute_sample:
            brute = sum(1 for r in sub.records
                        if has_substructure(qmol, sub.get_mol(r.id)))
            consistent = "是" if len(hits) == brute else "否!!(%d vs %d)" % (
                len(hits), brute)
        print("  %-8d %-10d %-9.1f%% %-10s %-10d %s"
              % (size, len(cands), reject * 100,
                 fmt_time(total_time), len(hits), consistent))

    # ---- [4] 相似性查询与剪枝 ----
    print("\n[4] 相似性查询（Tanimoto）与上下界剪枝")
    qsmi = rng.choice(dataset)
    for threshold in [0.3, 0.5, 0.7, 0.9]:
        t1 = time.perf_counter()
        res_pruned = db.similarity_search(qsmi, threshold, use_bound=True)
        t_pruned = time.perf_counter() - t1
        t2 = time.perf_counter()
        res_full = db.similarity_search(qsmi, threshold, use_bound=False)
        t_full = time.perf_counter() - t2
        same = {r["id"] for r in res_pruned} == \
            {r["id"] for r in res_full}
        print("  阈值 %.1f: 命中 %5d, 剪枝 %s, 全量 %s, 结果一致=%s"
              % (threshold, len(res_pruned), fmt_time(t_pruned),
                 fmt_time(t_full), same))

    # ---- [5] 骨架统计 ----
    print("\n[5] 出现最多的骨架 Top 5")
    for scaf, cnt in db.scaffold_frequency(top=5):
        print("  %6d  %s" % (cnt, scaf[:48]))


def _sub_db(db: ChemDB, size: int) -> ChemDB:
    sub = ChemDB(n_bits=db.n_bits, max_path=db.max_path)
    for rec in db.records[:size]:
        sub.records.append(rec)
        sub._by_canonical[rec.canonical] = rec.id
        x = rec.fp
        while x:
            low = x & -x
            sub._inverted.setdefault(low.bit_length() - 1, set()).add(rec.id)
            x ^= low
    return sub


if __name__ == "__main__":
    main()
