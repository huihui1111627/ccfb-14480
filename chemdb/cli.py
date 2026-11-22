"""命令行入口：解析、规范化、子结构/相似性查询、压测。

示例：
  python3 -I chemdb/cli.py parse "c1ccccc1"
  python3 -I chemdb/cli.py canonical "C1=CC=CC=C1"
  python3 -I chemdb/cli.py bench --n 20000
"""
from __future__ import annotations

import argparse
import os
import sys

if __package__ in (None, ""):  # 支持直接 python3 chemdb/cli.py 运行
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from chemdb.parser import parse_smiles, parse_query, ParseError
    from chemdb.canonical import canonical_smiles
    from chemdb.fingerprint import fingerprint
    from chemdb.database import ChemDB
else:
    from .parser import parse_smiles, parse_query, ParseError
    from .canonical import canonical_smiles
    from .fingerprint import fingerprint
    from .database import ChemDB


def cmd_parse(args) -> int:
    mol = parse_query(args.smiles) if args.query else parse_smiles(args.smiles)
    print("规范形式:", canonical_smiles(mol))
    print("分子式:  ", mol.formula())
    print("原子数 %d 键数 %d" % (mol.n_atoms(), mol.n_bonds()))
    for a in mol.atoms:
        print("  [%d] %-3s 芳香=%s 电荷=%+d 同位素=%s 隐式H=%d 显式H=%d"
              % (a.index, a.symbol, a.aromatic, a.charge,
                 a.isotope if a.isotope is not None else "-",
                 mol._implicit_h[a.index] if mol._implicit_h else 0,
                 a.explicit_h))
    for b in mol.bonds:
        print("  键 %d-%d 级=%d 环内=%s%s%s"
              % (b.a1, b.a2, b.order, b.in_ring,
                 " 环键查询" if b.is_ring_query else "",
                 " 链键查询" if b.is_chain_query else ""))
    return 0


def cmd_canonical(args) -> int:
    print(canonical_smiles(parse_smiles(args.smiles)))
    return 0


def cmd_search(args) -> None:
    db = ChemDB()
    with open(args.file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                db.add_smiles(line)
    print("库大小: %d" % len(db))
    if args.mode == "sub":
        hits = db.substructure_search(args.query)
        print("片段 %r 命中 %d 个:" % (args.query, len(hits)))
        for h in hits[: args.limit]:
            print("  [%d] %s (%s)" % (h["id"], h["canonical"], h["formula"]))
    else:
        res = db.similarity_search(args.query, args.threshold)
        print("相似性阈值 %.2f，命中 %d 个:" % (args.threshold, len(res)))
        for r in res[: args.limit]:
            print("  [%d] %.3f  %s" % (r["id"], r["score"], r["canonical"]))


def cmd_bench(args) -> None:
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from bench.benchmark import main as bench_main
    sys.argv = ["benchmark", "--n", str(args.n)]
    bench_main()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="化合物结构数据库命令行")
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("parse", help="解析结构式并打印分子图")
    pp.add_argument("smiles")
    pp.add_argument("--query", action="store_true", help="按查询片段解析")
    pp.set_defaults(func=cmd_parse)

    pc = sub.add_parser("canonical", help="输出规范 SMILES")
    pc.add_argument("smiles")
    pc.set_defaults(func=cmd_canonical)

    ps = sub.add_parser("search", help="在 SMILES 列表文件上查询")
    ps.add_argument("file")
    ps.add_argument("query")
    ps.add_argument("--mode", choices=["sub", "sim"], default="sub")
    ps.add_argument("--threshold", type=float, default=0.5)
    ps.add_argument("--limit", type=int, default=20)
    ps.set_defaults(func=cmd_search)

    pb = sub.add_parser("bench", help="随机分子压测")
    pb.add_argument("--n", type=int, default=10000)
    pb.set_defaults(func=cmd_bench)

    args = p.parse_args(argv)
    try:
        return args.func(args)
    except ParseError as e:
        print("解析错误: %s" % e, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
