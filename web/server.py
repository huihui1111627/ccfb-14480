"""化合物结构数据库 Web 服务（仅标准库，无第三方依赖）。

提供：
  GET /                 单页应用（解析可视化 + 片段查询 + 高亮）
  POST /api/parse       解析结构式，返回分子图（原子、键、环、隐式氢）
  POST /api/substructure 子结构查询，返回命中分子与高亮原子
  POST /api/similarity  相似性查询（Tanimoto）
  POST /api/add         入库
  GET  /api/scaffolds   骨架频次
  GET  /api/stats       库统计
"""
from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chemdb import ChemDB, parse_smiles, canonical_smiles, fingerprint
from chemdb.parser import parse_query, ParseError
from chemdb.aromaticity import AromaticityError
from chemdb.matcher import substructure_matches

WEB_DIR = os.path.dirname(os.path.abspath(__file__))


def mol_to_dict(mol) -> dict:
    return {
        "atoms": [
            {"index": a.index, "symbol": a.symbol, "element": a.element,
             "aromatic": a.aromatic, "charge": a.charge,
             "isotope": a.isotope, "explicit_h": a.explicit_h,
             "implicit_h": (mol._implicit_h[a.index]
                            if mol._implicit_h else 0),
             "total_h": mol.total_h(a.index),
             "wildcard": a.wildcard, "aromatic_query": a.aromatic_query,
             "aliphatic_query": a.aliphatic_query}
            for a in mol.atoms],
        "bonds": [
            {"a1": b.a1, "a2": b.a2, "order": b.order,
             "in_ring": b.in_ring, "is_ring_query": b.is_ring_query,
             "is_chain_query": b.is_chain_query}
            for b in mol.bonds],
        "formula": mol.formula(),
        "n_atoms": mol.n_atoms(),
        "n_bonds": mol.n_bonds(),
    }


class Handler(BaseHTTPRequestHandler):
    db: ChemDB = None  # 类属性，由 main 注入

    def log_message(self, *args):  # 静默默认访问日志
        pass

    def _send(self, code: int, payload) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw.decode("utf-8"))

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            with open(os.path.join(WEB_DIR, "index.html"), "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        elif self.path == "/api/stats":
            db = Handler.db
            self._send(200, {"size": len(db), "n_bits": db.n_bits})
        elif self.path == "/api/scaffolds":
            db = Handler.db
            self._send(200, [{"scaffold": s, "count": c}
                             for s, c in db.scaffold_frequency(top=10)])
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self) -> None:
        db = Handler.db
        try:
            data = self._read_json()
            if self.path == "/api/parse":
                mol = parse_query(data["smiles"]) \
                    if data.get("query") else parse_smiles(data["smiles"])
                can = canonical_smiles(mol)
                self._send(200, {"mol": mol_to_dict(mol),
                                 "canonical": can})
            elif self.path == "/api/add":
                rid, is_new = db.add_smiles(data["smiles"])
                self._send(200, {"id": rid, "new": is_new,
                                 "canonical": db.records[rid].canonical})
            elif self.path == "/api/substructure":
                qmol = parse_query(data["smiles"])
                qfp = fingerprint(qmol, db.n_bits, db.max_path)
                candidates = db.fingerprint_candidates(qfp)
                limit = int(data.get("limit", 100))
                hits = []
                for rid in candidates:
                    tmol = db.get_mol(rid)
                    matches = substructure_matches(qmol, tmol, max_matches=1)
                    if matches:
                        hits.append({
                            "id": rid,
                            "canonical": db.records[rid].canonical,
                            "formula": db.records[rid].formula,
                            "matched_atoms": sorted(matches[0].values()),
                            "mol": mol_to_dict(tmol),
                        })
                        if len(hits) >= limit:
                            break
                self._send(200, {"n_candidates": len(candidates),
                                 "n_hits": len(hits), "hits": hits})
            elif self.path == "/api/similarity":
                threshold = float(data.get("threshold", 0.5))
                results = db.similarity_search(
                    data["smiles"], threshold, use_bound=True)
                self._send(200, {"results": results[: int(
                    data.get("limit", 100))]})
            else:
                self._send(404, {"error": "not found"})
        except ParseError as e:
            self._send(400, {"error": str(e), "position": e.position})
        except AromaticityError as e:
            self._send(400, {"error": str(e)})
        except (KeyError, ValueError) as e:
            self._send(400, {"error": "请求参数错误: %s" % e})


def serve(db: ChemDB, host: str = "127.0.0.1", port: int = 8000) -> None:
    Handler.db = db
    httpd = ThreadingHTTPServer((host, port), Handler)
    print("化合物数据库 Web 服务已启动: http://%s:%d" % (host, port))
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.shutdown()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--load", type=int, default=0,
                    help="启动时随机生成并加载的分子数")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    db = ChemDB()
    if args.load:
        from chemdb.generator import generate_dataset
        for smi in generate_dataset(args.load, seed=args.seed):
            db.add_smiles(smi)
        print("已加载 %d 个随机分子" % len(db))
    serve(db, port=args.port)
