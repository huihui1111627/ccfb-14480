# -*- coding: utf-8 -*-
"""化合物结构数据库的 Web 服务（仅用 Python 标准库）。

接口：
  GET  /                 静态页面
  POST /api/parse        解析结构式，返回分子图（原子/键/芳香/隐氢等）
  POST /api/substructure 子结构查询，返回命中与高亮原子映射
  POST /api/similarity   相似性查询
  GET  /api/scaffolds    骨架频次
  POST /api/add          入库
"""
from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from molkit.aromatic import perceive_aromaticity, standardize  # noqa: E402
from molkit.canonical import canonicalize, canonical_smiles    # noqa: E402
from molkit.database import MoleculeDatabase                   # noqa: E402
from molkit.matcher import prepare_query, substructure_matches  # noqa: E402
from molkit.parser import ParseError, parse_smiles             # noqa: E402

WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "web_static")


def graph_payload(mol, highlight=None) -> dict:
    """把分子图序列化成前端可渲染的结构。"""
    arom_atoms, arom_bonds = perceive_aromaticity(mol)
    atoms = []
    for i, a in enumerate(mol.atoms):
        atoms.append({
            "idx": i,
            "symbol": a.symbol,
            "charge": a.charge,
            "isotope": a.isotope,
            "hydrogen": a.total_h,
            "aromatic": i in arom_atoms or a.aromatic,
            "wildcard": a.is_wildcard,
            "highlight": i in (highlight or set()),
            "degree": mol.degree(i),
        })
    bonds = []
    for bi, b in enumerate(mol.bonds):
        bonds.append({
            "idx": bi,
            "a1": b.a1,
            "a2": b.a2,
            "order": b.order,
            "aromatic": b.aromatic or bi in arom_bonds,
            "wildcard": b.is_wildcard,
            "in_ring": bi in mol.ring_bonds(),
        })
    return {"atoms": atoms, "bonds": bonds,
            "fragments": mol.compute_fragments()}


def build_db() -> MoleculeDatabase:
    db = MoleculeDatabase()
    # 内置演示数据
    demo = [
        "CCO", "CCCO", "c1ccccc1", "Cc1ccccc1", "c1ccc(O)cc1",
        "Clc1ccccc1", "c1ccncc1", "c1ccoc1", "c1ccsc1", "CC(=O)O",
        "CC(=O)[O-]", "CC#N", "c1ccc2ccccc2c1", "C1CCCCC1",
        "c1ccc(N)cc1", "O=P(O)(O)O", "CC(=O)NC", "c1cc[nH]c1",
        "c1ccc(Cl)cc1", "CCCC", "CC(C)C", "c1ccccc1C(=O)O",
    ]
    for s in demo:
        db.add_smiles(s)
    return db


DB = build_db()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # 静默常规访问日志
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
        return json.loads(raw.decode("utf-8") or "{}")

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            return self._serve_file("index.html", "text/html; charset=utf-8")
        if path in ("/app.js", "/web_static/app.js"):
            return self._serve_file("app.js",
                                    "application/javascript; charset=utf-8")
        if path == "/api/scaffolds":
            data = [{"scaffold": k or "(无环)", "count": c}
                    for k, c in DB.scaffold_counts()]
            return self._send(200, {"scaffolds": data, "size": len(DB)})
        if path.startswith("/web_static/") or path.endswith(".js"):
            return self._serve_static(path)
        self._send(404, {"error": "not found"})

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            data = self._read_json()
            if path == "/api/parse":
                return self._handle_parse(data)
            if path == "/api/substructure":
                return self._handle_substructure(data)
            if path == "/api/similarity":
                return self._handle_similarity(data)
            if path == "/api/add":
                return self._handle_add(data)
            self._send(404, {"error": "not found"})
        except ParseError as e:
            self._send(400, {"error": str(e), "column": e.column})
        except Exception as e:  # 界面兜底，不让服务崩
            self._send(500, {"error": f"{type(e).__name__}: {e}"})

    def _handle_parse(self, data):
        smiles = data.get("smiles", "").strip()
        raw = parse_smiles(smiles)
        std = standardize(parse_smiles(smiles))
        canon = canonicalize(std)
        return self._send(200, {
            "input_graph": graph_payload(raw),
            "canonical_smiles": canonical_smiles(canon),
            "canonical_graph": graph_payload(canon),
            "n_atoms": len(canon.atoms),
            "n_bonds": len(canon.bonds),
        })

    def _handle_substructure(self, data):
        smiles = data.get("smiles", "").strip()
        qmol = prepare_query(parse_smiles(smiles))
        hits, stats = DB.substructure_search(smiles, max_matches_per_mol=1)
        out = []
        for h in hits:
            mapping = h["matches"][0] if h["matches"] else {}
            out.append({
                "mol_id": h["mol_id"],
                "canonical": h["canonical"],
                "scaffold": h["scaffold"] or "(无环)",
                "highlight_target": sorted(set(mapping.values())),
                "mapping": {str(k): v for k, v in mapping.items()},
            })
        return self._send(200, {
            "hits": out,
            "query_graph": graph_payload(qmol),
            "stats": {
                "total": stats.total, "candidates": stats.candidates,
                "matched": stats.exact_matched,
                "filtered": stats.filtered,
                "filtered_ratio": round(stats.filtered_ratio, 4),
                "prefilter_ms": round(stats.prefilter_seconds * 1000, 2),
                "exact_ms": round(stats.exact_seconds * 1000, 2),
            },
        })

    def _handle_similarity(self, data):
        smiles = data.get("smiles", "").strip()
        threshold = float(data.get("threshold", 0.6))
        hits, stats = DB.similarity_search(smiles, threshold)
        return self._send(200, {
            "hits": hits,
            "stats": {
                "total": stats.total, "pruned": stats.filtered,
                "pruned_ratio": round(stats.filtered_ratio, 4),
                "ms": round((stats.prefilter_seconds
                             + stats.exact_seconds) * 1000, 2),
            },
        })

    def _handle_add(self, data):
        smiles = data.get("smiles", "").strip()
        mid, is_new = DB.add_smiles(smiles)
        return self._send(200, {"mol_id": mid, "is_new": is_new,
                                "size": len(DB)})

    def _serve_file(self, name: str, ctype: str):
        fpath = os.path.join(WEB_DIR, name)
        with open(fpath, "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_static(self, path: str):
        name = os.path.basename(path)
        ctype = "application/javascript; charset=utf-8" if name.endswith(
            ".js") else "text/plain; charset=utf-8"
        fpath = os.path.join(WEB_DIR, name)
        if not os.path.exists(fpath):
            return self._send(404, {"error": "not found"})
        self._serve_file(name, ctype)


def main():
    port = int(os.environ.get("MOLKIT_PORT", "8000"))
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"molkit Web 服务已启动: http://127.0.0.1:{port}")
    print(f"内置演示分子 {len(DB)} 个")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
