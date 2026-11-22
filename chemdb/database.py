"""化合物结构数据库：去重入库、指纹倒排索引、子结构/相似性查询、骨架统计。

存储：内存中保留记录（规范 SMILES、指纹、分子式等轻量字段），
可选 SQLite 持久化。分子图按需重解析，避免几十万个图对象常驻内存。
"""
from __future__ import annotations

import sqlite3
from collections import Counter, OrderedDict
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from .parser import parse_smiles, parse_query
from .canonical import canonical_smiles
from .fingerprint import fingerprint, tanimoto, bit_positions, popcount
from .matcher import has_substructure, substructure_matches
from .scaffold import scaffold_key, both_scaffold_keys
from .molecule import Mol


@dataclass
class Record:
    id: int
    smiles: str          # 原始输入
    canonical: str       # 规范 SMILES（去重 key）
    fp: int              # 指纹位向量
    formula: str
    scaffold: str
    generic_scaffold: str
    n_bits: int


class ChemDB:
    def __init__(self, n_bits: int = 1024, max_path: int = 7,
                 mol_cache_size: int = 4096) -> None:
        self.n_bits = n_bits
        self.max_path = max_path
        self.records: List[Record] = []
        self._by_canonical: Dict[str, int] = {}
        # 指纹倒排：bit 位 -> 记录 id 集合
        self._inverted: Dict[int, set] = {}
        self._conn: Optional[sqlite3.Connection] = None
        # 有界分子图缓存，避免查询时对同一分子反复解析
        self._mol_cache: "OrderedDict[int, Mol]" = OrderedDict()
        self._mol_cache_size = mol_cache_size
        self._last_sim_skipped = 0

    # ------------------------------------------------------------------
    # 入库
    # ------------------------------------------------------------------
    def add_smiles(self, text: str) -> Tuple[Optional[int], bool]:
        """返回 (记录id, 是否新入库)。重复分子按规范形式判重。"""
        mol = parse_smiles(text)
        return self.add_mol(mol, text)

    def add_mol(self, mol: Mol, raw: str = "") -> Tuple[Optional[int], bool]:
        can = canonical_smiles(mol)
        if can in self._by_canonical:
            return self._by_canonical[can], False
        fp = fingerprint(mol, self.n_bits, self.max_path)
        scaf_key, gscaf_key = both_scaffold_keys(mol)
        rid = len(self.records)
        rec = Record(
            id=rid,
            smiles=raw or can,
            canonical=can,
            fp=fp,
            formula=mol.formula(),
            scaffold=scaf_key,
            generic_scaffold=gscaf_key,
            n_bits=popcount(fp),
        )
        self.records.append(rec)
        self._by_canonical[can] = rid
        x = fp
        while x:
            low = x & -x
            self._inverted.setdefault(low.bit_length() - 1, set()).add(rid)
            x ^= low
        if self._conn is not None:
            self._conn.execute(
                "INSERT OR IGNORE INTO molecules "
                "(smiles, canonical, fp, formula, scaffold, generic) "
                "VALUES (?,?,?,?,?,?)",
                (rec.smiles, rec.canonical, str(rec.fp), rec.formula,
                 rec.scaffold, rec.generic_scaffold))
        return rid, True

    def __len__(self) -> int:
        return len(self.records)

    def contains_canonical(self, can: str) -> bool:
        return can in self._by_canonical

    def get_mol(self, rid: int) -> Mol:
        cached = self._mol_cache.get(rid)
        if cached is not None:
            self._mol_cache.move_to_end(rid)
            return cached
        mol = parse_smiles(self.records[rid].canonical)
        self._mol_cache[rid] = mol
        if len(self._mol_cache) > self._mol_cache_size:
            self._mol_cache.popitem(last=False)  # LRU 淘汰最久未用
        return mol

    # ------------------------------------------------------------------
    # 指纹预筛
    # ------------------------------------------------------------------
    def fingerprint_candidates(self, qfp: int) -> List[int]:
        """位子集筛选：命中所有查询位的记录（用最稀有的位驱动求交）。"""
        bits = []
        x = qfp
        while x:
            low = x & -x
            bits.append(low.bit_length() - 1)
            x ^= low
        if not bits:
            return list(range(len(self.records)))
        bits.sort(key=lambda b: len(self._inverted.get(b, ())))
        result: Optional[set] = None
        for b in bits:
            ids = self._inverted.get(b)
            if not ids:
                return []
            if result is None:
                result = set(ids)
            else:
                result &= ids
            if not result:
                return []
        return sorted(result or [])

    # ------------------------------------------------------------------
    # 子结构查询
    # ------------------------------------------------------------------
    def substructure_search(self, query_smiles: str,
                            collect_matches: bool = False
                            ) -> List[dict]:
        qmol = parse_query(query_smiles)
        qfp = fingerprint(qmol, self.n_bits, self.max_path)
        candidates = self.fingerprint_candidates(qfp)
        hits: List[dict] = []
        for rid in candidates:
            tmol = self.get_mol(rid)
            matches = substructure_matches(
                qmol, tmol, max_matches=None if collect_matches else 1)
            if matches:
                item = {"id": rid,
                        "canonical": self.records[rid].canonical,
                        "formula": self.records[rid].formula}
                if collect_matches:
                    item["matched_atoms"] = [
                        sorted(m.values()) for m in matches]
                hits.append(item)
        return hits

    # ------------------------------------------------------------------
    # 相似性查询（Tanimoto + 上下界剪枝）
    # ------------------------------------------------------------------
    def similarity_search(self, query_smiles: str, threshold: float,
                          use_bound: bool = True) -> List[dict]:
        qmol = parse_query(query_smiles)
        qfp = fingerprint(qmol, self.n_bits, self.max_path)
        qcount = popcount(qfp)
        results: List[dict] = []
        skipped = 0
        for rec in self.records:
            b = rec.n_bits
            # 上界：交集 <= min(A,B) → T <= min(A,B)/max(A,B)
            upper = min(qcount, b) / max(qcount, b)
            if use_bound and upper < threshold:
                skipped += 1
                continue
            score = tanimoto(qfp, rec.fp)
            if score < threshold:
                continue
            # 下界：交集 >= A+B-N → T >= max(0,A+B-N)/max(A,B)
            lower = max(0, qcount + b - self.n_bits) / max(qcount, b)
            results.append({"id": rec.id,
                            "canonical": rec.canonical,
                            "formula": rec.formula,
                            "score": score,
                            "bound_guaranteed": use_bound
                            and lower >= threshold})
        results.sort(key=lambda r: (-r["score"], r["id"]))
        self._last_sim_skipped = skipped
        return results

    # ------------------------------------------------------------------
    # 骨架统计
    # ------------------------------------------------------------------
    def scaffold_frequency(self, generic: bool = False,
                           top: Optional[int] = None) -> List[Tuple[str, int]]:
        field = "generic_scaffold" if generic else "scaffold"
        counter = Counter(getattr(r, field) for r in self.records
                          if getattr(r, field))
        return counter.most_common(top)

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------
    def attach_sqlite(self, path: str) -> None:
        self._conn = sqlite3.connect(path)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS molecules ("
            "id INTEGER PRIMARY KEY, smiles TEXT, canonical TEXT UNIQUE, "
            "fp TEXT, formula TEXT, scaffold TEXT, generic TEXT)")
        self._conn.commit()

    def commit(self) -> None:
        if self._conn is not None:
            self._conn.commit()

    def load_sqlite(self, path: str) -> int:
        conn = sqlite3.connect(path)
        rows = conn.execute(
            "SELECT smiles, canonical, fp, formula, scaffold, generic "
            "FROM molecules").fetchall()
        conn.close()
        for smiles, can, fp, formula, scaf, gscaf in rows:
            if can in self._by_canonical:
                continue
            fp = int(fp)
            rid = len(self.records)
            self.records.append(Record(rid, smiles, can, fp, formula,
                                       scaf, gscaf, popcount(fp)))
            self._by_canonical[can] = rid
            x = fp
            while x:
                low = x & -x
                self._inverted.setdefault(low.bit_length() - 1, set()).add(rid)
                x ^= low
        return len(rows)
