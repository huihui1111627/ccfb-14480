# -*- coding: utf-8 -*-
"""化合物结构数据库。

职责：
- 入库：解析 -> 标准化 -> 规范化，按规范哈希去重（同物同码）。
- 存储：每条记录保存规范图、指纹位集合、原始写法、骨架键。
- 索引：指纹位 -> 分子 id 的倒排表，支持快速子集预筛。
- 子结构查询：指纹子集预筛（保证不漏）+ VF2 精确匹配。
- 相似性查询：用指纹位数的 Tanimoto 上界先剪枝，再算真实系数。
- 骨架归类：Murcko 骨架频次统计。

纯 Python 实现，持久化用 JSON（指纹转排序列表），无第三方依赖。
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .aromatic import standardize
from .canonical import canonicalize, canonical_hash, canonical_smiles
from .fingerprint import (FP_SIZE, could_meet_threshold, fingerprint_bits,
                          is_subset_fp, tanimoto)
from .graph import Molecule
from .matcher import prepare_query, substructure_matches
from .parser import ParseError, parse_smiles
from .scaffold import scaffold_key


@dataclass
class Record:
    mol_id: int
    canon_mol: Molecule
    fp_bits: Set[int]
    canonical: str
    scaffold: str
    sources: List[str] = field(default_factory=list)
    popcount: int = 0


@dataclass
class SearchStats:
    candidates: int = 0          # 指纹预筛后进入精确匹配的数量
    total: int = 0               # 库内分子总数
    filtered: int = 0            # 被指纹过滤掉的数量
    exact_matched: int = 0
    prefilter_seconds: float = 0.0
    exact_seconds: float = 0.0

    @property
    def filtered_ratio(self) -> float:
        return self.filtered / self.total if self.total else 0.0


class MoleculeDatabase:
    def __init__(self, fp_size: int = FP_SIZE):
        self.fp_size = fp_size
        self.records: List[Record] = []
        self._hash_to_id: Dict[str, int] = {}
        # 指纹位 -> 包含该位的分子 id 集合
        self._inverted: Dict[int, Set[int]] = {}
        self._scaffold_cache: Dict[int, str] = {}

    # ------------------------------------------------------------------
    # 入库
    # ------------------------------------------------------------------
    def add_smiles(self, text: str, source: Optional[str] = None
                   ) -> Tuple[Optional[int], bool]:
        """返回 (mol_id, is_new)。重复分子返回已有 id 和 False。"""
        mol = parse_smiles(text)
        return self.add_molecule(mol, source or text)

    def add_molecule(self, mol: Molecule, source: str = ""
                     ) -> Tuple[Optional[int], bool]:
        std = standardize(mol)
        canon = canonicalize(std)
        chash = canonical_hash(canon)
        if chash in self._hash_to_id:
            mid = self._hash_to_id[chash]
            if source and source not in self.records[mid].sources:
                self.records[mid].sources.append(source)
            return mid, False

        bits = fingerprint_bits(canon, self.fp_size)
        scaf = scaffold_key(canon)
        rec = Record(
            mol_id=len(self.records),
            canon_mol=canon,
            fp_bits=bits,
            canonical=canonical_smiles(canon),
            scaffold=scaf,
            sources=[source] if source else [],
            popcount=len(bits),
        )
        mid = rec.mol_id
        self.records.append(rec)
        self._hash_to_id[chash] = mid
        for bit in bits:
            self._inverted.setdefault(bit, set()).add(mid)
        self._scaffold_cache[mid] = scaf
        return mid, True

    def __len__(self) -> int:
        return len(self.records)

    def contains_hash(self, chash: str) -> bool:
        return chash in self._hash_to_id

    # ------------------------------------------------------------------
    # 子结构查询
    # ------------------------------------------------------------------
    def substructure_search(self, smiles: str,
                            max_matches_per_mol: int = 1
                            ) -> Tuple[List[Dict], SearchStats]:
        qmol = prepare_query(parse_smiles(smiles))
        qbits = fingerprint_bits(qmol, self.fp_size)
        stats = SearchStats(total=len(self.records))

        t0 = time.perf_counter()
        candidate_ids = self._prefilter(qbits)
        stats.candidates = len(candidate_ids)
        stats.filtered = self.records and (len(self.records) - len(candidate_ids))
        stats.prefilter_seconds = time.perf_counter() - t0

        t1 = time.perf_counter()
        hits: List[Dict] = []
        for mid in candidate_ids:
            rec = self.records[mid]
            matches = substructure_matches(
                qmol, rec.canon_mol, max_matches=max_matches_per_mol)
            if matches:
                hits.append({
                    "mol_id": mid,
                    "canonical": rec.canonical,
                    "sources": list(rec.sources),
                    "scaffold": rec.scaffold,
                    "matches": matches,
                })
        stats.exact_matched = len(hits)
        stats.exact_seconds = time.perf_counter() - t1
        return hits, stats

    def _prefilter(self, qbits: Set[int]) -> List[int]:
        """倒排表求交集：候选 = 同时包含查询所有指纹位的分子。

        空查询位（如孤立通配）退化为全库扫描。
        """
        if not qbits:
            return list(range(len(self.records)))
        sets = [self._inverted.get(bit, set()) for bit in qbits]
        if any(not s for s in sets):
            return []
        # 从最小集合开始交，减少开销
        sets.sort(key=len)
        result = set(sets[0])
        for s in sets[1:]:
            result &= s
            if not result:
                return []
        return sorted(result)

    # ------------------------------------------------------------------
    # 相似性查询（Tanimoto + 位数上界剪枝）
    # ------------------------------------------------------------------
    def similarity_search(self, smiles: str, threshold: float = 0.7
                          ) -> Tuple[List[Dict], SearchStats]:
        qmol = canonicalize(standardize(parse_smiles(smiles)))
        qbits = fingerprint_bits(qmol, self.fp_size)
        qpop = len(qbits)
        stats = SearchStats(total=len(self.records))

        t0 = time.perf_counter()
        pruned_by_bound = 0
        candidate_ids: List[int] = []
        for rec in self.records:
            # 需求 5：先用位数上界跳过不可能达到阈值的分子
            if not could_meet_threshold(qpop, rec.popcount, threshold):
                pruned_by_bound += 1
                continue
            candidate_ids.append(rec.mol_id)
        stats.candidates = len(candidate_ids)
        stats.filtered = pruned_by_bound
        stats.prefilter_seconds = time.perf_counter() - t0

        t1 = time.perf_counter()
        hits: List[Dict] = []
        for mid in candidate_ids:
            rec = self.records[mid]
            coef = tanimoto(qbits, rec.fp_bits)
            if coef >= threshold - 1e-12:
                hits.append({
                    "mol_id": mid,
                    "canonical": rec.canonical,
                    "sources": list(rec.sources),
                    "scaffold": rec.scaffold,
                    "tanimoto": round(coef, 6),
                })
        hits.sort(key=lambda h: (-h["tanimoto"], h["mol_id"]))
        stats.exact_matched = len(hits)
        stats.exact_seconds = time.perf_counter() - t1
        return hits, stats

    # ------------------------------------------------------------------
    # 骨架统计
    # ------------------------------------------------------------------
    def scaffold_counts(self, top: Optional[int] = None
                        ) -> List[Tuple[str, int]]:
        counts: Dict[str, int] = {}
        for rec in self.records:
            key = rec.scaffold
            counts[key] = counts.get(key, 0) + 1
        ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        return ordered[:top] if top else ordered

    def group_by_scaffold(self) -> Dict[str, List[int]]:
        groups: Dict[str, List[int]] = {}
        for rec in self.records:
            groups.setdefault(rec.scaffold, []).append(rec.mol_id)
        return groups

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        data = {
            "fp_size": self.fp_size,
            "records": [
                {
                    "mol_id": r.mol_id,
                    "canonical": r.canonical,
                    "fp_bits": sorted(r.fp_bits),
                    "scaffold": r.scaffold,
                    "sources": r.sources,
                } for r in self.records
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "MoleculeDatabase":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        db = cls(fp_size=data.get("fp_size", FP_SIZE))
        for item in data["records"]:
            mol = canonicalize(standardize(parse_smiles(item["canonical"])))
            rec = Record(
                mol_id=item["mol_id"],
                canon_mol=mol,
                fp_bits=set(item["fp_bits"]),
                canonical=item["canonical"],
                scaffold=item["scaffold"],
                sources=item.get("sources", []),
                popcount=len(item["fp_bits"]),
            )
            db.records.append(rec)
            db._hash_to_id[canonical_hash(mol)] = rec.mol_id
            db._scaffold_cache[rec.mol_id] = rec.scaffold
            for bit in rec.fp_bits:
                db._inverted.setdefault(bit, set()).add(rec.mol_id)
        return db
