"""纯 Python 化合物结构数据库（标准库实现，无化学信息学外部依赖）。"""
from .parser import parse_smiles, ParseError
from .molecule import Mol, Atom, Bond
from .canonical import canonical_smiles
from .aromaticity import kekulize
from .fingerprint import fingerprint, tanimoto, path_features
from .matcher import has_substructure, substructure_matches
from .scaffold import scaffold_key, murcko_scaffold, generic_scaffold
from .database import ChemDB
from .valence import compute_implicit_h, check_valence

__all__ = [
    "parse_smiles", "ParseError", "Mol", "Atom", "Bond",
    "canonical_smiles", "kekulize", "fingerprint", "tanimoto",
    "path_features", "has_substructure", "substructure_matches",
    "scaffold_key", "murcko_scaffold", "generic_scaffold",
    "ChemDB", "compute_implicit_h", "check_valence",
]
