# molkit：自研化合物结构数据库

面向新药研发的化合物结构存储与检索系统。从线性结构式（SMILES 方言）的词法
语法解析，到芳香性识别、分子规范化、结构指纹、子结构/相似性检索与骨架归类，
**全部自研，不依赖任何第三方化学信息学库**，仅使用 Python 标准库。

## 能力对照需求

| # | 需求 | 实现 |
|---|------|------|
| 1 | 解析原子/键级/支链/环闭合/电荷/同位素/芳香小写/显隐氢，非法报字符位置 | `molkit/parser.py`（`ParseError.column` 为 1 基字符位置） |
| 2 | 同物同码，凯库勒式与芳香式统一，按规范形式去重 | `molkit/aromatic.py` + `molkit/canonical.py` |
| 3 | 子结构匹配，任意原子/键、环内键查询，芳香碳不匹配脂肪碳 | `molkit/matcher.py`（VF2 回溯 + 化学语义相容性） |
| 4 | 路径指纹预筛，片段指纹位是分子指纹位的子集，保证不漏 | `molkit/fingerprint.py` |
| 5 | Tanimoto 相似性，用位数上下界跳过不可能达阈值的分子 | `fingerprint.tanimoto_bounds` / `could_meet_threshold` |
| 6 | Murcko 骨架归类与频次统计 | `molkit/scaffold.py` |
| — | 几十万随机合理分子验收、暴力对照、筛选率与耗时曲线 | `tools/benchmark.py`、`molkit/generator.py` |
| — | 网页解析分子图、片段命中并高亮原子 | `tools/server.py` + `web_static/` |

## 关键设计

### 1. 解析（parser.py）
递归下降解析器，支持有机子集裸原子、方括号原子（同位素/氢/电荷/查询标记）、
单键 `-`、双键 `=`、三键 `#`、芳香键 `:`、任意键 `~`、环内键 `@`/`@!`、
单数字与 `%nn` 环闭合、点断多片段。所有语法错误抛 `ParseError`，
`column` 精确到第几个字符。

### 2. 芳香性统一（aromatic.py）
- 小写芳香式保留芳香标记；大写凯库勒式经 Hückel 规则（4n+2 π 电子、
  环原子 sp2）重新感知芳香环。
- 两条路径都进入同一个凯库勒化：在芳香子图上做带杂原子豁免的完美匹配，
  把芳香键确定性地指派为交替单/双键，元素符号统一大写。
- 最终规范图只含整数键级、不含芳香标记；芳香性是可由环+双键重新派生的属性。

### 3. 规范化（canonical.py）
- 初始不变量（元素/电荷/同位素/总氢/度/环尺寸）→ 按内容字典序排名。
- Weisfeiler-Lehman 颜色细化到分区稳定（按分区而非颜色号判收敛）。
- 仍对称的原子用个体优先 + 回溯求规范排列，带节点预算，超预算回退到
  确定性的颜色+idx 排序，避免高对称稠环的组合爆炸。
- 环内双键布局按规范位置最小化，使同一芳香环的不同合法写法落到同一表示。
- 显式氢折叠为隐式（同位素氢除外），`[NH]` 与隐式 N-H 同码。

### 4. 指纹与不漏筛（fingerprint.py）
路径指纹枚举所有简单路径并哈希。原子 token 只保留子图匹配下不变的内禀属性
（元素/电荷/同位素/芳香性），不含度、氢数、环标记——这些会随片段嵌入目标而
改变，放进去会破坏子集性。含通配 `*`/`~` 的路径不参与子集预筛（通配匹配
任意，无法形成排除约束）。由此 `bits(片段) ⊆ bits(分子)` 对任意真实子结构成立，
预筛只淘汰不满足子集关系的分子，**保证零漏检**。

### 5. 子结构与相似性（matcher.py / database.py）
- VF2 风格回溯子图同构，候选按约束强度排序；芳香/脂肪原子与键严格不混配，
  支持任意原子 `*`、任意键 `~`、环内/环外 `R`/`!R`、环大小 `r6`、氢数查询。
- 数据库用指纹位→分子 id 的倒排表做交集预筛，再逐候选精确匹配。
- 相似性先用 popcount 的 Tanimoto 上界剪枝，再对幸存者算真实系数。

## 使用

```python
from molkit import parse_smiles, standardize, canonicalize, canonical_smiles
from molkit.database import MoleculeDatabase

db = MoleculeDatabase()
db.add_smiles("c1ccccc1")
db.add_smiles("C1=CC=CC=C1")   # 与上同物，去重
print(len(db))                  # 1

hits, stats = db.substructure_search("c1ccncc1")
hits, stats = db.similarity_search("CCO", threshold=0.6)
print(db.scaffold_counts(top=10))
```

## 验收基准

```bash
python3 -I tools/benchmark.py --n 100000 \
    --scaling 10000,25000,50000,100000
```
检查：等价写法规范一致性、指纹预筛结果与逐分子暴力匹配完全一致、
相似性与全量 Tanimoto 一致、指纹筛选比例、查询耗时随库大小曲线、骨架 top。

## 网页

```bash
python3 -I tools/server.py        # http://127.0.0.1:8000
```
输入结构式看原始/规范分子图；输入片段看命中分子并高亮匹配原子；
相似性查询与骨架统计；可往库里加分子并实时去重。

## 测试

```bash
python3 -I -m unittest discover -s tests -p "test_*.py"
```

## 目录

```
molkit/        核心库（解析/芳香/规范/指纹/匹配/骨架/数据库/生成器）
tools/         server.py 网页服务、benchmark.py 验收基准
tests/         单元与集成测试
web_static/    前端页面与 SVG 渲染
```
