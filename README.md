# 化合物结构数据库（纯 Python 标准库实现）

面向新药研发场景的化合物结构数据库：自解析线性结构式、规范形式去重、
路径指纹筛选、化学语义子结构匹配、Tanimoto 相似性查询与骨架归类。
**不依赖任何第三方化学信息学库**，仅使用 Python 3.9+ 标准库。

## 模块结构

| 模块 | 职责 |
|------|------|
| `chemdb/elements.py` | 元素、有机子集、芳香符号、价态数据 |
| `chemdb/molecule.py` | 分子图（原子/键/邻接表）、环基缓存、分子式 |
| `chemdb/parser.py` | 递归下降 SMILES 解析器，错误定位到第几个字符 |
| `chemdb/rings.py` | 线性时间环内键标记 + Horton 最小环基（GF(2) 消元） |
| `chemdb/aromaticity.py` | 凯库勒式/芳香式归一、Hückel 校验、凯库勒化（完美匹配） |
| `chemdb/valence.py` | 隐式氢计算与价键校验 |
| `chemdb/canonical.py` | Morgan 迭代精化 + 确定性 DFS 生成规范 SMILES |
| `chemdb/fingerprint.py` | 路径哈希指纹（位向量）、Tanimoto |
| `chemdb/matcher.py` | 化学语义子图同构（回溯 VF2 风格） |
| `chemdb/scaffold.py` | Bemis–Murcko 骨架与通用骨架 |
| `chemdb/database.py` | 入库去重、指纹倒排索引、子结构/相似性查询、SQLite 持久化 |
| `chemdb/generator.py` | 随机合理分子生成（压测用） |
| `web/` | 标准库 http.server 单页应用（分子图可视化 + 原子高亮） |

## 六项需求的实现要点

1. **解析**：原子、键级（`- = # :`）、支链 `()`、环闭合（数字与 `%nn`）、
   电荷、同位素、芳香小写原子、隐式氢（价键补全）与显式氢（`[nH]`）。
   非法写法抛 `ParseError`，`position` 为从 1 开始的字符位置。
2. **规范形式**：Morgan 迭代精化给原子确定性 rank，再按 rank 做确定性 DFS，
   环号在开环/闭合两端规范放置。凯库勒式经芳香归一与芳香式统一，
   同一分子的任意写法得到同一规范 SMILES，入库据此判重。
3. **子结构匹配**：芳香原子只匹配芳香原子、脂肪只匹配脂肪；元素/电荷/同位素
   一致；显式氢作为最小约束（裸原子隐式氢可被取代）；支持 `*` 任意原子、
   `a`/`A` 芳香/脂肪原子、`~` 任意键、`-r` 环内键、`-k` 链键。
4. **指纹筛选**：枚举长度 1..7 的全部简单路径，规范化后双哈希到位向量。
   片段的每条具体路径都对应目标中的同一条路径，故 `bits(片段) ⊆ bits(分子)`，
   倒排索引按位求交，**数学上保证零漏报**；含通配的路径不参与哈希。
5. **相似性查询**：Tanimoto = 交集位/并集位。用位数上下界
   `min(A,B)/max(A,B)` 上界与 `max(0,A+B−N)/max(A,B)` 下界跳过不可能达阈值
   的分子，剪枝不改变结果集。
6. **骨架归类**：反复剥非环端基得到 Bemis–Murcko 骨架，元素通配得到通用骨架，
   按规范骨架 key 统计出现频次。

## 快速使用

```bash
# 解析并查看分子图
python3 -I -m chemdb.cli parse "CC(=O)Oc1ccccc1C(=O)O"
# 规范形式（凯库勒式与芳香式统一）
python3 -I -m chemdb.cli canonical "C1=CC=CC=C1"   # -> c1ccccc1
# Web 界面（含分子图绘制与匹配原子高亮）
python3 -I web/server.py --port 8000 --load 500
# 大规模压测
python3 -I bench/benchmark.py --n 50000
```

代码内调用：

```python
from chemdb import ChemDB
db = ChemDB()
db.add_smiles("c1ccccc1")
db.add_smiles("C1=CC=CC=C1")     # 与上者同规范形式，去重
print(len(db))                    # 1
hits = db.substructure_search("c1ccncc1")
sims = db.similarity_search("c1ccccc1", threshold=0.5)
top = db.scaffold_frequency(top=5)
```

## 测试

```bash
python3 -I tests/test_parser.py       # 解析/规范化/错误定位
python3 -I tests/test_acceptance.py   # 指纹零漏报、子结构==暴力匹配、相似性
python3 -I tests/test_zero_miss.py    # 较大随机库上的零漏报专项
```

## 性能（参考，Mac 笔记本 / 纯 Python）

- 随机合理分子生成约 400 分子/秒，入库约 250–370 分子/秒。
- 子结构查询先经指纹倒排筛除（高选择性片段如苯并咪唑可达 99.9% 淘汰率），
  再对少量候选做图匹配；苯并咪唑骨架在 5 千分子库中约 1–2 毫秒命中。
- 相似性查询的上下界剪枝在高阈值下比全量 Tanimoto 快约 3 倍，结果集不变。
- 分子图带 LRU 缓存，重复查询避免重复解析，热查询再快约一个数量级。

## 查询扩展语法（仅片段）

| 写法 | 含义 |
|------|------|
| `*` | 任意原子 |
| `a` / `A` | 任意芳香 / 脂肪原子 |
| `~` | 任意键级 |
| `c~rc` | 两芳香碳经**环内键**相连 |
| `C-kN` | C 与 N 经**链键**相连（用 k 避免与芳香碳 c 冲突） |
