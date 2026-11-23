// molkit 前端：调用后端 API，用 SVG 渲染分子图并高亮命中原子。
"use strict";

const $ = (id) => document.getElementById(id);

async function post(url, payload) {
  const resp = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await resp.json();
  if (!resp.ok) throw data;
  return data;
}

function fillExample(s) { $("parseInput").value = s; doParse(); }

// ---------------------------------------------------------------------------
// 图布局：环按正多边形摆放，链按力导向松弛；纯前端确定性计算。
// ---------------------------------------------------------------------------
function layout(graph, width, height) {
  const n = graph.atoms.length;
  const pos = new Array(n);
  const adj = graph.atoms.map(() => []);
  graph.bonds.forEach((b) => { adj[b.a1].push(b.a2); adj[b.a2].push(b.a1); });

  // 环集合：用键的环标记聚合出环（后端给了 in_ring，这里重建简单环）
  const ringEdges = graph.bonds.filter((b) => b.in_ring || b.aromatic);
  const placed = new Set();

  // 从环边提取连通的环原子组，按正多边形布局
  const ringAdj = new Map();
  ringEdges.forEach((b) => {
    if (!ringAdj.has(b.a1)) ringAdj.set(b.a1, new Set());
    if (!ringAdj.has(b.a2)) ringAdj.set(b.a2, new Set());
    ringAdj.get(b.a1).add(b.a2);
    ringAdj.get(b.a2).add(b.a1);
  });
  const ringGroups = [];
  const seenRing = new Set();
  ringAdj.forEach((_, start) => {
    if (seenRing.has(start)) return;
    const stack = [start], group = [];
    seenRing.add(start);
    while (stack.length) {
      const x = stack.pop(); group.push(x);
      ringAdj.get(x).forEach((y) => {
        if (!seenRing.has(y)) { seenRing.add(y); stack.push(y); }
      });
    }
    ringGroups.push(group);
  });

  const cx = width / 2, cy = height / 2;
  const R = Math.min(width, height) * 0.32;
  ringGroups.forEach((group, gi) => {
    const m = group.length;
    const off = gi * 40;
    group.forEach((atomIdx, k) => {
      const ang = -Math.PI / 2 + (2 * Math.PI * k) / m;
      pos[atomIdx] = {
        x: cx + off + R * Math.cos(ang),
        y: cy + R * Math.sin(ang),
      };
      placed.add(atomIdx);
    });
  });

  // 非环原子：从已放置邻居向外生长，剩余的用网格兜底
  let frontier = [...placed];
  let guard = 0;
  while (frontier.length && guard++ < n * 4) {
    const next = [];
    frontier.forEach((u) => {
      adj[u].forEach((v) => {
        if (placed.has(v)) return;
        const base = pos[u];
        const ang = (Math.PI / 3) + (v % 3) * (Math.PI / 3);
        pos[v] = { x: base.x + 34 * Math.cos(ang), y: base.y + 34 * Math.sin(ang) };
        placed.add(v); next.push(v);
      });
    });
    frontier = next;
  }
  let gk = 0;
  for (let i = 0; i < n; i++) {
    if (!placed.has(i)) {
      pos[i] = { x: 30 + (gk % 8) * 36, y: 30 + Math.floor(gk / 8) * 36 };
      gk++;
    }
  }

  // 力导向松弛，让链部分不重叠（固定环上的点）
  for (let iter = 0; iter < 120; iter++) {
    const force = pos.map(() => ({ x: 0, y: 0 }));
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
      const dx = pos[j].x - pos[i].x, dy = pos[j].y - pos[j].y * 0 + pos[j].y - pos[i].y;
      const d2 = dx * dx + dy * dy + 0.01;
      const rep = 900 / d2;
      const d = Math.sqrt(d2);
      force[i].x -= rep * dx / d; force[i].y -= rep * dy / d;
      force[j].x += rep * dx / d; force[j].y += rep * dy / d;
    }
    graph.bonds.forEach((b) => {
      const dx = pos[b.a2].x - pos[b.a1].x, dy = pos[b.a2].y - pos[b.a1].y;
      const d = Math.sqrt(dx * dx + dy * dy) || 1;
      const ideal = 38, pull = (d - ideal) * 0.06;
      force[b.a1].x += pull * dx / d; force[b.a1].y += pull * dy / d;
      force[b.a2].x -= pull * dx / d; force[b.a2].y -= pull * dy / d;
    });
    for (let i = 0; i < n; i++) {
      if (ringGroups.some((g) => g.includes(i))) continue; // 环点固定
      pos[i].x += Math.max(-8, Math.min(8, force[i].x));
      pos[i].y += Math.max(-8, Math.min(8, force[i].y));
    }
  }
  return pos;
}

// ---------------------------------------------------------------------------
// SVG 渲染
// ---------------------------------------------------------------------------
function renderMolecule(graph, container, opts = {}) {
  if (!graph || !graph.atoms.length) { container.innerHTML = ""; return; }
  const W = 360, H = 300;
  const pos = layout(graph, W, H);
  const highlight = new Set(opts.highlight || []);
  let svg = `<svg viewBox="0 0 ${W} ${H}" width="100%" style="max-width:${W}px">`;

  // 高亮光晕
  highlight.forEach((i) => {
    svg += `<circle cx="${pos[i].x}" cy="${pos[i].y}" r="17" fill="var(--hl)" opacity="0.35"/>`;
  });

  // 键
  graph.bonds.forEach((b) => {
    const p1 = pos[b.a1], p2 = pos[b.a2];
    const color = b.aromatic ? "var(--arom)" : "#374151";
    if (b.wildcard) {
      svg += `<line x1="${p1.x}" y1="${p1.y}" x2="${p2.x}" y2="${p2.y}"
               stroke="${color}" stroke-width="2" stroke-dasharray="4 3"/>`;
      return;
    }
    const drawOrder = b.aromatic ? 1 : b.order;
    const offs = drawOrder === 2 ? 3.5 : drawOrder === 3 ? 6 : 0;
    for (let k = 0; k < drawOrder; k++) {
      const o = drawOrder === 1 ? 0 : (k - (drawOrder - 1) / 2) * offs;
      const dx = p2.x - p1.x, dy = p2.y - p1.y;
      const len = Math.sqrt(dx * dx + dy * dy) || 1;
      const nx = -dy / len * o, ny = dx / len * o;
      svg += `<line x1="${p1.x + nx}" y1="${p1.y + ny}" x2="${p2.x + nx}" y2="${p2.y + ny}"
               stroke="${color}" stroke-width="2"/>`;
    }
    if (b.aromatic) {
      svg += `<circle cx="${(p1.x + p2.x) / 2}" cy="${(p1.y + p2.y) / 2}" r="2.4" fill="var(--arom)"/>`;
    }
  });

  // 原子
  graph.atoms.forEach((a) => {
    const p = pos[a.idx];
    const isArom = a.aromatic;
    const fill = a.wildcard ? "#6b7280" : isArom ? "var(--arom)" : "#111827";
    if (highlight.has(a.idx)) {
      svg += `<circle cx="${p.x}" cy="${p.y}" r="12" fill="var(--hl)" opacity="0.9"/>`;
    }
    const label = a.wildcard ? "*" : a.symbol;
    svg += `<text x="${p.x}" y="${p.y}" class="atom-label" fill="${fill}"
             style="paint-order:stroke;stroke:#fff;stroke-width:3px">${label}</text>`;
    if (a.hydrogen > 0 && !a.wildcard) {
      svg += `<text x="${p.x + 11}" y="${p.y - 8}" class="atom-h">H${
        a.hydrogen > 1 ? a.hydrogen : ""}</text>`;
    }
    if (a.charge) {
      const c = a.charge > 0 ? `+${a.charge > 1 ? a.charge : ""}` : `−${-a.charge > 1 ? -a.charge : ""}`;
      svg += `<text x="${p.x + 11}" y="${p.y + 9}" class="atom-h" fill="var(--err)">${c}</text>`;
    }
  });
  svg += "</svg>";
  container.innerHTML = svg;
}

// ---------------------------------------------------------------------------
// 业务交互
// ---------------------------------------------------------------------------
async function doParse() {
  const errEl = $("parseErr"); errEl.textContent = "";
  try {
    const data = await post("/api/parse", { smiles: $("parseInput").value });
    renderMolecule(data.input_graph, $("rawGraph"));
    renderMolecule(data.canonical_graph, $("canonGraph"));
    $("canonSmiles").innerHTML =
      `规范形式：<code>${data.canonical_smiles}</code> · ` +
      `${data.n_atoms} 原子 / ${data.n_bonds} 键`;
  } catch (e) {
    errEl.textContent = e.error || e;
    $("rawGraph").innerHTML = $("canonGraph").innerHTML = "";
    $("canonSmiles").textContent = "";
  }
}

async function doSubstructure() {
  const errEl = $("subErr"); errEl.textContent = "";
  $("subHits").innerHTML = "";
  try {
    const data = await post("/api/substructure", { smiles: $("subInput").value });
    const s = data.stats;
    $("subStats").style.display = "block";
    $("subStats").innerHTML =
      `库内 <b>${s.total}</b> 个分子；指纹预筛后候选 <b>${s.candidates}</b>，` +
      `过滤 <b>${s.filtered}</b>（过滤率 <b>${(s.filtered_ratio * 100).toFixed(1)}%</b>）；` +
      `精确命中 <b>${s.matched}</b>；预筛 ${s.prefilter_ms}ms + 匹配 ${s.exact_ms}ms`;
    if (!data.hits.length) {
      $("subHits").innerHTML = '<p class="muted">没有分子包含该片段。</p>';
      return;
    }
    data.hits.forEach((h) => {
      const div = document.createElement("div");
      div.className = "hit";
      div.innerHTML =
        `<div class="meta"><span>#${h.mol_id} <code>${h.canonical}</code></span>
           <span class="pill">${h.scaffold}</span></div>
         <div id="hit-${h.mol_id}"></div>`;
      $("subHits").appendChild(div);
      // 取该分子规范图并按映射高亮
      fetchGraphAndHighlight(h, div);
    });
  } catch (e) {
    errEl.textContent = e.error || e;
    $("subStats").style.display = "none";
  }
}

async function fetchGraphAndHighlight(hit, div) {
  try {
    const parsed = await post("/api/parse", { smiles: hit.canonical });
    const box = div.querySelector(`#hit-${hit.mol_id}`);
    renderMolecule(parsed.canonical_graph, box,
                   { highlight: hit.highlight_target });
  } catch (e) { /* 忽略单个渲染失败 */ }
}

async function doSimilarity() {
  const errEl = $("simErr"); errEl.textContent = "";
  try {
    const data = await post("/api/similarity", {
      smiles: $("simInput").value,
      threshold: parseFloat($("simThreshold").value),
    });
    const s = data.stats;
    $("simStats").style.display = "block";
    $("simStats").innerHTML =
      `位数上界剪枝 <b>${s.pruned}</b> 个（${(s.pruned_ratio * 100).toFixed(1)}%），` +
      `命中 <b>${data.hits.length}</b>，耗时 ${s.ms}ms`;
    if (!data.hits.length) {
      $("simHits").innerHTML = '<p class="muted">没有达到阈值的相似分子。</p>';
      return;
    }
    $("simHits").innerHTML =
      '<table><thead><tr><th>分子</th><th>骨架</th><th style="width:110px">Tanimoto</th></tr></thead><tbody>'
      + data.hits.map((h) =>
          `<tr><td><code>${h.canonical}</code></td><td>${h.scaffold}</td>
           <td><b>${h.tanimoto.toFixed(3)}</b></td></tr>`).join("")
      + "</tbody></table>";
  } catch (e) {
    errEl.textContent = e.error || e;
    $("simStats").style.display = "none";
  }
}

async function doAdd() {
  try {
    const data = await post("/api/add", { smiles: $("addInput").value });
    $("addMsg").innerHTML = data.is_new
      ? `<span class="ok">已入库，分配 id=${data.mol_id}，当前库大小 ${data.size}</span>`
      : `该分子已存在（id=${data.mol_id}），未重复入库，当前库大小 ${data.size}`;
    loadScaffolds();
  } catch (e) {
    $("addMsg").innerHTML = `<span style="color:var(--err)">${e.error || e}</span>`;
  }
}

async function loadScaffolds() {
  const resp = await fetch("/api/scaffolds");
  const data = await resp.json();
  $("scaffoldBody").innerHTML = data.scaffolds.map((s) =>
    `<tr><td><code>${s.scaffold}</code></td><td>${s.count}</td></tr>`).join("")
    || '<tr><td colspan="2" class="muted">暂无数据</td></tr>';
}

doParse();
loadScaffolds();
