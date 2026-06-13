using JSON, Printf

# ─────────────────────────────────────────────────────────────────────────────
# Utilitaires de formatage des tenseurs
# ─────────────────────────────────────────────────────────────────────────────

function format_tensor_short(value)
    try
        v = Array(value)
        if ndims(v) == 0
            @sprintf("%.4f", Float64(v[]))
        elseif ndims(v) == 1
            n = length(v)
            if n <= 4
                "[" * join([@sprintf("%.4f", Float64(x)) for x in v], ", ") * "]"
            else
                "[" * join([@sprintf("%.4f", Float64(x)) for x in v[1:2]], ", ") * ", …]"
            end
        elseif ndims(v) == 2
            rows, cols = size(v)
            "[$(rows)×$(cols) matrix]"
        else
            "$(ndims(v))-tensor"
        end
    catch
        "?"
    end
end

function format_tensor_full(value)
    try
        v = Array(value)
        if ndims(v) == 0
            return @sprintf("%.4f", Float64(v[]))
        elseif ndims(v) == 1
            return "[" * join([@sprintf("%.4f", Float64(x)) for x in v], ", ") * "]"
        elseif ndims(v) == 2
            rows, cols = size(v)
            row_strs = [join([@sprintf("%.4f", Float64(v[r,c])) for c in 1:cols], "  ") for r in 1:rows]
            return "[" * join(row_strs, "\n ") * "]"
        else
            return "$(ndims(v))-tensor"
        end
    catch
        return "?"
    end
end

function node_formula_text(graph::JuliusGraph, sym::Symbol, ns::Symbol)
    rules_dict = get(graph.rules, ns, Dict())
    if !haskey(rules_dict, sym)
        return string(sym)
    end
    rule = rules_dict[sym]
    op   = rule.op
    inputs = rule.inputs
    if op == :linear
        X, W, b = inputs[1], inputs[2], inputs[3]
        return "$sym = Linear($X,$W,$b)"
    elseif op == :matmul
        A, B = inputs[1], inputs[2]
        tb = get(rule.attrs, :trans_b, false)
        return tb ? "$sym = $A·$Bᵀ" : "$sym = $A·$B"
    elseif op == :relu
        return "$sym = ReLU($(inputs[1]))"
    elseif op == :add
        return "$sym = $(inputs[1])+$(inputs[2])"
    elseif op == :sum_matrix
        return "$sym = Σ $(inputs[1])"
    elseif op == :wsum
        a, b = inputs[1], inputs[2]
        return "$sym = 0.3·$a+0.7·$b"
    elseif op == :nsum
        return "$sym = Σ($(join(inputs, ",")))"
    elseif op == :fused_matmul_relu
        A, B = inputs[1], inputs[2]
        return "$sym = ReLU($A·$B)"
    else
        return "$sym = $op(…)"
    end
end

# ─────────────────────────────────────────────────────────────────────────────
#  save_interactive_graph
#  Layout entièrement délégué à Dagre.js (algorithme Sugiyama = dot de Graphviz)
#  → pas d'intersections, espacement optimal, routage des arêtes avec waypoints
# ─────────────────────────────────────────────────────────────────────────────
function save_interactive_graph(graph::JuliusGraph, log::ExecutionLog,
                                filepath::String; title="NeuroDSL Trace")
    ns       = graph.active_ns
    rules_ns = get(graph.rules, ns, Dict{Symbol,Any}())

    # ── Log ────────────────────────────────────────────────────────────────────
    log_json = JSON.json([Dict(
        :node   => e[:node],
        :phase  => e[:phase],
        :status => e[:status],
        :val    => e[:value]
    ) for e in log.events])

    # ── Nœuds ──────────────────────────────────────────────────────────────────
    init_vals  = Dict{String,String}()
    full_vals  = Dict{String,String}()
    formulas   = Dict{String,String}()
    is_leaf_d  = Dict{String,Bool}()
    is_param_d = Dict{String,Bool}()

    for (sym, nd) in graph.nodes[ns]
        k = string(sym)
        init_vals[k]  = nd.value !== nothing ? format_tensor_short(nd.value) : "?"
        full_vals[k]  = nd.value !== nothing ? format_tensor_full(nd.value)  : "?"
        formulas[k]   = node_formula_text(graph, sym, ns)
        is_leaf_d[k]  = !haskey(rules_ns, sym)
        is_param_d[k] = nd.is_param
    end

    # ── Arêtes ─────────────────────────────────────────────────────────────────
    edges = Tuple{Symbol,Symbol}[]
    for (out_sym, rule) in rules_ns
        for inp in rule.inputs
            push!(edges, (inp, out_sym))
        end
    end

    # ── JSON ───────────────────────────────────────────────────────────────────
    nodes_json    = JSON.json([Dict(:id => string(s),
                                   :is_param => is_param_d[string(s)],
                                   :is_leaf  => is_leaf_d[string(s)])
                               for s in keys(graph.nodes[ns])])
    edges_json    = JSON.json([[string(a), string(b)] for (a, b) in edges])
    init_json     = JSON.json(init_vals)
    full_json     = JSON.json(full_vals)
    formulas_json = JSON.json(formulas)

    # ── HTML ───────────────────────────────────────────────────────────────────
    html = """
<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<title>$title</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/dagre/0.8.5/dagre.min.js"></script>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  background: #f0f2f5; display: flex; height: 100vh; overflow: hidden;
}
/* ── Sidebar ─────────────────────────────────────────────────────────────── */
#sidebar {
  width: 290px; min-width: 290px; background: #fff;
  border-right: 1px solid #e2e8f0; display: flex; flex-direction: column;
  padding: 16px; gap: 10px; z-index: 10;
  box-shadow: 2px 0 10px rgba(0,0,0,0.07);
}
#sidebar h2 { font-size: 13px; color: #1e293b; font-weight: 700; line-height: 1.4; }
.controls   { display: flex; gap: 6px; }
.btn {
  flex: 1; padding: 7px 4px; border-radius: 7px; border: 1px solid #e2e8f0;
  cursor: pointer; background: #fff; font-size: 12px; font-weight: 500;
  color: #334155; transition: all .15s;
}
.btn:hover { background: #f1f5f9; border-color: #94a3b8; }
#playBtn   { background: #3b82f6; color: #fff; border-color: #2563eb; }
#playBtn:hover { background: #2563eb; }
#status { font-size: 11px; color: #94a3b8; text-align: center; }
#log-panel {
  flex: 1; overflow-y: auto; font-family: 'Consolas', monospace; font-size: 10px;
  border: 1px solid #e2e8f0; padding: 8px; background: #fafafa; border-radius: 8px;
}
.log-entry { padding: 3px 6px; border-radius: 4px; margin-bottom: 2px; line-height: 1.5; }
.log-fwd { background: #dbeafe; color: #1d4ed8; }
.log-bwd { background: #fee2e2; color: #b91c1c; }
/* ── Canvas ──────────────────────────────────────────────────────────────── */
#canvas-wrap {
  flex: 1; position: relative; overflow: hidden;
  background: #f8fafc;
  background-image: radial-gradient(circle, #d1d5db 1px, transparent 1px);
  background-size: 24px 24px;
}
#svg-canvas { position: absolute; top: 0; left: 0; transform-origin: 0 0; overflow: visible; }
/* ── Nœuds ───────────────────────────────────────────────────────────────── */
.node-body {
  transition: all .2s ease; cursor: pointer;
  filter: drop-shadow(0 1px 4px rgba(0,0,0,.08));
}
.node-default { fill: #fff;    stroke: #cbd5e0; stroke-width: 1.5px; }
.node-param   { fill: #f1f5f9; stroke: #94a3b8; stroke-width: 1.5px; stroke-dasharray: 5,3; }
.node-fwd     { fill: #dbeafe; stroke: #3b82f6; stroke-width: 2.5px; }
.node-bwd     { fill: #fee2e2; stroke: #ef4444; stroke-width: 2.5px; }
.node-done    { fill: #dcfce7; stroke: #22c55e; stroke-width: 2px;   }
.node-final   {
  fill: #bbf7d0; stroke: #16a34a; stroke-width: 2.5px;
  filter: drop-shadow(0 2px 8px rgba(22,163,74,.4));
}
.node-label { font-family: 'Consolas', monospace; font-size: 10.5px; font-weight: 700;
              fill: #1e293b; pointer-events: none; }
.node-val   { font-family: 'Consolas', monospace; font-size: 9.5px;  fill: #64748b;  pointer-events: none; }
.node-grad  { font-family: 'Consolas', monospace; font-size: 9.5px;  fill: #ef4444;  pointer-events: none; }
/* ── Arêtes ──────────────────────────────────────────────────────────────── */
.edge         { fill: none; stroke-width: 1.5px; transition: all .2s ease; }
.edge-default { stroke: #94a3b8; stroke-opacity: .5; }
.edge-fwd     { stroke: #3b82f6; stroke-width: 2.5px; stroke-opacity: 1; }
.edge-bwd     { stroke: #ef4444; stroke-width: 2.5px; stroke-opacity: 1; }
.edge-final   { stroke: #16a34a; stroke-width: 2.5px; stroke-opacity: 1; }
/* ── Tooltip ─────────────────────────────────────────────────────────────── */
.tooltip {
  position: fixed; background: #1e293b; color: #f1f5f9;
  padding: 10px 14px; border-radius: 10px; font-size: 12px;
  pointer-events: none; display: none; z-index: 9999;
  max-width: 380px; box-shadow: 0 8px 24px rgba(0,0,0,.35);
}
.tooltip b   { color: #7dd3fc; font-family: 'Consolas', monospace; }
.tooltip pre { margin: 5px 0 0; font-family: 'Consolas', monospace; font-size: 10.5px;
               white-space: pre; overflow-x: auto; line-height: 1.5; color: #e2e8f0; }
/* ── Zoom ────────────────────────────────────────────────────────────────── */
.zoom-bar {
  position: absolute; bottom: 14px; right: 14px;
  display: flex; flex-direction: column; gap: 4px;
}
.zoom-btn {
  width: 34px; height: 34px; background: #fff; border: 1px solid #e2e8f0;
  border-radius: 8px; font-size: 17px; cursor: pointer;
  display: flex; align-items: center; justify-content: center;
  box-shadow: 0 2px 6px rgba(0,0,0,.08); transition: background .1s; color: #334155;
}
.zoom-btn:hover { background: #f1f5f9; }
</style>
</head>
<body>

<div id="sidebar">
  <h2>$title</h2>
  <div class="controls">
    <button class="btn" onclick="step(-1)">◀ Prev</button>
    <button class="btn" id="playBtn" onclick="togglePlay()">▶ Play</button>
    <button class="btn" onclick="step(1)">Next ▶</button>
  </div>
  <div id="status">Step : 0 / 0</div>
  <div id="log-panel"></div>
</div>

<div id="canvas-wrap">
  <svg id="svg-canvas"></svg>
  <div class="zoom-bar">
    <button class="zoom-btn" onclick="zoomIn()"  title="Zoom +">+</button>
    <button class="zoom-btn" onclick="zoomOut()" title="Zoom −">−</button>
    <button class="zoom-btn" onclick="zoomFit()" title="Fit">⊡</button>
  </div>
  <div id="tooltip" class="tooltip"></div>
</div>

<script>
// ── Données injectées depuis Julia ────────────────────────────────────────────
const NODES_RAW = $nodes_json;
const EDGES     = $edges_json;
const LOG       = $log_json;
const INIT_VALS = $init_json;
const FULL_VALS = $full_json;
const FORMULAS  = $formulas_json;

// ── Layout via Dagre.js (algorithme Sugiyama / dot de Graphviz) ───────────────
const NW = 185, NH = 72;          // dimensions d'un nœud (px)

const dg = new dagre.graphlib.Graph({ multigraph: false });
dg.setGraph({
  rankdir  : 'LR',   // gauche → droite comme graphviz dot par défaut
  nodesep  : 65,     // espace vertical entre nœuds du même rang
  ranksep  : 110,    // espace horizontal entre rangs
  marginx  : 55,
  marginy  : 55,
  edgesep  : 25,     // espace minimal entre deux arêtes parallèles
  acyclicer: 'greedy',
  ranker   : 'network-simplex'   // meilleur algorithme de rangement
});
dg.setDefaultEdgeLabel(() => ({}));

NODES_RAW.forEach(n => dg.setNode(n.id, { width: NW, height: NH }));
EDGES.forEach(([s, d]) => dg.setEdge(s, d));

dagre.layout(dg);   // ← tout le calcul Sugiyama se passe ici

// Conversion coin supérieur-gauche (Dagre donne le centre)
const NODES = NODES_RAW.map(n => {
  const dn = dg.node(n.id);
  return { ...n, x: Math.round(dn.x - NW/2), y: Math.round(dn.y - NH/2), w: NW, h: NH };
});
const NMAP = Object.fromEntries(NODES.map(n => [n.id, n]));

// Dimensions totales du SVG
const gi = dg.graph();
const SW = Math.round(gi.width  || 800) + 110;
const SH = Math.round(gi.height || 600) + 110;

const svgEl = document.getElementById('svg-canvas');
svgEl.setAttribute('width',  SW);
svgEl.setAttribute('height', SH);

// ── Marqueurs de flèche ───────────────────────────────────────────────────────
const SVGNS = 'http://www.w3.org/2000/svg';
const defs  = document.createElementNS(SVGNS, 'defs');
[
  ['arr-default', '#94a3b8'],
  ['arr-fwd',     '#3b82f6'],
  ['arr-bwd',     '#ef4444'],
  ['arr-final',   '#16a34a'],
].forEach(([id, color]) => {
  const m = document.createElementNS(SVGNS, 'marker');
  m.setAttribute('id',           id);
  m.setAttribute('markerWidth',  '9');
  m.setAttribute('markerHeight', '7');
  m.setAttribute('refX',         '9');
  m.setAttribute('refY',         '3.5');
  m.setAttribute('orient',       'auto');
  const p = document.createElementNS(SVGNS, 'polygon');
  p.setAttribute('points', '0 0, 9 3.5, 0 7');
  p.setAttribute('fill',   color);
  m.appendChild(p); defs.appendChild(m);
});
svgEl.appendChild(defs);

// ── Spline Catmull-Rom → cubiques Bézier ──────────────────────────────────────
// Trace une courbe lisse à travers tous les points (y compris les waypoints Dagre)
function crPath(pts) {
  if (!pts || pts.length < 2) return '';
  const n = pts.length;
  if (n === 2) {
    // S-curve pour une arête directe gauche→droite
    const dx = pts[1].x - pts[0].x;
    return `M \${pts[0].x} \${pts[0].y} C \${pts[0].x + dx*.45} \${pts[0].y}, \${pts[1].x - dx*.45} \${pts[1].y}, \${pts[1].x} \${pts[1].y}`;
  }
  let d = `M \${pts[0].x} \${pts[0].y}`;
  for (let i = 0; i < n - 1; i++) {
    const p0 = pts[Math.max(0, i-1)], p1 = pts[i];
    const p2 = pts[i+1],              p3 = pts[Math.min(n-1, i+2)];
    // Points de contrôle Catmull-Rom
    const cp1x = +(p1.x + (p2.x - p0.x) / 6).toFixed(1);
    const cp1y = +(p1.y + (p2.y - p0.y) / 6).toFixed(1);
    const cp2x = +(p2.x - (p3.x - p1.x) / 6).toFixed(1);
    const cp2y = +(p2.y - (p3.y - p1.y) / 6).toFixed(1);
    d += ` C \${cp1x} \${cp1y} \${cp2x} \${cp2y} \${p2.x} \${p2.y}`;
  }
  return d;
}

// Assemble : port de sortie du nœud source → waypoints Dagre → port d'entrée du nœud cible
function edgePts(src, dst) {
  const ed = dg.edge(src, dst);             // waypoints calculés par Dagre
  const sn = NMAP[src], dn = NMAP[dst];
  if (!sn || !dn) return [];
  const start = { x: sn.x + sn.w,  y: sn.y + sn.h / 2 };
  const end   = { x: dn.x,          y: dn.y + dn.h / 2  };
  const wpts  = (ed && ed.points) ? ed.points : [];
  return [start, ...wpts, end];
}

function trunc(s, max=24) { return s.length > max ? s.slice(0, max-1) + '…' : s; }

// ── État partagé ──────────────────────────────────────────────────────────────
const nodeVals = Object.fromEntries(
  NODES.map(n => [n.id, { fwd: FULL_VALS[n.id] || '?', bwd: '' }])
);
const tooltip = document.getElementById('tooltip');

// ── Construction initiale du SVG ──────────────────────────────────────────────
function init() {
  // Couche arêtes (z-order bas, sous les nœuds)
  const eLayer = document.createElementNS(SVGNS, 'g');
  eLayer.id = 'edge-layer';
  EDGES.forEach(([s, d]) => {
    const path = document.createElementNS(SVGNS, 'path');
    path.setAttribute('d',          crPath(edgePts(s, d)));
    path.setAttribute('class',      'edge edge-default');
    path.setAttribute('id',         `edge-\${s}-\${d}`);
    path.setAttribute('marker-end', 'url(#arr-default)');
    eLayer.appendChild(path);
  });
  svgEl.appendChild(eLayer);

  // Couche nœuds
  const nLayer = document.createElementNS(SVGNS, 'g');
  nLayer.id = 'node-layer';

  NODES.forEach(n => {
    const g = document.createElementNS(SVGNS, 'g');

    // Rectangle principal
    const rect = document.createElementNS(SVGNS, 'rect');
    rect.setAttribute('x',      n.x);  rect.setAttribute('y',      n.y);
    rect.setAttribute('width',  n.w);  rect.setAttribute('height', n.h);
    rect.setAttribute('rx',     '8');
    rect.setAttribute('id',     `node-\${n.id}`);
    rect.setAttribute('class',  'node-body ' + (n.is_param ? 'node-param' : 'node-default'));

    // Tooltip au survol
    rect.addEventListener('mouseenter', e => {
      const v = nodeVals[n.id];
      let htm = `<b>\${n.id}</b>`;
      if (v.fwd && v.fwd !== '?') htm += `<br>fwd :<pre>\${v.fwd}</pre>`;
      if (v.bwd)                   htm += `<br>grad :<pre>\${v.bwd}</pre>`;
      tooltip.style.display = 'block';
      const r = e.target.getBoundingClientRect();
      tooltip.style.left = (r.right + 14) + 'px';
      tooltip.style.top  = Math.max(8, r.top - 8) + 'px';
      tooltip.innerHTML  = htm;
    });
    rect.addEventListener('mouseleave', () => { tooltip.style.display = 'none'; });

    // Séparateur visuel entre formule et valeurs
    const sep = document.createElementNS(SVGNS, 'line');
    sep.setAttribute('x1', n.x + 10);      sep.setAttribute('x2', n.x + n.w - 10);
    sep.setAttribute('y1', n.y + 30);      sep.setAttribute('y2', n.y + 30);
    sep.setAttribute('stroke', '#e2e8f0'); sep.setAttribute('stroke-width', '1');
    sep.setAttribute('pointer-events', 'none');

    // Textes
    const mkT = (id, dy, cls) => {
      const t = document.createElementNS(SVGNS, 'text');
      t.setAttribute('x', n.x + n.w / 2); t.setAttribute('y', n.y + dy);
      t.setAttribute('text-anchor', 'middle'); t.setAttribute('class', cls);
      if (id) t.setAttribute('id', id);
      return t;
    };

    const tLabel = mkT(null,             21, 'node-label');
    tLabel.textContent = trunc(FORMULAS[n.id] || n.id);

    const tVal  = mkT(`val-\${n.id}`,  44, 'node-val');
    tVal.textContent   = (n.is_leaf && INIT_VALS[n.id] !== '?') ? INIT_VALS[n.id] : '';

    const tGrad = mkT(`grad-\${n.id}`, 60, 'node-grad');
    tGrad.textContent  = '';

    g.append(rect, sep, tLabel, tVal, tGrad);
    nLayer.appendChild(g);
  });
  svgEl.appendChild(nLayer);
}

// ── Helpers de mise à jour ────────────────────────────────────────────────────
function setNC(id, cls) {
  const r = document.getElementById(`node-\${id}`);
  if (r) r.setAttribute('class', `node-body \${cls}`);
}
function setEC(s, d, cls, arr) {
  const e = document.getElementById(`edge-\${s}-\${d}`);
  if (e) {
    e.setAttribute('class',      `edge \${cls}`);
    e.setAttribute('marker-end', `url(#\${arr})`);
  }
}

// ── Lecture pas à pas ─────────────────────────────────────────────────────────
let step_i = -1, playing = false, timer = null;

function step(dir) {
  step_i = Math.max(-1, Math.min(LOG.length - 1, step_i + dir));
  updateUI();
}

function updateUI() {
  document.getElementById('status').textContent = `Step : \${step_i + 1} / \${LOG.length}`;

  // Remise à zéro de l'affichage
  NODES.forEach(n => {
    setNC(n.id, n.is_param ? 'node-param' : 'node-default');
    const vt = document.getElementById(`val-\${n.id}`);
    if (vt) vt.textContent = (n.is_leaf && INIT_VALS[n.id] !== '?') ? INIT_VALS[n.id] : '';
    const gt = document.getElementById(`grad-\${n.id}`);
    if (gt) gt.textContent = '';
    nodeVals[n.id] = { fwd: FULL_VALS[n.id] || '?', bwd: '' };
  });
  EDGES.forEach(([s, d]) => setEC(s, d, 'edge-default', 'arr-default'));

  if (step_i === -1) { document.getElementById('log-panel').innerHTML = ''; return; }

  // Rejeu des évènements jusqu'à step_i
  let lastFwd = null;
  for (let i = 0; i <= step_i; i++) {
    const ev = LOG[i];
    if (ev.status === 'starting') {
      setNC(ev.node, ev.phase === 'forward' ? 'node-fwd' : 'node-bwd');
    } else {
      if (ev.phase === 'forward') {
        setNC(ev.node, 'node-default');
        lastFwd = ev.node;
        const vt = document.getElementById(`val-\${ev.node}`);
        if (vt) vt.textContent = INIT_VALS[ev.node] || ev.val || '';
        nodeVals[ev.node].fwd = FULL_VALS[ev.node] || ev.val;
      } else {
        setNC(ev.node, 'node-done');
        const gt = document.getElementById(`grad-\${ev.node}`);
        if (gt) gt.textContent = ev.val || '';
        nodeVals[ev.node].bwd = ev.val || '';
      }
    }
    // Colorier les arêtes entrantes du nœud actif
    EDGES.forEach(([s, d]) => {
      if (d === ev.node) {
        const [ec, ea] = ev.phase === 'forward'
          ? ['edge-fwd', 'arr-fwd']
          : ['edge-bwd', 'arr-bwd'];
        setEC(s, d, ec, ea);
      }
    });
  }

  // Dernier nœud forward calculé → surbrillance verte
  if (lastFwd) {
    setNC(lastFwd, 'node-final');
    EDGES.forEach(([s, d]) => {
      if (d === lastFwd) setEC(s, d, 'edge-final', 'arr-final');
    });
  }

  // Panneau de log (ordre chronologique inversé)
  document.getElementById('log-panel').innerHTML =
    LOG.slice(0, step_i + 1).slice().reverse().map(ev =>
      `<div class="log-entry log-\${ev.phase}">` +
      `<b>\${ev.phase.toUpperCase()}</b> \${ev.node} ` +
      `\${ev.status === 'starting' ? '→ computing…' : ': ' + (ev.val || '')}` +
      `</div>`
    ).join('');
}

function togglePlay() {
  playing = !playing;
  const btn = document.getElementById('playBtn');
  btn.textContent = playing ? '⏸ Pause' : '▶ Play';
  if (playing) {
    timer = setInterval(() => {
      if (step_i < LOG.length - 1) step(1);
      else { playing = false; clearInterval(timer); btn.textContent = '▶ Play'; }
    }, 500);
  } else clearInterval(timer);
}

// ── Pan & Zoom ────────────────────────────────────────────────────────────────
let sc = 1, tx = 0, ty = 0, pan = false, px = 0, py = 0;
const wrap = document.getElementById('canvas-wrap');

function applyT() { svgEl.style.transform = `translate(\${tx}px,\${ty}px) scale(\${sc})`; }
function zoomIn()  { sc *= 1.2; applyT(); }
function zoomOut() { sc /= 1.2; applyT(); }
function zoomFit() {
  const ww = wrap.clientWidth, wh = wrap.clientHeight;
  sc = Math.min(ww / SW, wh / SH) * 0.9;
  tx = (ww - SW * sc) / 2;
  ty = (wh - SH * sc) / 2;
  applyT();
}

wrap.addEventListener('wheel', e => {
  e.preventDefault();
  const r  = wrap.getBoundingClientRect();
  const mx = e.clientX - r.left, my = e.clientY - r.top;
  const f  = e.deltaY > 0 ? 0.9 : 1.1;
  const ns2 = sc * f;
  tx = mx - (mx - tx) * (ns2 / sc);
  ty = my - (my - ty) * (ns2 / sc);
  sc = ns2; applyT();
}, { passive: false });

wrap.addEventListener('mousedown', e => {
  if (e.button === 0) {
    pan = true; px = e.clientX - tx; py = e.clientY - ty;
    svgEl.style.cursor = 'grabbing'; e.preventDefault();
  }
});
window.addEventListener('mousemove', e => {
  if (pan) { tx = e.clientX - px; ty = e.clientY - py; applyT(); }
});
window.addEventListener('mouseup', () => { pan = false; svgEl.style.cursor = ''; });

// ── Démarrage ─────────────────────────────────────────────────────────────────
init();
requestAnimationFrame(zoomFit);   // auto-fit initial après rendu
</script>
</body>
</html>
"""

    write(filepath, html)
    println("✅ Interactive Trace exporté → $filepath")
end