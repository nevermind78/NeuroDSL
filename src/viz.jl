using JSON, Printf

# ═════════════════════════════════════════════════════════════════════════════
# Types pour les snapshots d'entraînement
# ═════════════════════════════════════════════════════════════════════════════
mutable struct TrainingSnapshot
    epoch  :: Int
    iter   :: Int
    loss   :: Float32
    log    :: ExecutionLog
    params :: Dict{Symbol, AbstractArray{Float32}}
    # Constructeur avec paramètres optionnel (params vide par défaut)
    function TrainingSnapshot(epoch, iter, loss, log, params = Dict{Symbol, AbstractArray{Float32}}())
        new(epoch, iter, loss, log, params)
    end
end

mutable struct TrainingRecorder
    snapshots       :: Vector{TrainingSnapshot}
    capture_epochs  :: Set{Int}
end

function TrainingRecorder(; capture_epochs=[1,10,50,100,200,300,500])
    TrainingRecorder(TrainingSnapshot[], Set(capture_epochs))
end

function should_capture(rec::TrainingRecorder, epoch::Int)
    epoch ∈ rec.capture_epochs
end

# ═════════════════════════════════════════════════════════════════════════════
# Utilitaires de formatage (identiques pour les deux versions)
# ═════════════════════════════════════════════════════════════════════════════
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

function node_formula_text(graph::NeuroGraph, sym::Symbol, ns::Symbol)
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
        return tb ? "$sym = $A·$(B)ᵀ" : "$sym = $A·$B"
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

# ═════════════════════════════════════════════════════════════════════════════
# save_interactive_graph — version simple (pour un seul forward/backward)
# ═════════════════════════════════════════════════════════════════════════════
function save_interactive_graph(graph::NeuroGraph, log::ExecutionLog,
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
        init_vals[k] = nd.value !== nothing ? begin
                                                    s = size(nd.value)
                                                    "[" * join(s, "×") * "]"
                                                end : "?"
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
.tooltip pre {
  margin:5px 0 0;
  font-family:'Consolas',monospace; font-size:10px;
  white-space:pre; color:#e2e8f0;
  overflow-x:auto;    /* ← ajout pour permettre le défilement horizontal */
  max-width:100%;      /* garantit que le pre ne dépasse pas la tooltip */
}
.tooltip-left, .tooltip-right {
  position:fixed;
  background:#1e2030;
  color:#e5e7eb;
  padding:12px 16px;
  border-radius:12px;
  font-size:12px;
  pointer-events:none;
  display:none;
  z-index:9999;
  max-width:600px;
  max-height:70vh;
  overflow:auto;
  box-shadow:0 8px 24px rgba(0,0,0,.5);
  border:1px solid #3b3f53;
  font-family: 'Consolas', monospace;
}
.tooltip-left {
  left: auto;
  right: calc(100% + 14px);
}
.tooltip-right {
  left: calc(100% + 14px);
  right: auto;
}
.tooltip-left b, .tooltip-right b { color:#d4a373; }
.tooltip-left pre, .tooltip-right pre {
  margin:6px 0 0;
  font-size:10px;
  white-space:pre;
  color:#d1d5db;
}
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
  <div id="tooltip-left" class="tooltip-left"></div>
  <div id="tooltip-right" class="tooltip-right"></div>
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

// ── Points d'une arête : waypoints Dagre (déjà correctement placés) ─────────
function edgePts(src, dst) {
  const ed = dg.edge(src, dst);             // waypoints calculés par Dagre
  if (ed && ed.points && ed.points.length >= 2) {
    // Les waypoints commencent sur le bord du nœud source et finissent sur le bord du nœud destination
    return ed.points;
  }
  // Fallback : calculer l'intersection entre le segment centre‑centre et les rectangles
  const sn = NMAP[src], dn = NMAP[dst];
  if (!sn || !dn) return [];
  const sx = sn.x + sn.w/2, sy = sn.y + sn.h/2;
  const dx = dn.x + dn.w/2, dy = dn.y + dn.h/2;
  // Fonction pour trouver l'intersection d'un rayon avec un rectangle (bords)
  const intersect = (rx, ry, rw, rh, x1, y1, x2, y2) => {
    const left = rx, right = rx + rw, top = ry, bottom = ry + rh;
    const pts = [];
    // Haut
    if (y1 !== y2) {
      const t = (top - y1) / (y2 - y1);
      if (t >= 0 && t <= 1) {
        const xi = x1 + t * (x2 - x1);
        if (xi >= left && xi <= right) pts.push({x: xi, y: top});
      }
    }
    // Bas
    if (y1 !== y2) {
      const t = (bottom - y1) / (y2 - y1);
      if (t >= 0 && t <= 1) {
        const xi = x1 + t * (x2 - x1);
        if (xi >= left && xi <= right) pts.push({x: xi, y: bottom});
      }
    }
    // Gauche
    if (x1 !== x2) {
      const t = (left - x1) / (x2 - x1);
      if (t >= 0 && t <= 1) {
        const yi = y1 + t * (y2 - y1);
        if (yi >= top && yi <= bottom) pts.push({x: left, y: yi});
      }
    }
    // Droite
    if (x1 !== x2) {
      const t = (right - x1) / (x2 - x1);
      if (t >= 0 && t <= 1) {
        const yi = y1 + t * (y2 - y1);
        if (yi >= top && yi <= bottom) pts.push({x: right, y: yi});
      }
    }
    // Prendre le point le plus proche de (x1,y1) (source)
    if (pts.length === 0) return {x: x1, y: y1};
    pts.sort((a,b) => (a.x-x1)**2 + (a.y-y1)**2 - (b.x-x1)**2 + (b.y-y1)**2);
    return pts[0];
  };
  const start = intersect(sn.x, sn.y, sn.w, sn.h, sx, sy, dx, dy);
  const end   = intersect(dn.x, dn.y, dn.w, dn.h, dx, dy, sx, sy);
  return [start, end];
}

// ── Tracé d'une ligne brisée (ou courbe si suffisamment de points) ──────────
function polylinePath(pts) {
  if (!pts || pts.length < 2) return '';
  // Si seulement 2 points, utiliser une courbe quadratique simple
  if (pts.length === 2) {
    const dx = pts[1].x - pts[0].x;
    const dy = pts[1].y - pts[0].y;
    return `M \${pts[0].x} \${pts[0].y} Q \${pts[0].x + dx/2} \${pts[0].y + dy/2}, \${pts[1].x} \${pts[1].y}`;
  }
  // Pour plus de 2 points, ligne brisée (les angles seront adoucis par stroke-linejoin:round)
  let d = `M \${pts[0].x} \${pts[0].y}`;
  for (let i = 1; i < pts.length; i++) {
    d += ` L \${pts[i].x} \${pts[i].y}`;
  }
  return d;
}

function trunc(s, max=24) { return s.length > max ? s.slice(0, max-1) + '…' : s; }

// ── État partagé ──────────────────────────────────────────────────────────────
const nodeVals = Object.fromEntries(
  NODES.map(n => [n.id, { fwd: FULL_VALS[n.id] || '?', bwd: '' }])
);
const tooltipLeft = document.getElementById('tooltip-left');
const tooltipRight = document.getElementById('tooltip-right');

let currentParamValues = {};
let currentParamFull = {};

// ── Construction initiale du SVG ──────────────────────────────────────────────
function init() {
  // Couche arêtes (z-order bas, sous les nœuds)
  const eLayer = document.createElementNS(SVGNS, 'g');
  eLayer.id = 'edge-layer';
  EDGES.forEach(([s, d]) => {
    const path = document.createElementNS(SVGNS, 'path');
    path.setAttribute('d',          polylinePath(edgePts(s, d)));
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
    // Tooltip gauche : poids (version complète)
    if (n.is_param && currentParamFull[n.id]) {
        tooltipLeft.innerHTML = "<b>📦 Poids (" + n.id + ")</b><pre>" + currentParamFull[n.id] + "</pre>";
        tooltipLeft.style.display = 'block';
        const r = e.target.getBoundingClientRect();
        tooltipLeft.style.right = (window.innerWidth - r.left + 14) + 'px';
        tooltipLeft.style.top = Math.max(8, r.top - 8) + 'px';
    } else {
        tooltipLeft.style.display = 'none';
    }
    // Tooltip droite : forward + gradient
    let rightHtml = "<b>➡️ Forward (" + n.id + ")</b><pre>" + (v.fwd || '?') + "</pre>";
    if (v.bwd) rightHtml += "<b>🔻 Gradient (" + n.id + ")</b><pre>" + v.bwd + "</pre>";
    tooltipRight.innerHTML = rightHtml;
    tooltipRight.style.display = 'block';
    const r = e.target.getBoundingClientRect();
    tooltipRight.style.left = (r.right + 14) + 'px';
    tooltipRight.style.top = Math.max(8, r.top - 8) + 'px';
});
rect.addEventListener('mouseleave', () => {
    tooltipLeft.style.display = 'none';
    tooltipRight.style.display = 'none';
});
    

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

# ═════════════════════════════════════════════════════════════════════════════
# save_interactive_graph_animated — version avec slider et side‑bar redimensionnable
# ═════════════════════════════════════════════════════════════════════════════
function save_interactive_graph_animated(
        graph::NeuroGraph,
        snapshots::Vector{TrainingSnapshot},
        filepath::String;
        title="NeuroDSL Training Trace",
        losses::Vector{Float32}=Float32[])

    ns       = graph.active_ns
    rules_ns = get(graph.rules, ns, Dict{Symbol,Any}())

    # ── Nœuds et arêtes ──────────────────────────────────────────────────────
    init_vals  = Dict{String,String}()
    full_vals  = Dict{String,String}()
    formulas   = Dict{String,String}()
    is_leaf_d  = Dict{String,Bool}()
    is_param_d = Dict{String,Bool}()

    for (sym, nd) in graph.nodes[ns]
        k = string(sym)
        init_vals[k] = nd.value !== nothing ? begin
                                                      s = size(nd.value)
                                                      "[" * join(s, "×") * "]"
                                                  end : "?"
        full_vals[k]  = nd.value !== nothing ? format_tensor_full(nd.value)  : "?"
        formulas[k]   = node_formula_text(graph, sym, ns)
        is_leaf_d[k]  = !haskey(rules_ns, sym)
        is_param_d[k] = nd.is_param
    end

    edges = Tuple{Symbol,Symbol}[]
    for (out_sym, rule) in rules_ns
        for inp in rule.inputs
            push!(edges, (inp, out_sym))
        end
    end

    # ── Snapshots → JSON ─────────────────────────────────────────────────────
    snaps_json = JSON.json([Dict(
      :epoch => s.epoch,
      :iter  => s.iter,
      :loss  => Float64(s.loss),
      :events => [Dict(
          :node   => string(e[:node]),
          :phase  => e[:phase],
          :status => e[:status],
          :val    => something(e[:value], "")
      ) for e in s.log.events],
      :params => Dict(string(k) => format_tensor_short(v) for (k,v) in s.params),
      :params_full => Dict(string(k) => format_tensor_full(v) for (k,v) in s.params)
  ) for s in snapshots])

    losses_json = JSON.json([Float64(l) for l in losses])

    nodes_json    = JSON.json([Dict(:id => string(s),
                                   :is_param => is_param_d[string(s)],
                                   :is_leaf  => is_leaf_d[string(s)])
                               for s in keys(graph.nodes[ns])])
    edges_json    = JSON.json([[string(a), string(b)] for (a, b) in edges])
    init_json     = JSON.json(init_vals)
    full_json     = JSON.json(full_vals)
    formulas_json = JSON.json(formulas)

    # ── Époques capturées ────────────────────────────────────────────────────
    epoch_to_snapshots = Dict{Int,Vector{TrainingSnapshot}}()
    for s in snapshots
        push!(get!(epoch_to_snapshots, s.epoch, TrainingSnapshot[]), s)
    end
    epochs = sort(collect(keys(epoch_to_snapshots)))
    epochs_json = JSON.json(epochs)

    html = """
<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<title>$title</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/dagre/0.8.5/dagre.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
* { box-sizing:border-box; margin:0; padding:0; }
body {
  font-family:'Segoe UI',system-ui,sans-serif;
  background:#12141c; color:#d1d5db;
  display:flex; flex-direction:column; height:100vh; overflow:hidden;
}

/* ── Top bar ─────────────────────────────────────────────── */
#topbar {
  background:#1e2030; border-bottom:1px solid #2d3143;
  padding:10px 20px; display:flex; align-items:center; gap:16px;
  flex-shrink:0; flex-wrap:wrap;
}
#topbar h1 { font-size:0.95rem; font-weight:700; color:#d4a373; white-space:nowrap; }

/* ── Slider + input ──────────────────────────────────────── */
.epoch-selector { display:flex; align-items:center; gap:8px; }
.epoch-selector input[type=range] { width:180px; accent-color:#d4a373; }
.epoch-selector input[type=number] {
  width:55px; background:#1e2030; border:1px solid #3b3f53;
  color:#d1d5db; border-radius:4px; padding:2px 5px; font-size:0.8rem;
  text-align:center;
}
.epoch-selector button {
  background:#1e2030; border:1px solid #3b3f53; color:#9ca3af;
  border-radius:4px; padding:2px 8px; cursor:pointer; font-size:0.8rem;
}
.epoch-selector button:hover { background:#2d3143; color:#e5e7eb; }

/* ── Step controls ───────────────────────────────────────── */
.step-controls { display:flex; gap:6px; align-items:center; margin-left:auto; }
.btn {
  padding:5px 14px; border-radius:7px; border:1px solid #3b3f53;
  background:#1e2030; color:#9ca3af; cursor:pointer; font-size:0.8rem;
  font-weight:500; transition:all .15s; white-space:nowrap;
}
.btn:hover   { background:#2d3143; border-color:#d4a373; color:#d4a373; }
.btn.primary { background:#d4a373; border-color:#d4a373; color:#1e2030; }
.btn.primary:hover { background:#b5835a; }
#stepStatus  { font-size:0.75rem; color:#6b7280; white-space:nowrap; }

/* ── Main layout ─────────────────────────────────────────── */
#main { display:flex; flex:1; overflow:hidden; }

/* ── Sidebar ──────────────────────────────────────────── */
#sidebar {
  width:300px; min-width:200px; max-width:600px;
  background:#1e2030; border-right:1px solid #2d3143;
  display:flex; flex-direction:column; padding:12px; gap:10px;
  overflow:hidden; position:relative;
}
#sidebar .resize-handle {
  position:absolute; top:0; right:0; width:5px; height:100%;
  cursor:ew-resize; background:transparent;
}
#sidebar .resize-handle:hover { background:#d4a37355; }
#sidebar h3 { font-size:0.75rem; color:#9ca3af; text-transform:uppercase;
              letter-spacing:.5px; font-weight:600; }

/* ── Loss panel (redimensionnable en hauteur) ─────────── */
#loss-panel {
  border-bottom:1px solid #2d3143;
  position:relative;
  overflow:hidden;
}
#loss-panel .resize-v-handle {
  position:absolute; bottom:0; left:0; right:0; height:5px;
  cursor:ns-resize; background:transparent; z-index:10;
}
#loss-panel .resize-v-handle:hover { background:#d4a37355; }
#loss-chart-wrap { width:100%; height:100%; }

/* ── Log panel ────────────────────────────────────────── */
#log-panel {
  flex:1; overflow-y:auto; font-family:'Consolas',monospace; font-size:10px;
  border:1px solid #2d3143; padding:6px; background:#12141c; border-radius:6px;
  color:#d1d5db;
}
.log-entry { padding:2px 5px; border-radius:3px; margin-bottom:2px; line-height:1.5; }
.log-fwd { background:#1a2530; color:#7aa2f7; }
.log-bwd { background:#2e1a1a; color:#e06c75; }

/* ── Canvas ──────────────────────────────────────────────── */
#canvas-wrap {
  flex:1; position:relative; overflow:hidden;
  background:#12141c;
  background-image:radial-gradient(circle, #1e2030 1px, transparent 1px);
  background-size:24px 24px;
}
#svg-canvas { position:absolute; top:0; left:0; transform-origin:0 0; overflow:visible; }

/* ── Nœuds ───────────────────────────────────────────────── */
.node-body { transition:all .2s; cursor:pointer;
             filter:drop-shadow(0 1px 4px rgba(0,0,0,.2)); }
.node-default { fill:#1e2030; stroke:#3b3f53;  stroke-width:1.5px; }
.node-param   { fill:#232636; stroke:#4b5268;  stroke-width:1.5px; stroke-dasharray:5,3; }
.node-fwd     { fill:#1a2530; stroke:#7aa2f7;  stroke-width:2.5px; }
.node-bwd     { fill:#2e1a1a; stroke:#e06c75;  stroke-width:2.5px; }
.node-done    { fill:#1a2e1a; stroke:#6a9b6a;  stroke-width:2px;   }
.node-final   { fill:#1e3320; stroke:#4caf50;  stroke-width:2.5px;
                filter:drop-shadow(0 2px 8px rgba(76,175,80,.5)); }
.node-label { font-family:'Consolas',monospace; font-size:10.5px; font-weight:700;
              fill:#d1d5db; pointer-events:none; }
.node-val   { font-family:'Consolas',monospace; font-size:9px; fill:#9ca3af; pointer-events:none; }
.node-grad  { font-family:'Consolas',monospace; font-size:9px; fill:#e06c75; pointer-events:none; }

/* ── Arêtes ──────────────────────────────────────────────── */
.edge         { fill:none; stroke-width:1.5px; stroke-linejoin:round; }
.edge-default { stroke:#3b3f53; stroke-opacity:.6; }
.edge-fwd     { stroke:#7aa2f7; stroke-width:2.5px; }
.edge-bwd     { stroke:#e06c75; stroke-width:2.5px; }
.edge-final   { stroke:#4caf50; stroke-width:2.5px; }

/* ── Tooltip (largeur augmentée, défilement autorisé) ───── */
.tooltip {
  position:fixed; background:#1e2030; color:#e5e7eb;
  padding:10px 14px; border-radius:10px; font-size:12px;
  pointer-events:none; display:none; z-index:9999;
  max-width:600px;         /* ← largeur confortable pour les matrices */
  box-shadow:0 8px 24px rgba(0,0,0,.5);
  border:1px solid #3b3f53;
  overflow-x:auto;         /* ← barre de défilement si contenu plus large */
}
.tooltip b   { color:#d4a373; font-family:'Consolas',monospace; }
.tooltip pre { margin:5px 0 0; font-family:'Consolas',monospace; font-size:10px;
               white-space:pre; color:#d1d5db;
               overflow-x:auto; max-width:100%; }
.tooltip-left, .tooltip-right {
  position:fixed;
  background:#1e2030;
  color:#e5e7eb;
  padding:12px 16px;
  border-radius:12px;
  font-size:12px;
  pointer-events:none;
  display:none;
  z-index:9999;
  max-width:600px;
  max-height:70vh;
  overflow:auto;
  box-shadow:0 8px 24px rgba(0,0,0,.5);
  border:1px solid #3b3f53;
  font-family: 'Consolas', monospace;
}
.tooltip-left {
  left: auto;
  right: calc(100% + 14px);
}
.tooltip-right {
  left: calc(100% + 14px);
  right: auto;
}
.tooltip-left b, .tooltip-right b { color:#d4a373; }
.tooltip-left pre, .tooltip-right pre {
  margin:6px 0 0;
  font-size:10px;
  white-space:pre;
  color:#d1d5db;
}

/* ── Zoom bar ────────────────────────────────────────────── */
.zoom-bar {
  position:absolute; bottom:14px; right:14px;
  display:flex; flex-direction:column; gap:4px;
}
.zoom-btn {
  width:32px; height:32px; background:#1e2030; border:1px solid #3b3f53;
  border-radius:7px; font-size:16px; cursor:pointer; color:#9ca3af;
  display:flex; align-items:center; justify-content:center;
  transition:background .1s;
}
.zoom-btn:hover { background:#2d3143; color:#e5e7eb; }

/* ── Epoch info badge ────────────────────────────────────── */
#epoch-badge {
  position:absolute; top:12px; left:12px;
  background:#1e2030; border:1px solid #3b3f53;
  border-radius:8px; padding:8px 14px; font-size:0.8rem;
  pointer-events:none;
}
#epoch-badge .ep  { color:#d4a373; font-weight:700; font-size:1rem; }
#epoch-badge .los { color:#6a9b6a; font-size:0.75rem; }
</style>
</head>
<body>

<!-- ── Top bar ────────────────────────────────────────────── -->
<div id="topbar">
  <h1>$title</h1>
  <div class="epoch-selector">
    <button onclick="changeEpoch(-1)">◀</button>
    <input type="range" id="epochSlider" min="0" max="0" value="0" oninput="onSliderChange()">
    <input type="number" id="epochInput" value="1" min="1" max="1" onchange="onEpochInputChange()">
    <button onclick="changeEpoch(1)">▶</button>
  </div>
  <div class="step-controls">
    <button class="btn" onclick="stepSnap(-1)">◀ Étape</button>
    <button class="btn primary" id="playBtn" onclick="togglePlay()">▶ Play</button>
    <button class="btn" onclick="stepSnap(1)">Étape ▶</button>
    <span id="stepStatus">–</span>
  </div>
</div>

<!-- ── Main ───────────────────────────────────────────────── -->
<div id="main">
  <!-- Sidebar -->
  <div id="sidebar">
    <div class="resize-handle" id="resizeHandle"></div>
    <h3>Courbe de perte</h3>
    <div id="loss-panel" style="height:160px;">
      <div class="resize-v-handle" id="resizeVHandle"></div>
      <div id="loss-chart-wrap"><canvas id="lossChart"></canvas></div>
    </div>
    <h3 style="margin-top:4px">Log d'exécution</h3>
    <div id="log-panel"></div>
  </div>

  <!-- Graph canvas -->
  <div id="canvas-wrap">
    <svg id="svg-canvas"></svg>
    <div id="epoch-badge">
      <div class="ep" id="badge-epoch">–</div>
      <div class="los" id="badge-loss">–</div>
    </div>
    <div class="zoom-bar">
      <button class="zoom-btn" onclick="zoomIn()">+</button>
      <button class="zoom-btn" onclick="zoomOut()">−</button>
      <button class="zoom-btn" onclick="zoomFit()">⊡</button>
    </div>
    <div id="tooltip-left" class="tooltip-left"></div>
    <div id="tooltip-right" class="tooltip-right"></div>
  </div>
</div>

<script>
// ── Données Julia ─────────────────────────────────────────────────────────────
const NODES_RAW   = $nodes_json;
const EDGES       = $edges_json;
const SNAPSHOTS   = $snaps_json;
const LOSSES      = $losses_json;
const INIT_VALS   = $init_json;
const FULL_VALS   = $full_json;
const FORMULAS    = $formulas_json;
const EPOCHS      = $epochs_json;



// ── Layout Dagre ──────────────────────────────────────────────────────────────
const NW = 185, NH = 72;
const dg = new dagre.graphlib.Graph({ multigraph:false });
dg.setGraph({ rankdir:'LR', nodesep:65, ranksep:110,
              marginx:55, marginy:55, acyclicer:'greedy',
              ranker:'network-simplex' });
dg.setDefaultEdgeLabel(() => ({}));
NODES_RAW.forEach(n => dg.setNode(n.id, { width:NW, height:NH }));
EDGES.forEach(([s,d]) => dg.setEdge(s, d));
dagre.layout(dg);
const NODES = NODES_RAW.map(n => {
  const dn = dg.node(n.id);
  return { ...n, x:Math.round(dn.x-NW/2), y:Math.round(dn.y-NH/2), w:NW, h:NH };
});
const NMAP = Object.fromEntries(NODES.map(n => [n.id, n]));
const gi = dg.graph();
const SW = Math.round(gi.width  || 800) + 110;
const SH = Math.round(gi.height || 600) + 110;
const svgEl = document.getElementById('svg-canvas');
svgEl.setAttribute('width', SW); svgEl.setAttribute('height', SH);

// ── Marqueurs ─────────────────────────────────────────────────────────────────
const SVGNS = 'http://www.w3.org/2000/svg';
const defs  = document.createElementNS(SVGNS, 'defs');
[['arr-default','#3b3f53'],['arr-fwd','#7aa2f7'],
 ['arr-bwd','#e06c75'],['arr-final','#4caf50']].forEach(([id,color]) => {
  const m = document.createElementNS(SVGNS,'marker');
  m.setAttribute('id',id); m.setAttribute('markerWidth','9');
  m.setAttribute('markerHeight','7'); m.setAttribute('refX','9');
  m.setAttribute('refY','3.5'); m.setAttribute('orient','auto');
  const p = document.createElementNS(SVGNS,'polygon');
  p.setAttribute('points','0 0, 9 3.5, 0 7'); p.setAttribute('fill',color);
  m.appendChild(p); defs.appendChild(m);
});
svgEl.appendChild(defs);

// ── Edge path ────────────────────────────────────────────────────────────────
function edgePts(src, dst) {
  const ed = dg.edge(src, dst);
  if (ed && ed.points && ed.points.length >= 2) return ed.points;
  const sn = NMAP[src], dn = NMAP[dst];
  if (!sn || !dn) return [];
  return [{ x: sn.x+sn.w, y: sn.y+sn.h/2 },
          { x: dn.x,       y: dn.y+dn.h/2 }];
}
function polylinePath(pts) {
  if (!pts || pts.length < 2) return '';
  let d = \`M \${pts[0].x} \${pts[0].y}\`;
  for (let i=1; i<pts.length; i++) d += \` L \${pts[i].x} \${pts[i].y}\`;
  return d;
}
function trunc(s, max=26) { return s.length>max ? s.slice(0,max-1)+'…' : s; }

// ── État ─────────────────────────────────────────────────────────────────────
const epochSnapshots = {};
EPOCHS.forEach(e => {
  epochSnapshots[e] = SNAPSHOTS.filter(s => s.epoch == e);
});

let currentEpochIdx = 0;
let currentSnapIdxInEpoch = 0;
let stepIdx = -1;
let playing = false;
let timer = null;
let hoveredNodeId = null;
const nodeVals = Object.fromEntries(NODES.map(n=>[n.id,{fwd:'',bwd:''}]));
const tooltipLeft = document.getElementById('tooltip-left');
const tooltipRight = document.getElementById('tooltip-right');

const slider = document.getElementById('epochSlider');
const epochInput = document.getElementById('epochInput');
slider.max = EPOCHS.length - 1;
slider.value = 0;
epochInput.min = EPOCHS[0];
epochInput.max = EPOCHS[EPOCHS.length-1];
epochInput.value = EPOCHS[0];

function changeEpoch(dir) {
  currentEpochIdx = Math.max(0, Math.min(EPOCHS.length-1, currentEpochIdx + dir));
  slider.value = currentEpochIdx;
  epochInput.value = EPOCHS[currentEpochIdx];
  currentSnapIdxInEpoch = epochSnapshots[EPOCHS[currentEpochIdx]].length - 1;
  stepIdx = -1;
  updateLossMarker(currentEpochIdx);
  renderStep();
}

function onSliderChange() {
  currentEpochIdx = parseInt(slider.value);
  epochInput.value = EPOCHS[currentEpochIdx];
  currentSnapIdxInEpoch = epochSnapshots[EPOCHS[currentEpochIdx]].length - 1;
  stepIdx = -1;
  updateLossMarker(currentEpochIdx);
  renderStep();
}

function onEpochInputChange() {
  let val = parseInt(epochInput.value);
  if (isNaN(val)) return;
  let idx = EPOCHS.indexOf(val);
  if (idx < 0) {
    idx = 0;
    let minDiff = Math.abs(EPOCHS[0] - val);
    for (let i=1; i<EPOCHS.length; i++) {
      const diff = Math.abs(EPOCHS[i] - val);
      if (diff < minDiff) { minDiff = diff; idx = i; }
    }
  }
  currentEpochIdx = idx;
  slider.value = idx;
  epochInput.value = EPOCHS[idx];
  currentSnapIdxInEpoch = epochSnapshots[EPOCHS[idx]].length - 1;
  stepIdx = -1;
  updateLossMarker(idx);
  renderStep();
}

// ── Loss chart avec grille ────────────────────────────────────────────────
let lossChart = null;
function buildLossChart() {
  if (!LOSSES.length) return;
  const ctx2 = document.getElementById('lossChart').getContext('2d');
  lossChart = new Chart(ctx2, {
    type:'line',
    data:{
      labels: LOSSES.map((_,i)=>i+1),
      datasets:[
        { label:'Loss', data:LOSSES, borderColor:'#d4a373', backgroundColor:'rgba(212,163,115,0.1)',
          borderWidth:2, pointRadius:0, tension:0.3, fill:true },
        { label:'Snapshot', data:EPOCHS.map(e => ({x:e, y:LOSSES[e-1]})),
          borderColor:'transparent', backgroundColor:'#e06c75',
          pointRadius:5, pointHoverRadius:7, type:'scatter',
          showLine:false }
      ]
    },
    options:{
      responsive:true, maintainAspectRatio:false, animation:false,
      plugins:{ legend:{display:false} },
      scales:{
        x:{ ticks:{color:'#6b7280',font:{size:9},maxTicksLimit:8},
            grid:{color:'#2d3143', drawOnChartArea:true} },
        y:{ ticks:{color:'#6b7280',font:{size:9}},
            grid:{color:'#2d3143', drawOnChartArea:true} }
      }
    }
  });
}

function updateLossMarker(idx) {
  if (!lossChart) return;
  lossChart.data.datasets[1].pointBackgroundColor =
    EPOCHS.map((_,j) => j===idx ? '#e06c75' : '#4b5268');
  lossChart.update();
}

// ── SVG init ─────────────────────────────────────────────────────────────────
function buildSVG() {
  const eLayer = document.createElementNS(SVGNS,'g'); eLayer.id='edge-layer';
  EDGES.forEach(([s,d]) => {
    const path = document.createElementNS(SVGNS,'path');
    path.setAttribute('d', polylinePath(edgePts(s,d)));
    path.setAttribute('class','edge edge-default');
    path.setAttribute('id',\`edge-\${s}-\${d}\`);
    path.setAttribute('marker-end','url(#arr-default)');
    eLayer.appendChild(path);
  });
  svgEl.appendChild(eLayer);
  const nLayer = document.createElementNS(SVGNS,'g'); nLayer.id='node-layer';
  NODES.forEach(n => {
    const g = document.createElementNS(SVGNS,'g');
    const rect = document.createElementNS(SVGNS,'rect');
    rect.setAttribute('x',n.x); rect.setAttribute('y',n.y);
    rect.setAttribute('width',n.w); rect.setAttribute('height',n.h);
    rect.setAttribute('rx','8');
    rect.setAttribute('id',\`node-\${n.id}\`);
    rect.setAttribute('class','node-body '+(n.is_param?'node-param':'node-default'));
    // Remplace ton ancien bloc rect.addEventListener par ceci :
    rect.addEventListener('mouseenter', e => {
        hoveredNodeId = n.id;
        updateTooltips(n.id, e.target);
    });

    rect.addEventListener('mouseleave', () => {
        hoveredNodeId = null;
        tooltipLeft.style.display = 'none';
        tooltipRight.style.display = 'none';
    });
    
    const sep=document.createElementNS(SVGNS,'line');
    sep.setAttribute('x1',n.x+10); sep.setAttribute('x2',n.x+n.w-10);
    sep.setAttribute('y1',n.y+28); sep.setAttribute('y2',n.y+28);
    sep.setAttribute('stroke','#3b3f53'); sep.setAttribute('stroke-width','1');
    sep.setAttribute('pointer-events','none');
    const mkT=(id,dy,cls)=>{
      const t=document.createElementNS(SVGNS,'text');
      t.setAttribute('x',n.x+n.w/2); t.setAttribute('y',n.y+dy);
      t.setAttribute('text-anchor','middle'); t.setAttribute('class',cls);
      if(id) t.setAttribute('id',id); return t;
    };
    const tL=mkT(null,20,'node-label'); tL.textContent=trunc(FORMULAS[n.id]||n.id);
    const tV=mkT(\`val-\${n.id}\`,44,'node-val');  tV.textContent='';
    const tG=mkT(\`grad-\${n.id}\`,60,'node-grad'); tG.textContent='';
    g.append(rect,sep,tL,tV,tG); nLayer.appendChild(g);
  });
  svgEl.appendChild(nLayer);
}

function setNC(id, cls) {
  const r = document.getElementById(\`node-\${id}\`);
  if (r) r.setAttribute('class',\`node-body \${cls}\`);
}
function setEC(s,d,cls,arr) {
  const e = document.getElementById(\`edge-\${s}-\${d}\`);
  if (e) { e.setAttribute('class',\`edge \${cls}\`);
           e.setAttribute('marker-end',\`url(#\${arr})\`); }
}

function updateTooltips(nodeId, rectElement) {
    const v = nodeVals[nodeId];
    const n = NMAP[nodeId];
    const r = rectElement.getBoundingClientRect();

    // Tooltip gauche (inchangé)
    if (n.is_param && currentParamFull[nodeId]) {
        tooltipLeft.innerHTML = "<b>📦 Poids (" + nodeId + ")</b><pre>" + currentParamFull[nodeId] + "</pre>";
        tooltipLeft.style.display = 'block';
        tooltipLeft.style.right = (window.innerWidth - r.left + 14) + 'px';
        tooltipLeft.style.top = Math.max(8, r.top - 8) + 'px';
    } else {
        tooltipLeft.style.display = 'none';
    }

    // Tooltip droit
    let rightHtml = "";
    if (n.is_param) {
        // Pour les paramètres : afficher ||nom|| avec la valeur backward (norme)
        rightHtml = "<b>||" + nodeId + "||</b><pre>" + (v.bwd || '?') + "</pre>";
    } else if (v.fwd) {
        rightHtml = "<b>➡️ Forward (" + nodeId + ")</b><pre>" + v.fwd + "</pre>";
        if (v.bwd) {
            rightHtml += "<b>🔻 Gradient (" + nodeId + ")</b><pre>" + v.bwd + "</pre>";
        }
    } else {
        rightHtml = "<b>➡️ Forward (" + nodeId + ")</b><pre>?</pre>";
    }
    tooltipRight.innerHTML = rightHtml;
    tooltipRight.style.display = 'block';
    tooltipRight.style.left = (r.right + 14) + 'px';
    tooltipRight.style.top = Math.max(8, r.top - 8) + 'px';
}


function renderStep() {
  const epochVal = EPOCHS[currentEpochIdx];
  const snap = epochSnapshots[epochVal][currentSnapIdxInEpoch];
  if (!snap) return;
  currentParamValues = snap.params || {};
  currentParamFull = snap.params_full || {};


// Afficher les versions courtes des poids sur les nœuds
for (let id in currentParamValues) {
    const elem = document.getElementById("val-" + id);
    if (elem) elem.textContent = currentParamValues[id];
}
  document.getElementById('badge-epoch').textContent = \`Epoch \${snap.epoch}  |  iter \${snap.iter}\`;
  document.getElementById('badge-loss').textContent = \`Loss : \${snap.loss.toFixed(6)}\`;
  const total = snap.events.length;
  document.getElementById('stepStatus').textContent =
    stepIdx < 0 ? \`0 / \${total}\` : \`\${stepIdx+1} / \${total}\`;

  NODES.forEach(n => {
    setNC(n.id, n.is_param ? 'node-param' : 'node-default');
    const vt = document.getElementById(\`val-\${n.id}\`);
    const gt = document.getElementById(\`grad-\${n.id}\`);
    if (vt) vt.textContent = n.is_leaf && INIT_VALS[n.id] !== '?' ? INIT_VALS[n.id] : '';
    if (gt) gt.textContent = '';
    });
  EDGES.forEach(([s,d]) => setEC(s,d,'edge-default','arr-default'));

  if (stepIdx < 0) { document.getElementById('log-panel').innerHTML=''; return; }

  let lastFwd = null;
  for (let i = 0; i <= stepIdx; i++) {
    const ev = snap.events[i];
    if (ev.status === 'starting') {
      setNC(ev.node, ev.phase === 'forward' ? 'node-fwd' : 'node-bwd');
    } else {
      if (ev.phase === 'forward') {
        setNC(ev.node, 'node-default');
        lastFwd = ev.node;
        // Ne change plus le texte dans le nœud
        nodeVals[ev.node].fwd = ev.val || '';
      } else { // phase === 'backward'
        setNC(ev.node, 'node-done');
        // Pas de mise à jour du texte du nœud non plus
        nodeVals[ev.node].bwd = ev.val || '';
      }
    }
    EDGES.forEach(([s, d]) => {
      if (d === ev.node) {
        const [ec, ea] = ev.phase === 'forward'
          ? ['edge-fwd', 'arr-fwd']
          : ['edge-bwd', 'arr-bwd'];
        setEC(s, d, ec, ea);
      }
    });
  }
  if (lastFwd) {
    setNC(lastFwd, 'node-final');
    EDGES.forEach(([s, d]) => { if (d === lastFwd) setEC(s, d, 'edge-final', 'arr-final'); });
  }
  document.getElementById('log-panel').innerHTML =
    snap.events.slice(0,stepIdx+1).slice().reverse().map(ev =>
      \`<div class="log-entry log-\${ev.phase}">
        <b>\${ev.phase.toUpperCase()}</b> \${ev.node}
        \${ev.status==='starting' ? '→ …' : ': '+(ev.val||'')}
      </div>\`
    ).join('');

    if (hoveredNodeId) {
        const rectEl = document.getElementById(\`node-\${hoveredNodeId}\`);
        if (rectEl) updateTooltips(hoveredNodeId, rectEl);
    }
}

function stepSnap(dir) {
  const epochVal = EPOCHS[currentEpochIdx];
  const snap = epochSnapshots[epochVal][currentSnapIdxInEpoch];
  if (!snap) return;
  const total = snap.events.length;
  stepIdx = Math.max(-1, Math.min(total-1, stepIdx+dir));
  renderStep();
}

function togglePlay() {
  playing = !playing;
  const btn = document.getElementById('playBtn');
  btn.textContent = playing ? '⏸ Pause' : '▶ Play';
  btn.classList.toggle('primary', !playing);
  if (playing) {
    timer = setInterval(() => {
      const epochVal = EPOCHS[currentEpochIdx];
      const snap = epochSnapshots[epochVal][currentSnapIdxInEpoch];
      if (!snap) return;
      const total = snap.events.length;
      if (stepIdx < total-1) {
        stepIdx++;
        renderStep();
      } else {
        if (currentSnapIdxInEpoch < epochSnapshots[epochVal].length-1) {
          currentSnapIdxInEpoch++;
          stepIdx = -1;
          renderStep();
        } else if (currentEpochIdx < EPOCHS.length-1) {
          currentEpochIdx++;
          slider.value = currentEpochIdx;
          epochInput.value = EPOCHS[currentEpochIdx];
          currentSnapIdxInEpoch = 0;
          stepIdx = -1;
          updateLossMarker(currentEpochIdx);
          renderStep();
        } else {
          playing = false; clearInterval(timer);
          btn.textContent = '▶ Play'; btn.classList.add('primary');
        }
      }
    }, 300);
  } else { clearInterval(timer); }
}

// ── Redimensionnement sidebar (largeur) ────────────────────────────────────
const sidebar = document.getElementById('sidebar');
const handleW = document.getElementById('resizeHandle');
let isResizingW = false, startX, startWidth;
handleW.addEventListener('mousedown', e => {
  isResizingW = true;
  startX = e.clientX;
  startWidth = parseInt(getComputedStyle(sidebar).width, 10);
  e.preventDefault();
});
document.addEventListener('mousemove', e => {
  if (!isResizingW) return;
  const newWidth = startWidth + e.clientX - startX;
  sidebar.style.width = Math.min(600, Math.max(200, newWidth)) + 'px';
});
document.addEventListener('mouseup', () => { isResizingW = false; });

// ── Redimensionnement hauteur courbe de perte ──────────────────────────────
const lossPanel = document.getElementById('loss-panel');
const handleH = document.getElementById('resizeVHandle');
let isResizingH = false, startY, startHeight;
handleH.addEventListener('mousedown', e => {
  isResizingH = true;
  startY = e.clientY;
  startHeight = parseInt(getComputedStyle(lossPanel).height, 10);
  e.preventDefault();
});
document.addEventListener('mousemove', e => {
  if (!isResizingH) return;
  const newHeight = startHeight + e.clientY - startY;
  lossPanel.style.height = Math.min(400, Math.max(80, newHeight)) + 'px';
  if (lossChart) lossChart.resize();
});
document.addEventListener('mouseup', () => { isResizingH = false; });

// ── Pan & Zoom ─────────────────────────────────────────────────────────────
let sc=1,tx=0,ty=0,pan=false,px=0,py=0;
const wrap=document.getElementById('canvas-wrap');
function applyT(){ svgEl.style.transform=\`translate(\${tx}px,\${ty}px) scale(\${sc})\`; }
function zoomIn(){ sc*=1.2; applyT(); }
function zoomOut(){ sc/=1.2; applyT(); }
function zoomFit(){
  const ww=wrap.clientWidth,wh=wrap.clientHeight;
  sc=Math.min(ww/SW,wh/SH)*.9;
  tx=(ww-SW*sc)/2; ty=(wh-SH*sc)/2; applyT();
}
wrap.addEventListener('wheel',e=>{
  e.preventDefault();
  const r=wrap.getBoundingClientRect();
  const mx=e.clientX-r.left,my=e.clientY-r.top;
  const f=e.deltaY>0?.9:1.1, ns2=sc*f;
  tx=mx-(mx-tx)*(ns2/sc); ty=my-(my-ty)*(ns2/sc); sc=ns2; applyT();
},{passive:false});
wrap.addEventListener('mousedown',e=>{
  if(e.button===0){pan=true;px=e.clientX-tx;py=e.clientY-ty;
    svgEl.style.cursor='grabbing';e.preventDefault();}
});
window.addEventListener('mousemove',e=>{
  if(pan){tx=e.clientX-px;ty=e.clientY-py;applyT();}
});
window.addEventListener('mouseup',()=>{pan=false;svgEl.style.cursor='';});

// ── Init ────────────────────────────────────────────────────────────────────
buildSVG();
buildLossChart();
updateLossMarker(0);
renderStep();
requestAnimationFrame(zoomFit);
</script>
</body>
</html>
"""
    write(filepath, html)
    println("✅ Viewer animé exporté → $filepath  ($(length(snapshots)) snapshots)")
end