using JSON, Printf

# ═══════════════════════════════════════════════════════════
# Types pour les snapshots d'entraînement (inchangés)
# ═══════════════════════════════════════════════════════════
mutable struct TrainingSnapshot
    epoch  :: Int
    iter   :: Int
    loss   :: Float32
    log    :: ExecutionLog
    params :: Dict{Symbol, AbstractArray{Float32}}
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

# ═══════════════════════════════════════════════════════════
# Utilitaires de formatage (inchangés)
# ═══════════════════════════════════════════════════════════
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

# ═══════════════════════════════════════════════════════════
# Formules avec indices Unicode + Pascal
# ═══════════════════════════════════════════════════════════
function node_formula_text(graph::NeuroGraph, sym::Symbol, ns::Symbol)
    rules_dict = get(graph.rules, ns, Dict())
    if !haskey(rules_dict, sym)
        return string(sym)   # feuille
    end
    rule = rules_dict[sym]
    op   = rule.op
    inputs = rule.inputs
    attrs = rule.attrs

    # -------- Cas spécial : formule de Pascal ----------
    if (op == :add || op == :wsum) && haskey(attrs, :n) && haskey(attrs, :k)
        n, k = Int(attrs[:n]), Int(attrs[:k])
        return "C_{$n,$k} = C_{$(n-1),$(k-1)} + C_{$(n-1),$k}"
    end

    # -------- Générique (indices Unicode) --------------
    function fmt(s)
        s_str = string(s)
        return replace(s_str, r"(\d+)" => m -> join([Char(0x2080 + parse(Int, d)) for d in m.match], ""))
    end

    if op == :add
        return "$(fmt(sym)) = $(fmt(inputs[1])) + $(fmt(inputs[2]))"
    elseif op == :wsum
        ws = get(attrs, :weights, Float32[])
        parts = String[]
        for (i, inp) in enumerate(inputs)
            coeff = i <= length(ws) ? @sprintf("%.2f", ws[i]) : "?"
            push!(parts, "$coeff·$(fmt(inp))")
        end
        return "$(fmt(sym)) = " * join(parts, " + ")
    elseif op == :nsum
        inps = join([fmt(i) for i in inputs], " + ")
        return "$(fmt(sym)) = Σ($inps)"
    elseif op == :identity
        return "$(fmt(sym)) = $(fmt(inputs[1]))"
    else
        return "$(fmt(sym)) = $op(…)"
    end
end

# ═══════════════════════════════════════════════════════════
# save_interactive_graph (version simple, sans KaTeX)
# ═══════════════════════════════════════════════════════════
function save_interactive_graph(graph::NeuroGraph, log::ExecutionLog,
                                filepath::String; title="NeuroDSL Trace")
    ns       = graph.active_ns
    rules_ns = get(graph.rules, ns, Dict{Symbol,Any}())

    log_json = JSON.json([Dict(
        :node   => e[:node],
        :phase  => e[:phase],
        :status => e[:status],
        :val    => e[:value]
    ) for e in log.events])

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

    edges = [[string(inp), string(out)] for (out, rule) in rules_ns for inp in rule.inputs]

    nodes_json    = JSON.json([Dict(:id => string(s),
                                   :is_param => is_param_d[string(s)],
                                   :is_leaf  => is_leaf_d[string(s)])
                               for s in keys(graph.nodes[ns])])
    edges_json    = JSON.json(edges)
    init_json     = JSON.json(init_vals)
    full_json     = JSON.json(full_vals)
    formulas_json = JSON.json(formulas)

    html = """
<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<title>$title</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/dagre/0.8.5/dagre.min.js"></script>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f0f2f5; display: flex; height: 100vh; overflow: hidden; }
#sidebar { width: 290px; min-width: 290px; background: #fff; border-right: 1px solid #e2e8f0; display: flex; flex-direction: column; padding: 16px; gap: 10px; z-index: 10; box-shadow: 2px 0 10px rgba(0,0,0,0.07); }
#sidebar h2 { font-size: 13px; color: #1e293b; font-weight: 700; line-height: 1.4; }
.controls   { display: flex; gap: 6px; }
.btn { flex: 1; padding: 7px 4px; border-radius: 7px; border: 1px solid #e2e8f0; cursor: pointer; background: #fff; font-size: 12px; font-weight: 500; color: #334155; transition: all .15s; }
.btn:hover { background: #f1f5f9; border-color: #94a3b8; }
#playBtn   { background: #3b82f6; color: #fff; border-color: #2563eb; }
#playBtn:hover { background: #2563eb; }
#status { font-size: 11px; color: #94a3b8; text-align: center; }
#log-panel { flex: 1; overflow-y: auto; font-family: 'Consolas', monospace; font-size: 10px; border: 1px solid #e2e8f0; padding: 8px; background: #fafafa; border-radius: 8px; }
.log-entry { padding: 3px 6px; border-radius: 4px; margin-bottom: 2px; line-height: 1.5; }
.log-fwd { background: #dbeafe; color: #1d4ed8; }
.log-bwd { background: #fee2e2; color: #b91c1c; }
#canvas-wrap { flex: 1; position: relative; overflow: hidden; background: #f8fafc; background-image: radial-gradient(circle, #d1d5db 1px, transparent 1px); background-size: 24px 24px; }
#svg-canvas { position: absolute; top: 0; left: 0; transform-origin: 0 0; overflow: visible; }
.node-body { transition: all .2s ease; cursor: pointer; filter: drop-shadow(0 1px 4px rgba(0,0,0,.08)); }
.node-default { fill: #fff;    stroke: #cbd5e0; stroke-width: 1.5px; }
.node-param   { fill: #f1f5f9; stroke: #94a3b8; stroke-width: 1.5px; stroke-dasharray: 5,3; }
.node-fwd     { fill: #dbeafe; stroke: #3b82f6; stroke-width: 2.5px; }
.node-bwd     { fill: #fee2e2; stroke: #ef4444; stroke-width: 2.5px; }
.node-done    { fill: #dcfce7; stroke: #22c55e; stroke-width: 2px;   }
.node-final   { fill: #bbf7d0; stroke: #16a34a; stroke-width: 2.5px; filter: drop-shadow(0 2px 8px rgba(22,163,74,.4)); }
.node-label { font-family: 'Segoe UI', system-ui, sans-serif; font-size: 10px; fill: #1e293b; pointer-events: none; }
.node-val   { font-family: 'Consolas', monospace; font-size: 9.5px;  fill: #64748b;  pointer-events: none; }
.node-grad  { font-family: 'Consolas', monospace; font-size: 9.5px;  fill: #ef4444;  pointer-events: none; }
.edge         { fill: none; stroke-width: 1.5px; transition: all .2s ease; }
.edge-default { stroke: #94a3b8; stroke-opacity: .5; }
.edge-fwd     { stroke: #3b82f6; stroke-width: 2.5px; stroke-opacity: 1; }
.edge-bwd     { stroke: #ef4444; stroke-width: 2.5px; stroke-opacity: 1; }
.edge-final   { stroke: #16a34a; stroke-width: 2.5px; stroke-opacity: 1; }
.tooltip { position: fixed; background: #1e293b; color: #f1f5f9; padding: 10px 14px; border-radius: 10px; font-size: 12px; pointer-events: none; display: none; z-index: 9999; max-width: 380px; box-shadow: 0 8px 24px rgba(0,0,0,.35); }
.tooltip b   { color: #7dd3fc; font-family: 'Consolas', monospace; }
.tooltip pre { margin:5px 0 0; font-family:'Consolas',monospace; font-size:10px; white-space:pre; color:#e2e8f0; overflow-x:auto; max-width:100%; }
.tooltip-left, .tooltip-right { position:fixed; background:#1e2030; color:#e5e7eb; padding:12px 16px; border-radius:12px; font-size: inherit; pointer-events:none; display:none; z-index:9999; max-width:600px; max-height:70vh; overflow:auto; box-shadow:0 8px 24px rgba(0,0,0,.5); border:1px solid #3b3f53; font-family: 'Consolas', monospace; }
.tooltip-left { left: auto; right: calc(100% + 14px); }
.tooltip-right { left: calc(100% + 14px); right: auto; }
.tooltip-left b, .tooltip-right b { color:#d4a373; }
.tooltip-left pre, .tooltip-right pre { margin:6px 0 0; font-size: inherit; white-space:pre; color:#d1d5db; }
.zoom-bar { position: absolute; bottom: 14px; right: 14px; display: flex; flex-direction: column; gap: 4px; }
.zoom-btn { width: 34px; height: 34px; background: #fff; border: 1px solid #e2e8f0; border-radius: 8px; font-size: 17px; cursor: pointer; display: flex; align-items: center; justify-content: center; box-shadow: 0 2px 6px rgba(0,0,0,.08); transition: background .1s; color: #334155; }
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
const NODES_RAW = $nodes_json;
const EDGES     = $edges_json;
const LOG       = $log_json;
const INIT_VALS = $init_json;
const FULL_VALS = $full_json;
const FORMULAS  = $formulas_json;

const NW = 185, NH = 72;
const dg = new dagre.graphlib.Graph({ multigraph: false });
dg.setGraph({
  rankdir  : 'LR',
  nodesep  : 65,
  ranksep  : 110,
  marginx  : 55,
  marginy  : 55,
  edgesep  : 25,
  acyclicer: 'greedy',
  ranker   : 'network-simplex'
});
dg.setDefaultEdgeLabel(() => ({}));
NODES_RAW.forEach(n => dg.setNode(n.id, { width: NW, height: NH }));
EDGES.forEach(([s, d]) => dg.setEdge(s, d));
dagre.layout(dg);
const NODES = NODES_RAW.map(n => {
  const dn = dg.node(n.id);
  return { ...n, x: Math.round(dn.x - NW/2), y: Math.round(dn.y - NH/2), w: NW, h: NH };
});
const NMAP = Object.fromEntries(NODES.map(n => [n.id, n]));
const gi = dg.graph();
const SW = Math.round(gi.width  || 800) + 110;
const SH = Math.round(gi.height || 600) + 110;
const svgEl = document.getElementById('svg-canvas');
svgEl.setAttribute('width',  SW);
svgEl.setAttribute('height', SH);

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

function edgePts(src, dst) {
  const ed = dg.edge(src, dst);
  if (ed && ed.points && ed.points.length >= 2) return ed.points;
  const sn = NMAP[src], dn = NMAP[dst];
  if (!sn || !dn) return [];
  const sx = sn.x + sn.w/2, sy = sn.y + sn.h/2;
  const dx = dn.x + dn.w/2, dy = dn.y + dn.h/2;
  const intersect = (rx, ry, rw, rh, x1, y1, x2, y2) => {
    const left = rx, right = rx + rw, top = ry, bottom = ry + rh;
    const pts = [];
    if (y1 !== y2) { const t = (top - y1) / (y2 - y1); if (t >= 0 && t <= 1) { const xi = x1 + t * (x2 - x1); if (xi >= left && xi <= right) pts.push({x: xi, y: top}); } }
    if (y1 !== y2) { const t = (bottom - y1) / (y2 - y1); if (t >= 0 && t <= 1) { const xi = x1 + t * (x2 - x1); if (xi >= left && xi <= right) pts.push({x: xi, y: bottom}); } }
    if (x1 !== x2) { const t = (left - x1) / (x2 - x1); if (t >= 0 && t <= 1) { const yi = y1 + t * (y2 - y1); if (yi >= top && yi <= bottom) pts.push({x: left, y: yi}); } }
    if (x1 !== x2) { const t = (right - x1) / (x2 - x1); if (t >= 0 && t <= 1) { const yi = y1 + t * (y2 - y1); if (yi >= top && yi <= bottom) pts.push({x: right, y: yi}); } }
    if (pts.length === 0) return {x: x1, y: y1};
    pts.sort((a,b) => (a.x-x1)**2 + (a.y-y1)**2 - (b.x-x1)**2 + (b.y-y1)**2);
    return pts[0];
  };
  const start = intersect(sn.x, sn.y, sn.w, sn.h, sx, sy, dx, dy);
  const end   = intersect(dn.x, dn.y, dn.w, dn.h, dx, dy, sx, sy);
  return [start, end];
}

function polylinePath(pts) {
  if (!pts || pts.length < 2) return '';
  if (pts.length === 2) {
    const dx = pts[1].x - pts[0].x;
    const dy = pts[1].y - pts[0].y;
    return `M \${pts[0].x} \${pts[0].y} Q \${pts[0].x + dx/2} \${pts[0].y + dy/2}, \${pts[1].x} \${pts[1].y}`;
  }
  let d = `M \${pts[0].x} \${pts[0].y}`;
  for (let i = 1; i < pts.length; i++) {
    d += ` L \${pts[i].x} \${pts[i].y}`;
  }
  return d;
}

const nodeVals = Object.fromEntries(
  NODES.map(n => [n.id, { fwd: FULL_VALS[n.id] || '?', bwd: '' }])
);
const tooltipLeft = document.getElementById('tooltip-left');
const tooltipRight = document.getElementById('tooltip-right');

let currentParamValues = {};
let currentParamFull = {};

function init() {
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

  const nLayer = document.createElementNS(SVGNS, 'g');
  nLayer.id = 'node-layer';

  NODES.forEach(n => {
    const g = document.createElementNS(SVGNS, 'g');

    const rect = document.createElementNS(SVGNS, 'rect');
    rect.setAttribute('x',      n.x);  rect.setAttribute('y',      n.y);
    rect.setAttribute('width',  n.w);  rect.setAttribute('height', n.h);
    rect.setAttribute('rx',     '8');
    rect.setAttribute('id',     `node-\${n.id}`);
    rect.setAttribute('class',  'node-body ' + (n.is_param ? 'node-param' : 'node-default'));

    rect.addEventListener('mouseenter', e => {
        const bbox = rect.getBBox();
        const ctm = svgEl.getScreenCTM();
        const tl = svgEl.createSVGPoint(); tl.x = bbox.x; tl.y = bbox.y;
        const br = svgEl.createSVGPoint(); br.x = bbox.x + bbox.width; br.y = bbox.y + bbox.height;
        const stl = tl.matrixTransform(ctm);
        const sbr = br.matrixTransform(ctm);
        const r = { left: stl.x, top: stl.y, right: sbr.x, bottom: sbr.y };

        const baseFontSize = 12 * sc;
        tooltipLeft.style.fontSize = Math.max(baseFontSize, 10) + 'px';
        tooltipRight.style.fontSize = Math.max(baseFontSize, 10) + 'px';
        const pad = Math.max(8, 12 * sc / 1.5);
        tooltipLeft.style.padding = pad + 'px';
        tooltipRight.style.padding = pad + 'px';

        const v = nodeVals[n.id];
        if (n.is_param && currentParamFull[n.id]) {
            tooltipLeft.innerHTML = "<b>📦 Poids (" + n.id + ")</b><pre>" + currentParamFull[n.id] + "</pre>";
            tooltipLeft.style.display = 'block';
            tooltipLeft.style.right = (window.innerWidth - r.left + 14) + 'px';
            tooltipLeft.style.top = Math.max(8, r.top - 8) + 'px';
        } else {
            tooltipLeft.style.display = 'none';
        }
        let rightHtml = "<b>➡️ Forward (" + n.id + ")</b><pre>" + (v.fwd || '?') + "</pre>";
        if (v.bwd) rightHtml += "<b>🔻 Gradient (" + n.id + ")</b><pre>" + v.bwd + "</pre>";
        tooltipRight.innerHTML = rightHtml;
        tooltipRight.style.display = 'block';
        tooltipRight.style.left = (r.right + 14) + 'px';
        tooltipRight.style.top = Math.max(8, r.top - 8) + 'px';
    });
    rect.addEventListener('mouseleave', () => {
        tooltipLeft.style.display = 'none';
        tooltipRight.style.display = 'none';
    });
    
    const sep = document.createElementNS(SVGNS, 'line');
    sep.setAttribute('x1', n.x + 10);      sep.setAttribute('x2', n.x + n.w - 10);
    sep.setAttribute('y1', n.y + 30);      sep.setAttribute('y2', n.y + 30);
    sep.setAttribute('stroke', '#e2e8f0'); sep.setAttribute('stroke-width', '1');
    sep.setAttribute('pointer-events', 'none');

    // Label en texte simple (indices Unicode)
    const tLabel = document.createElementNS(SVGNS, 'text');
    tLabel.setAttribute('x', n.x + n.w / 2);
    tLabel.setAttribute('y', n.y + 20);
    tLabel.setAttribute('text-anchor', 'middle');
    tLabel.setAttribute('class', 'node-label');
    tLabel.textContent = FORMULAS[n.id] || n.id;
    g.appendChild(tLabel);

    const tVal = document.createElementNS(SVGNS, 'text');
    tVal.setAttribute('x', n.x + n.w / 2); tVal.setAttribute('y', n.y + 44);
    tVal.setAttribute('text-anchor', 'middle'); tVal.setAttribute('class', 'node-val');
    tVal.setAttribute('id', `val-\${n.id}`);
    tVal.textContent = (n.is_leaf && INIT_VALS[n.id] !== '?') ? INIT_VALS[n.id] : '';
    g.appendChild(tVal);

    const tGrad = document.createElementNS(SVGNS, 'text');
    tGrad.setAttribute('x', n.x + n.w / 2); tGrad.setAttribute('y', n.y + 60);
    tGrad.setAttribute('text-anchor', 'middle'); tGrad.setAttribute('class', 'node-grad');
    tGrad.setAttribute('id', `grad-\${n.id}`);
    tGrad.textContent = '';
    g.appendChild(tGrad);

    nLayer.appendChild(g);
  });
  svgEl.appendChild(nLayer);
}

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

let step_i = -1, playing = false, timer = null;

function step(dir) {
  step_i = Math.max(-1, Math.min(LOG.length - 1, step_i + dir));
  updateUI();
}

function updateUI() {
  document.getElementById('status').textContent = `Step : \${step_i + 1} / \${LOG.length}`;

  NODES.forEach(n => {
    setNC(n.id, n.is_param ? 'node-param' : 'node-default');
    const vt = document.getElementById(`val-\${n.id}`);
    if (vt) vt.textContent = (n.is_leaf && INIT_VALS[n.id] !== '?') ? INIT_VALS[n.id] : '';
    const gt = document.getElementById(`grad-\${n.id}`);
    if (gt) gt.textContent = '';
  });
  EDGES.forEach(([s, d]) => setEC(s, d, 'edge-default', 'arr-default'));

  if (step_i === -1) { document.getElementById('log-panel').innerHTML = ''; return; }

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
    EDGES.forEach(([s, d]) => {
      if (d === lastFwd) setEC(s, d, 'edge-final', 'arr-final');
    });
  }

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

init();
requestAnimationFrame(zoomFit);
</script>
</body>
</html>
"""
    write(filepath, html)
    println("✅ Interactive Trace exporté → $filepath")
end