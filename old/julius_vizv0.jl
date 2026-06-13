# ══════════════════════════════════════════════════════════════════════════════
# NeuroViz — HTML Graph Visualizer for NeuroDSL  (v1.0)
#
# Usage :
#   include("neuro_viz.jl")
#   save_graph_html(mlp,  "mlp.html";  namespace=:mlp)
#   save_graph_html(attn, "attn.html"; namespace=:attn, open_browser=true)
# ══════════════════════════════════════════════════════════════════════════════

# ── Minimal JSON serializer (aucune dépendance externe) ────────────────────
_jv(v::AbstractString) = "\"$(replace(v, "\\" => "\\\\", "\"" => "\\\""))\""
_jv(v::Symbol)         = _jv(string(v))
_jv(v::Integer)        = string(v)
_jv(v::Bool)           = v ? "true" : "false"
_jv(v::Nothing)        = "null"
_jv(v::Vector)         = "[" * join(_jv.(v), ",") * "]"
function _jv(v::Dict)
    "{" * join([_jv(string(k)) * ":" * _jv(val) for (k, val) in v], ",") * "}"
end

# ── Graph → Dict ─────────────────────────────────────────────────────────────
function _jv_data(g::JuliusGraph; namespace=g.active_ns)
    nodes_raw = fix_get_all_nodes(g; namespace=namespace)
    edges_raw = fix_get_edges(g; namespace=namespace)
    layers    = fix_assign_layers(g; namespace=namespace)

    nodes = Dict{String,Any}[]
    for sym in sort(collect(nodes_raw), by=string)
        push!(nodes, Dict{String,Any}(
            "id"    => string(sym),
            "layer" => get(layers, sym, 0),
            "type"  => fix_node_kind(g, sym; namespace=namespace),
        ))
    end

    edges = Dict{String,Any}[]
    for (src, dst) in edges_raw
        push!(edges, Dict{String,Any}("source" => string(src), "target" => string(dst)))
    end

    n_params  = count(nd -> nd.is_param, values(g.nodes[namespace]))
    max_layer = maximum(values(layers); init=0)

    return Dict{String,Any}(
        "namespace" => string(namespace),
        "nodes"     => nodes,
        "edges"     => edges,
        "stats"     => Dict{String,Any}(
            "n_nodes"  => length(nodes),
            "n_edges"  => length(edges),
            "n_params" => n_params,
            "depth"    => max_layer + 1,
        ),
    )
end

# ── Template HTML (raw string → aucun échappement $ nécessaire) ───────────
const _JV_TMPL = raw"""<!DOCTYPE html>
<html lang="fr"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>JULIUS_TITLE</title>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.9.0/d3.min.js"></script>
<style>
:root{--bg:#070c14;--sf:#0e1827;--br:#1c2e4a;--tx:#d8e4f0;--mu:#4e6680}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--tx);font-family:'JetBrains Mono','Fira Code','Consolas',monospace;
     min-height:100vh;display:flex;flex-direction:column;padding:24px;gap:14px}
.sub{font-size:11px;color:var(--mu);letter-spacing:2px;text-transform:uppercase}
.sub::before{content:"◈  ";color:#f59e0b}
h1{font-size:17px;font-weight:700}
.card{background:var(--sf);border:1px solid var(--br);flex:1;display:flex;flex-direction:column;border-radius:4px}
.srow{display:flex;gap:22px;padding:9px 18px;border-bottom:1px solid var(--br);font-size:11px;color:var(--mu);flex-wrap:wrap}
.srow b{color:var(--tx);font-weight:700;margin-left:4px}
#sv{flex:1;overflow:hidden;min-height:360px}
svg{display:block}
.edge{fill:none;stroke:#253858;stroke-width:1.5;opacity:.55}
.nc{stroke-width:1.5;transition:opacity .12s;cursor:pointer}
.nc:hover{opacity:1 !important}
.np{fill:#FAEEDA;stroke:#854F0B}
.ni{fill:#E6F1FB;stroke:#185FA5}
.nco{fill:#EEEDFE;stroke:#534AB7}
.no{fill:#E1F5EE;stroke:#0F6E56}
.nt{font-size:10px;font-weight:700;text-anchor:middle;dominant-baseline:middle;pointer-events:none}
.tp{fill:#412402}
.ti{fill:#042C53}
.tco{fill:#26215C}
.to{fill:#04342C}
.gl{stroke:#1c2e4a;stroke-dasharray:2 6;opacity:.6}
.arw{fill:#253858}
.hi{padding:7px 18px;font-size:11px;color:var(--mu);min-height:30px;border-top:1px solid var(--br)}
.hi b{color:var(--tx)}
.leg{display:flex;gap:20px;padding:9px 18px;border-top:1px solid var(--br);font-size:11px;color:var(--mu);flex-wrap:wrap}
.li{display:flex;align-items:center;gap:6px}
.ld{width:9px;height:9px;border-radius:50%;border:1.5px solid}
</style></head><body>
<header>
  <div class="sub">NeuroDSL &nbsp;·&nbsp; JuliusViz</div>
  <h1>JULIUS_TITLE</h1>
</header>
<div class="card">
  <div class="srow" id="srow"></div>
  <div id="sv"></div>
  <div class="hi" id="hi">Hover over a node to inspect it</div>
  <div class="leg">
    <div class="li"><div class="ld" style="background:#FAEEDA;border-color:#854F0B"></div>param</div>
    <div class="li"><div class="ld" style="background:#E6F1FB;border-color:#185FA5"></div>input</div>
    <div class="li"><div class="ld" style="background:#EEEDFE;border-color:#534AB7"></div>computed</div>
    <div class="li"><div class="ld" style="background:#E1F5EE;border-color:#0F6E56"></div>output</div>
  </div>
</div>
<script>
const DATA = JULIUS_GRAPH_DATA;
const KC = {param:"nc np",input:"nc ni",computed:"nc nco",output:"nc no"};
const KT = {param:"nt tp",input:"nt ti",computed:"nt tco",output:"nt to"};

function ep(s, t) {
  const r=18, dx=t.x-s.x, dy=t.y-s.y, len=Math.hypot(dx,dy);
  if (len < 2*r+4) return "";
  const ux=dx/len, uy=dy/len;
  const x1=s.x+ux*(r+1), y1=s.y+uy*(r+1);
  const x2=t.x-ux*(r+6), y2=t.y-uy*(r+6);
  const c=(x2-x1)*0.42;
  return `M${x1},${y1} C${x1+c},${y1} ${x2-c},${y2} ${x2},${y2}`;
}

function lay(nd, W, H) {
  const px=80, py=52, ls={};
  nd.forEach(n => (ls[n.layer] = ls[n.layer] || []).push(n));
  const ml = Math.max(...nd.map(n => n.layer));
  nd.forEach(n => { n.x = px + n.layer*(W-2*px)/Math.max(ml,1); });
  Object.values(ls).forEach(g => {
    const c = g.length;
    g.forEach((n,i) => { n.y = c===1 ? H/2 : py + i*(H-2*py)/(c-1); });
  });
}

function draw() {
  const sv = document.getElementById("sv");
  sv.innerHTML = "";
  const W = sv.clientWidth || 860;
  const H = Math.max(360, sv.clientHeight || 420);
  const nd = DATA.nodes.map(n => ({...n}));
  const nm = {};
  nd.forEach(n => nm[n.id] = n);
  lay(nd, W, H);

  const s = DATA.stats;
  document.getElementById("srow").innerHTML =
    `<span>:${DATA.namespace}</span>` +
    `<span>nodes<b>${s.n_nodes}</b></span>` +
    `<span>edges<b>${s.n_edges}</b></span>` +
    `<span>params<b>${s.n_params}</b></span>` +
    `<span>depth<b>${s.depth}</b></span>`;

  const svg = d3.select("#sv").append("svg").attr("width", W).attr("height", H);

  svg.append("defs").append("marker")
    .attr("id","arw").attr("markerUnits","userSpaceOnUse")
    .attr("markerWidth",8).attr("markerHeight",6)
    .attr("refX",7).attr("refY",3).attr("orient","auto")
    .append("polygon").attr("points","0 0,8 3,0 6").attr("class","arw");

  const ml = Math.max(...nd.map(n => n.layer));
  for (let l=0; l<=ml; l++) {
    const xp = 80 + l*(W-160)/Math.max(ml,1);
    svg.append("line").attr("class","gl")
       .attr("x1",xp).attr("x2",xp).attr("y1",0).attr("y2",H);
  }

  svg.append("g").selectAll("path").data(DATA.edges).join("path")
    .attr("class","edge")
    .attr("d", e => { const s=nm[e.source], t=nm[e.target]; return s&&t ? ep(s,t) : ""; })
    .attr("marker-end","url(#arw)");

  const hi = document.getElementById("hi");
  const ng = svg.append("g").selectAll("g").data(nd).join("g")
    .attr("transform", d => `translate(${d.x},${d.y})`)
    .on("mouseenter", (ev, d) => {
      hi.innerHTML = `<b>:${d.id}</b> &nbsp;·&nbsp; ${d.type} &nbsp;·&nbsp; layer ${d.layer}`;
      d3.selectAll(".nc").style("opacity", n => n.id===d.id ? 1 : 0.2);
      d3.selectAll(".edge").style("opacity", e =>
        (e.source===d.id || e.target===d.id) ? 0.85 : 0.05);
    })
    .on("mouseleave", () => {
      hi.textContent = "Hover over a node to inspect it";
      d3.selectAll(".nc,.edge").style("opacity", null);
    });

  ng.append("circle").attr("r", 18)
    .attr("class", d => KC[d.type] || "nc nco");

  ng.append("text")
    .attr("class", d => KT[d.type] || "nt tco")
    .attr("font-family", "'JetBrains Mono','Fira Code','Consolas',monospace")
    .attr("textLength", d => d.id.length > 5 ? "30" : null)
    .attr("lengthAdjust", "spacingAndGlyphs")
    .text(d => d.id);
}

draw();
window.addEventListener("resize", draw);
</script></body></html>"""

function _jv_html(json_str::String, title::String)
    replace(
        replace(_JV_TMPL, "JULIUS_GRAPH_DATA" => json_str),
        "JULIUS_TITLE"    => title,
    )
end

"""
    save_graph_html(g, filepath; namespace, title, open_browser)

Génère une visualisation HTML interactive du DAG NeuroDSL et la sauvegarde.

# Arguments
- `g`            : JuliusGraph à visualiser
- `filepath`     : chemin de sortie (ex: "mlp.html")
- `namespace`    : namespace à visualiser (défaut: `g.active_ns`)
- `title`        : titre de la page (défaut: "NeuroDSL — :namespace")
- `open_browser` : ouvre dans le navigateur après la sauvegarde (défaut: false)

# Exemple
```julia
include("julius_viz.jl")

save_graph_html(mlp,  "mlp.html";  namespace=:mlp)
save_graph_html(attn, "attn.html"; namespace=:attn, open_browser=true)

# Plusieurs graphes dans une boucle
for (g, ns) in [(mlp,:mlp),(attn,:attn)]
    save_graph_html(g, "viz_\$ns.html"; namespace=ns)
end
```
"""
function save_graph_html(g::JuliusGraph, filepath::String;
                          namespace    = g.active_ns,
                          title        = "NeuroDSL — :$(namespace)",
                          open_browser = false)
    data = _jv_data(g; namespace=namespace)
    html = _jv_html(_jv(data), title)
    open(filepath, "w") do f; write(f, html); end

    s = data["stats"]
    println("✅ Sauvegardé : $filepath")
    println("   :$(namespace)  $(s["n_nodes"]) nœuds · " *
            "$(s["n_edges"]) arêtes · profondeur $(s["depth"])")

    if open_browser
        cmd = Sys.iswindows() ? `cmd /c start $filepath` :
              Sys.isapple()   ? `open $filepath`          :
                                `xdg-open $filepath`
        run(cmd; wait=false)
    end

    return filepath
end