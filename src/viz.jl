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
    grads  :: Dict{Symbol, AbstractArray{Float32}}
    function TrainingSnapshot(epoch, iter, loss, log,
                               params = Dict{Symbol, AbstractArray{Float32}}(),
                               grads  = Dict{Symbol, AbstractArray{Float32}}())
        new(epoch, iter, loss, log, params, grads)
    end
end

"""
    capture_snapshot(g, ns, epoch, iter, loss, log) -> TrainingSnapshot

Construit un `TrainingSnapshot` en capturant automatiquement la valeur ET le
gradient courants de chaque paramètre du graphe — remplace le duo
`capture_params`/`capture_grads` + construction manuelle de `TrainingSnapshot`
qu'un notebook devait sinon réécrire lui-même pour chaque script. Fonctionne
pour n'importe quel graphe/namespace, pas seulement un MLP en particulier.

À appeler juste après `backward_graph!` et AVANT l'étape d'optimiseur : la
plupart des règles de mise à jour (ex. `adamw_step!`) remettent le gradient à
zéro comme dernière étape, donc capturer après donnerait toujours un
gradient nul.
"""
function capture_snapshot(g::NeuroGraph, ns::Symbol, epoch::Int, iter::Int,
                          loss::Real, log::ExecutionLog)
    params = Dict{Symbol, AbstractArray{Float32}}()
    grads  = Dict{Symbol, AbstractArray{Float32}}()
    for (sym, nd) in g.nodes[ns]
        nd.is_param || continue
        nd.value    !== nothing && (params[sym] = copy(nd.value))
        nd.gradient !== nothing && (grads[sym]  = copy(nd.gradient))
    end
    return TrainingSnapshot(epoch, iter, Float32(loss), log, params, grads)
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
# Utilitaires de formatage
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

"""
    format_tensor_preview(value; max_rows=8, max_cols=10)

Comme `format_tensor_full`, mais tronque les tenseurs volumineux (coin
haut-gauche `max_rows`×`max_cols`, avec `…`/`⋮` pour indiquer la troncature
et la forme complète en rappel). Utilisé pour les tooltips des viewers : une
matrice 64×64 ou 128×128 intégralement dépliée en texte y était illisible
(et lourde à positionner/scroller) sans apporter d'information utile.
"""
function format_tensor_preview(value; max_rows::Int=8, max_cols::Int=10)
    try
        v = Array(value)
        if ndims(v) == 0
            return @sprintf("%.4f", Float64(v[]))
        elseif ndims(v) == 1
            n = length(v)
            n <= max_cols && return format_tensor_full(v)
            head = join([@sprintf("%.4f", Float64(x)) for x in v[1:max_cols]], ", ")
            return "[" * head * ", …]  ($n éléments)"
        elseif ndims(v) == 2
            rows, cols = size(v)
            (rows <= max_rows && cols <= max_cols) && return format_tensor_full(v)
            rshow, cshow = min(rows, max_rows), min(cols, max_cols)
            row_strs = String[]
            for r in 1:rshow
                line = join([@sprintf("%.4f", Float64(v[r,c])) for c in 1:cshow], "  ")
                cols > cshow && (line *= "  …")
                push!(row_strs, line)
            end
            rows > rshow && push!(row_strs, "⋮")
            return "[" * join(row_strs, "\n ") * "]\n($(rows)×$(cols) — aperçu $(rshow)×$(cshow))"
        else
            return "$(ndims(v))-tensor"
        end
    catch
        return "?"
    end
end

_safe_num(x) = isfinite(x) ? round(Float64(x); digits=6) :
               (isnan(x) ? "NaN" : (x > 0 ? "Inf" : "-Inf"))

"""
    format_tensor_grid(value; max_rows=12, max_cols=12) -> Dict

Représentation STRUCTURÉE (JSON-friendly, pas du texte préformaté comme
`format_tensor_preview`) d'un tenseur, tronquée au coin haut-gauche
`max_rows`×`max_cols`. Consommée par le panneau "tableau" cliquable de
viz.jl pour construire un vrai `<table>` HTML plutôt que de parser du texte.
Les valeurs non finies (NaN/Inf, ex. gradients divergents) sont converties
en chaînes pour rester JSON-valides.
"""
function format_tensor_grid(value; max_rows::Int=64, max_cols::Int=64)
    try
        v = Array(value)
        if ndims(v) == 0
            return Dict("shape" => "scalaire", "rows" => Any[[_safe_num(Float64(v[]))]],
                        "trunc_rows" => false, "trunc_cols" => false)
        elseif ndims(v) == 1
            n = length(v)
            cshow = min(n, max_cols)
            row = Any[_safe_num(Float64(v[i])) for i in 1:cshow]
            return Dict("shape" => "($n,)", "rows" => Any[row],
                        "trunc_rows" => false, "trunc_cols" => n > cshow)
        elseif ndims(v) == 2
            rows, cols = size(v)
            rshow, cshow = min(rows, max_rows), min(cols, max_cols)
            grid = Any[Any[_safe_num(Float64(v[r, c])) for c in 1:cshow] for r in 1:rshow]
            return Dict("shape" => "($rows×$cols)", "rows" => grid,
                        "trunc_rows" => rows > rshow, "trunc_cols" => cols > cshow)
        else
            return Dict("shape" => "$(ndims(v))-tensor", "rows" => Any[],
                        "trunc_rows" => false, "trunc_cols" => false)
        end
    catch
        return Dict("shape" => "?", "rows" => Any[], "trunc_rows" => false, "trunc_cols" => false)
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

    function label_of(s::Symbol)
        nd = get(graph.nodes[ns], s, nothing)
        if nd !== nothing
            return get(nd.aux_data, :label, string(s))
        else
            return string(s)
        end
    end

    if op == :linear
        X, W, b = inputs[1], inputs[2], inputs[3]
        return "Linear($X,$W,$b)"
    elseif op == :matmul
        A, B = inputs[1], inputs[2]
        tb = get(rule.attrs, :trans_b, false)
        return tb ? "$(label_of(A))·$(label_of(B))ᵀ" : "$(label_of(A))·$(label_of(B))"
    elseif op == :relu
        return "ReLU($(label_of(inputs[1])))"
    elseif op == :add
        return "$(label_of(inputs[1])) + $(label_of(inputs[2]))"
    elseif op == :sum_matrix
        return "Σ $(label_of(inputs[1]))"
    elseif op == :wsum
        a, b = inputs[1], inputs[2]
        weights = get(rule.attrs, :weights, Float32[0,0])
        w1, w2 = weights[1], weights[2]
        return "$w1·$(label_of(a)) + $w2·$(label_of(b))"
    elseif op == :nsum
        return "Σ(" * join(label_of.(inputs), ", ") * ")"
    elseif op == :fused_matmul_relu
        A, B = inputs[1], inputs[2]
        return "ReLU($(label_of(A))·$(label_of(B)))"
    elseif op == :scale_add
        a, b = inputs[1], inputs[2]
        factor = get(rule.attrs, :factor, 0)
        return "$factor·$(label_of(a)) + $(label_of(b))"
    else
        return "$sym = $op(…)"
    end
end

function is_operator_node(sym::Symbol)
    name = string(sym)
    return startswith(name, "wsum_") || startswith(name, "nsum_") ||
           startswith(name, "add_") || startswith(name, "scale_add_") ||
           name in ("wsum", "nsum", "identity", "add", "scale_add")
end

# ═════════════════════════════════════════════════════════════════════════════
# _graph_viz_json — préparation des données partagée entre les deux viewers
#
# Avant refactor, ce bloc (formulas/short_labels/nodes_json/edges_json/
# init_vals/full_vals) était dupliqué à l'identique dans save_interactive_graph
# et save_interactive_graph_animated. Extrait ici une seule fois.
# ═════════════════════════════════════════════════════════════════════════════
function _graph_viz_json(graph::NeuroGraph, ns::Symbol)
    rules_ns = get(graph.rules, ns, Dict{Symbol,Any}())

    init_vals  = Dict{String,String}()
    full_vals  = Dict{String,String}()
    full_grids = Dict{String,Any}()
    formulas   = Dict{String,String}()
    is_leaf_d  = Dict{String,Bool}()
    is_param_d = Dict{String,Bool}()
    is_rule_d  = Dict{String,Bool}()

    for (sym, nd) in graph.nodes[ns]
        k = string(sym)
        init_vals[k] = nd.value !== nothing ? begin
                                                    s = size(nd.value)
                                                    "[" * join(s, "×") * "]"
                                                end : "?"
        full_vals[k]  = nd.value !== nothing ? format_tensor_preview(nd.value) : "?"
        # Cap plus généreux qu'ailleurs : ce champ est calculé UNE FOIS par
        # nœud (pas par étape), donc son coût ne se multiplie pas avec la
        # longueur de la trace — contrairement aux grilles loguées à chaque
        # forward (dispatch.jl) ou aux poids par snapshot (params_grid), dont
        # le cap reste modéré pour ne pas faire exploser la taille du HTML.
        full_grids[k] = nd.value !== nothing ? format_tensor_grid(nd.value; max_rows=200, max_cols=200) : nothing
        formulas[k]   = get(nd.aux_data, :label, node_formula_text(graph, sym, ns))
        is_leaf_d[k]  = !haskey(rules_ns, sym)
        is_param_d[k] = nd.is_param
        is_rule_d[k]  = haskey(nd.aux_data, :is_rule) && nd.aux_data[:is_rule]
    end

    short_labels = Dict{String,String}()
    for (sym, nd) in graph.nodes[ns]
        k = string(sym)
        if is_rule_d[k]
            short_labels[k] = formulas[k]
        elseif haskey(rules_ns, sym)
            op = rules_ns[sym].op
            if op in (:wsum, :nsum, :add, :identity, :relu, :tanh, :matmul, :fused_matmul_relu, :scale_add)
                short_labels[k] = string(op)
            else
                short_labels[k] = formulas[k]
            end
        else
            short_labels[k] = formulas[k]
        end
    end

    edges = Tuple{Symbol,Symbol}[]
    for (out_sym, rule) in rules_ns
        for inp in rule.inputs
            push!(edges, (inp, out_sym))
        end
    end

    nodes_json = JSON.json([Dict(:id => string(s),
                                  :is_param => is_param_d[string(s)],
                                  :is_leaf  => is_leaf_d[string(s)],
                                  :is_op    => is_operator_node(s) && !is_rule_d[string(s)])
                             for s in keys(graph.nodes[ns])])

    return (
        nodes_json        = nodes_json,
        edges_json        = JSON.json([[string(a), string(b)] for (a, b) in edges]),
        init_json         = JSON.json(init_vals),
        full_json         = JSON.json(full_vals),
        full_grids_json   = JSON.json(full_grids),
        formulas_json     = JSON.json(formulas),
        short_labels_json = JSON.json(short_labels),
    )
end

# ═════════════════════════════════════════════════════════════════════════════
# Thèmes de couleur — clair (trace statique) / sombre (trace d'entraînement)
#
# Seules les règles CSS de rendu du graphe (nœuds/arêtes/tooltips/zoom) sont
# paramétrées par thème : le chrome de page (sidebar, topbar, slider d'époque,
# courbe de perte…) diffère structurellement entre les deux viewers et reste
# propre à chacun.
# ═════════════════════════════════════════════════════════════════════════════
const _VIZ_LIGHT_THEME = (
    node_default_fill="#fff",    node_default_stroke="#cbd5e0",
    node_op_fill="#2a2e3f",      node_op_stroke="#d4a373",
    node_param_fill="#f1f5f9",   node_param_stroke="#94a3b8",
    node_fwd_fill="#dcfce7",     node_fwd_stroke="#22c55e",
    node_bwd_fill="#fee2e2",     node_bwd_stroke="#ef4444",
    node_final_fill="#bbf7d0",   node_final_stroke="#16a34a", node_final_shadow="rgba(22,163,74,.4)",
    node_label_fill="#1e293b",   node_val_fill="#64748b",     node_grad_fill="#ef4444",
    edge_default="#94a3b8", edge_default_opacity=".5", edge_fwd="#22c55e", edge_bwd="#ef4444", edge_final="#16a34a",
    canvas_bg="#f8fafc",         canvas_dot="#d1d5db",
    node_shadow="rgba(0,0,0,.08)",
    zoom_bg="#fff",   zoom_border="#e2e8f0", zoom_fg="#334155", zoom_size="34px",
    zoom_radius="8px", zoom_fontsize="17px", zoom_shadow="box-shadow: 0 2px 6px rgba(0,0,0,.08);",
    zoom_hover_bg="#f1f5f9", zoom_hover_fg="",
    sep_stroke="#e2e8f0",
    marker_colors=("#94a3b8", "#22c55e", "#ef4444", "#16a34a"),
)

const _VIZ_DARK_THEME = (
    node_default_fill="#1e2030", node_default_stroke="#3b3f53",
    node_op_fill="#2a2e3f",      node_op_stroke="#d4a373",
    node_param_fill="#232636",   node_param_stroke="#4b5268",
    node_fwd_fill="#1a2e1a",     node_fwd_stroke="#6a9b6a",
    node_bwd_fill="#2e1a1a",     node_bwd_stroke="#e06c75",
    node_final_fill="#1e3320",   node_final_stroke="#4caf50", node_final_shadow="rgba(76,175,80,.5)",
    node_label_fill="#d1d5db",   node_val_fill="#9ca3af",     node_grad_fill="#e06c75",
    edge_default="#3b3f53", edge_default_opacity=".6", edge_fwd="#6a9b6a", edge_bwd="#e06c75", edge_final="#4caf50",
    canvas_bg="#12141c",         canvas_dot="#1e2030",
    node_shadow="rgba(0,0,0,.2)",
    zoom_bg="#1e2030", zoom_border="#3b3f53", zoom_fg="#9ca3af", zoom_size="32px",
    zoom_radius="7px", zoom_fontsize="16px", zoom_shadow="",
    zoom_hover_bg="#2d3143", zoom_hover_fg="#e5e7eb",
    sep_stroke="#3b3f53",
    marker_colors=("#3b3f53", "#6a9b6a", "#e06c75", "#4caf50"),
)

function _viz_graph_css(t)
    """
#graph-area {
  flex: 1; display: flex; flex-direction: column; overflow: hidden;
  min-width: 0; min-height: 0;
}
#canvas-wrap {
  flex: 1; position: relative; overflow: hidden; min-height: 0;
  background: $(t.canvas_bg);
  background-image: radial-gradient(circle, $(t.canvas_dot) 1px, transparent 1px);
  background-size: 24px 24px;
}
#svg-canvas { position: absolute; top: 0; left: 0; transform-origin: 0 0; overflow: visible; }
.node-body {
  transition: all .2s ease; cursor: pointer;
  filter: drop-shadow(0 1px 4px $(t.node_shadow));
}
.node-default { fill: $(t.node_default_fill); stroke: $(t.node_default_stroke); stroke-width: 1.5px; }
.node-op      { fill: $(t.node_op_fill);      stroke: $(t.node_op_stroke);      stroke-width: 1.2px; }
.node-param   { fill: $(t.node_param_fill);   stroke: $(t.node_param_stroke);   stroke-width: 1.5px; stroke-dasharray: 5,3; }
.node-fwd     { fill: $(t.node_fwd_fill);     stroke: $(t.node_fwd_stroke);     stroke-width: 2.5px; }
.node-bwd     { fill: $(t.node_bwd_fill);     stroke: $(t.node_bwd_stroke);     stroke-width: 2.5px; }
.node-final   {
  fill: $(t.node_final_fill); stroke: $(t.node_final_stroke); stroke-width: 2.5px;
  filter: drop-shadow(0 2px 8px $(t.node_final_shadow));
}
.node-label { font-family: 'Consolas', monospace; font-size: 10.5px; font-weight: 700;
              fill: $(t.node_label_fill); pointer-events: none; }
.node-val   { font-family: 'Consolas', monospace; font-size: 9.5px; fill: $(t.node_val_fill);  pointer-events: none; }
.edge         { fill: none; stroke-width: 1.5px; stroke-linejoin: round; transition: all .2s ease; }
.edge-default { stroke: $(t.edge_default); stroke-opacity: $(t.edge_default_opacity); }
.edge-fwd     { stroke: $(t.edge_fwd);   stroke-width: 2.5px; stroke-opacity: 1; }
.edge-bwd     { stroke: $(t.edge_bwd);   stroke-width: 2.5px; stroke-opacity: 1; }
.edge-final   { stroke: $(t.edge_final); stroke-width: 2.5px; stroke-opacity: 1; }
.tooltip-left, .tooltip-right {
  position: fixed;
  /* légèrement transparent (pas opaque à 100%) : un nœud voisin recouvert
     par la bulle reste visible en transparence plutôt que d'être caché */
  background: rgba(30,32,48,.94);
  backdrop-filter: blur(2px);
  color: #e5e7eb;
  padding: 12px 16px;
  border-radius: 12px;
  font-size: inherit;
  pointer-events: none;
  display: none;
  z-index: 9999;
  max-width: none !important;
  width: max-content;
  max-height: 40vh;
  overflow: auto;
  box-shadow: 0 8px 24px rgba(0,0,0,.5);
  border: 1px solid #3b3f53;
  font-family: 'Consolas', monospace;
  word-break: normal;
}
.tooltip-left b, .tooltip-right b { color: #d4a373; }
.ntbl-hint { margin-top: 8px; opacity: .6; font-size: .85em; }
.tooltip-left pre, .tooltip-right pre {
  margin: 6px 0 0;
  font-size: inherit;
  white-space: pre;
  word-break: normal;
  color: #d1d5db;
}
.zoom-bar {
  position: absolute; bottom: 14px; right: 14px;
  display: flex; flex-direction: column; gap: 4px;
}
.zoom-btn {
  width: $(t.zoom_size); height: $(t.zoom_size); background: $(t.zoom_bg); border: 1px solid $(t.zoom_border);
  border-radius: $(t.zoom_radius); font-size: $(t.zoom_fontsize); cursor: pointer;
  display: flex; align-items: center; justify-content: center;
  $(t.zoom_shadow) transition: background .1s; color: $(t.zoom_fg);
}
.zoom-btn:hover { background: $(t.zoom_hover_bg);$(isempty(t.zoom_hover_fg) ? "" : " color: $(t.zoom_hover_fg);") }
#node-table-panel {
  display: none; flex-direction: column; flex-shrink: 0;
  border-top: 1px solid $(t.node_default_stroke);
  background: $(t.canvas_bg); position: relative;
}
#node-table-panel .ntbl-resize {
  position: absolute; top: -3px; left: 0; right: 0; height: 6px;
  cursor: ns-resize; z-index: 5;
}
#node-table-panel .ntbl-header {
  display: flex; align-items: center; justify-content: space-between;
  padding: 6px 12px; border-bottom: 1px solid $(t.node_default_stroke);
  font-family: 'Consolas', monospace; font-size: 11px; font-weight: 700;
  color: $(t.node_label_fill); flex-shrink: 0;
}
#node-table-panel .ntbl-close {
  cursor: pointer; border: none; background: transparent; font-size: 12px;
  color: $(t.node_val_fill); line-height: 1; padding: 4px 8px; border-radius: 4px;
  font-family: 'Consolas', monospace;
}
#node-table-panel .ntbl-close:hover { background: $(t.node_shadow); }
#node-table-panel .ntbl-body {
  flex: 1; overflow-x: auto; overflow-y: hidden; padding: 10px 12px;
  display: flex; gap: 16px; align-items: flex-start;
}
.ntbl-card {
  flex: 0 0 auto; width: 320px; height: 100%;
  min-width: 220px; min-height: 160px; max-width: 90vw; max-height: 80vh;
  border: 1px solid $(t.node_default_stroke); border-radius: 8px;
  display: flex; flex-direction: column;
  background: $(t.node_default_fill);
  resize: both; overflow: auto;
}
.ntbl-card-header {
  display: flex; align-items: center; justify-content: space-between;
  padding: 5px 10px; border-bottom: 1px solid $(t.node_default_stroke);
  font-family: 'Consolas', monospace; font-size: 10.5px; font-weight: 700;
  color: $(t.node_label_fill); flex-shrink: 0;
}
.ntbl-card-header .ntbl-close { font-size: 13px; padding: 2px 6px; }
.ntbl-card-body { flex: 1; overflow-y: auto; padding: 8px 10px; }
.ntbl-table-wrap { overflow: auto; max-height: 220px; }
.ntbl-section h4 {
  font-size: 10px; text-transform: uppercase; letter-spacing: .4px;
  color: $(t.node_val_fill); margin-bottom: 5px; font-weight: 600;
}
.ntbl-plot-hint { font-size: 9px; font-weight: 400; text-transform: none; letter-spacing: 0; opacity: .65; }
table.ntbl { border-collapse: collapse; font-family: 'Consolas', monospace; font-size: 10.5px; color: $(t.node_label_fill); }
table.ntbl td { border: 1px solid $(t.node_default_stroke); padding: 2px 7px; text-align: right; }
table.ntbl td[data-r] { cursor: pointer; }
table.ntbl td[data-r]:hover { background: $(t.node_shadow); }
table.ntbl td.ntbl-cell-active { outline: 2px solid #d4a373; outline-offset: -2px; background: $(t.node_shadow); }
.ntbl-trunc { color: $(t.node_val_fill); text-align: center !important; cursor: default !important; }
.ntbl-shape { font-size: 10px; color: $(t.node_val_fill); margin-top: 5px; }
.ntbl-empty { font-size: 11px; color: $(t.node_val_fill); font-style: italic; }
.ntbl-raw {
  font-family: 'Consolas', monospace; font-size: 10.5px;
  color: $(t.node_grad_fill); white-space: pre-wrap;
}
.ntbl-raw-clickable { cursor: pointer; border-radius: 4px; padding: 2px 4px; margin: -2px -4px; }
.ntbl-raw-clickable:hover { background: $(t.node_shadow); text-decoration: underline; }
/* Courbe de valeur et courbe de gradient côte à côte (pas empilées) : la
   carte est redimensionnable (resize:both) donc l'utilisateur peut l'élargir
   pour leur laisser de la place. */
.ntbl-plot-slot {
  margin-top: 10px; padding-top: 8px; border-top: 1px dashed $(t.node_default_stroke);
  display: flex; gap: 12px; align-items: flex-start;
}
.ntbl-plot-block { flex: 1 1 0; min-width: 0; }
.ntbl-plot-title {
  font-size: 9.5px; font-family: 'Consolas', monospace; color: $(t.node_op_stroke);
  font-weight: 700; margin-bottom: 4px;
}
.ntbl-plot-block canvas { width: 100% !important; height: 130px !important; }
.ntbl-plot-header { display: flex; align-items: center; justify-content: space-between; }
.ntbl-plot-header .ntbl-close { font-size: 12px; padding: 1px 6px; }
.ntbl-plot-grad { padding-left: 12px; border-left: 1px dotted $(t.node_default_stroke); }
"""
end

# ═════════════════════════════════════════════════════════════════════════════
# _viz_table_panel_html — panneau "tableau dynamique" partagé entre les deux
# viewers. Caché par défaut ; ouvert au clic sur un nœud, refermable (✕) pour
# rendre l'espace au graphe, redimensionnable via la poignée du haut.
# ═════════════════════════════════════════════════════════════════════════════
function _viz_table_panel_html()
    """
<div id="node-table-panel">
  <div class="ntbl-resize" id="ntblResize"></div>
  <div class="ntbl-header">
    <span id="ntbl-title">Cliquez sur un nœud pour explorer ses valeurs</span>
    <button class="ntbl-close" onclick="closeAllNodeTables()" title="Tout fermer">✕ Tout fermer</button>
  </div>
  <div class="ntbl-body" id="ntbl-body"></div>
</div>
"""
end

# ═════════════════════════════════════════════════════════════════════════════
# _viz_shared_js — moteur JS commun aux deux viewers : layout dagre, rendu SVG
# des nœuds/arêtes, tooltips au survol, zoom/pan.
#
# Deux améliorations apportées ici s'appliquent désormais aux DEUX viewers
# (avant refactor, seule la version animée les avait, ou aucune des deux) :
#   1. edgePts/polylinePath utilisent le routage robuste (intersection de
#      rectangles + courbe de Bézier pour les arêtes à 2 points) partout —
#      la version animée avait un routage plus grossier.
#   2. Le tooltip au survol se rafraîchit automatiquement pendant la lecture
#      pas-à-pas (hoveredNodeId) — avant, seule la version animée le faisait.
#
# Suppose définis en amont dans le <script> : NODES_RAW, EDGES, SHORT_LABELS,
# INIT_VALS, FULL_VALS, FORMULAS, et les éléments DOM #svg-canvas,
# #tooltip-left, #tooltip-right, #canvas-wrap.
# ═════════════════════════════════════════════════════════════════════════════
function _viz_shared_js(marker_colors::NTuple{4,String}, sep_stroke::String)
    marker_defs = join(["  ['$id', '$color']" for (id, color) in
                        zip(("arr-default","arr-fwd","arr-bwd","arr-final"), marker_colors)], ",\n")
    """
const NH = 48;
function estimateTextWidth(text, fontSize) {
  if (!text) return 60;
  const charWidth = fontSize * 0.6;
  return Math.max(60, text.length * charWidth + 20);
}
const labelFontSize = 10.5;

const dg = new dagre.graphlib.Graph({ multigraph: false });
dg.setGraph({
  rankdir: 'LR', nodesep: 65, ranksep: 110,
  marginx: 55, marginy: 55, edgesep: 25,
  acyclicer: 'greedy', ranker: 'network-simplex'
});
dg.setDefaultEdgeLabel(() => ({}));
NODES_RAW.forEach(n => {
  const label = SHORT_LABELS[n.id] || n.id;
  const w = estimateTextWidth(label, labelFontSize);
  dg.setNode(n.id, { width: w, height: NH });
});
EDGES.forEach(([s, d]) => dg.setEdge(s, d));
dagre.layout(dg);

const NODES = NODES_RAW.map(n => {
  const dn = dg.node(n.id);
  return { ...n, x: Math.round(dn.x - dn.width/2), y: Math.round(dn.y - NH/2), w: dn.width, h: NH };
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
$marker_defs
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
    if (y1 !== y2) {
      const t = (top - y1) / (y2 - y1);
      if (t >= 0 && t <= 1) {
        const xi = x1 + t * (x2 - x1);
        if (xi >= left && xi <= right) pts.push({x: xi, y: top});
      }
    }
    if (y1 !== y2) {
      const t = (bottom - y1) / (y2 - y1);
      if (t >= 0 && t <= 1) {
        const xi = x1 + t * (x2 - x1);
        if (xi >= left && xi <= right) pts.push({x: xi, y: bottom});
      }
    }
    if (x1 !== x2) {
      const t = (left - x1) / (x2 - x1);
      if (t >= 0 && t <= 1) {
        const yi = y1 + t * (y2 - y1);
        if (yi >= top && yi <= bottom) pts.push({x: left, y: yi});
      }
    }
    if (x1 !== x2) {
      const t = (right - x1) / (x2 - x1);
      if (t >= 0 && t <= 1) {
        const yi = y1 + t * (y2 - y1);
        if (yi >= top && yi <= bottom) pts.push({x: right, y: yi});
      }
    }
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
    return 'M ' + pts[0].x + ' ' + pts[0].y + ' Q ' + (pts[0].x + dx/2) + ' ' + (pts[0].y + dy/2) + ', ' + pts[1].x + ' ' + pts[1].y;
  }
  let d = 'M ' + pts[0].x + ' ' + pts[0].y;
  for (let i = 1; i < pts.length; i++) d += ' L ' + pts[i].x + ' ' + pts[i].y;
  return d;
}
function trunc(s, max=35) { return s.length > max ? s.slice(0, max-1) + '…' : s; }

let hoveredNodeId = null;
let currentParamFull = {};
let currentParamGrid = {};
let openNodeIds = [];          // nœuds dont le tableau est ouvert, dans l'ordre de clic
let activeCellByNode = {};     // nodeId -> {r,c} : cellule sélectionnée pour le graphe d'évolution
let weightCharts = {};         // "nodeId:r:c" -> instance Chart.js, détruite/reconstruite à chaque rendu
let panelHeight = 220;
const tooltipLeft  = document.getElementById('tooltip-left');
const tooltipRight = document.getElementById('tooltip-right');

// ── Conversion grille structurée (format_tensor_grid côté Julia) → texte / table ──
// ev.val peut être soit une chaîne (anciens événements backward poussés
// manuellement par un notebook), soit un objet grille {shape, rows,
// trunc_rows, trunc_cols} (nouveaux événements forward). Ces helpers
// unifient les deux pour le tooltip (texte) et le panneau (table HTML).
function isGridVal(v) { return v && typeof v === 'object' && Array.isArray(v.rows); }
function fmtCell(x) { return typeof x === 'number' ? x.toFixed(4) : String(x); }
// Aperçu volontairement PETIT et FIXE (8×8 max), indépendant de la taille
// réelle de la grille (jusqu'à 64×64 depuis dispatch.jl) — le tooltip est une
// bulle flottante non scrollable qui reste ouverte tant qu'on survole le
// nœud : sans ce plafond séparé, une grande matrice y produisait un pavé de
// texte qui recouvrait les nœuds voisins. La grille complète reste
// disponible dans le panneau tableau (gridToTableHTML), qui lui défile.
function gridToText(g) {
  if (!g || !g.rows || !g.rows.length) return '';
  const MAX_R = 8, MAX_C = 8;
  const rShow = Math.min(g.rows.length, MAX_R);
  const cShow = Math.min((g.rows[0] || []).length, MAX_C);
  const truncR = g.trunc_rows || g.rows.length > rShow;
  const truncC = g.trunc_cols || (g.rows[0] || []).length > cShow;
  const lines = [];
  for (let r = 0; r < rShow; r++) {
    let line = g.rows[r].slice(0, cShow).map(fmtCell).join('  ');
    if (truncC) line += '  …';
    lines.push(line);
  }
  if (truncR) lines.push('⋮');
  return lines.join('\\n') + (g.shape ? '\\n(' + g.shape + ')' : '');
}
function valToText(v) { return isGridVal(v) ? gridToText(v) : (v || ''); }
function gridToTableHTML(g, emptyMsg) {
  if (!g || !g.rows || !g.rows.length) return '<div class="ntbl-empty">' + (emptyMsg || '—') + '</div>';
  let html = '<table class="ntbl">';
  g.rows.forEach((row, r) => {
    html += '<tr>' + row.map((x, c) => '<td data-r="' + r + '" data-c="' + c + '">' + fmtCell(x) + '</td>').join('') +
            (g.trunc_cols ? '<td class="ntbl-trunc">…</td>' : '') + '</tr>';
  });
  if (g.trunc_rows) {
    html += '<tr><td colspan="' + (g.rows[0].length + (g.trunc_cols ? 1 : 0)) + '" class="ntbl-trunc">⋮</td></tr>';
  }
  html += '</table>';
  if (g.shape) html += '<div class="ntbl-shape">' + g.shape + '</div>';
  return html;
}

// Place un tooltip sur le côté demandé du nœud, en clampant dans la fenêtre
// (aussi bien à gauche/droite qu'en haut/bas — l'ancien code ne clampait pas
// le repli de tooltipLeft contre le bord droit, ce qui pouvait le faire
// déborder après son propre repli).
function placeTooltip(el, r, side) {
  const w = el.offsetWidth, h = el.offsetHeight;
  let left = side === 'right' ? r.right + 14 : r.left - w - 14;
  left = Math.max(8, Math.min(left, window.innerWidth - w - 8));
  let top = Math.max(8, r.top - 8);
  top = Math.min(top, window.innerHeight - h - 8);
  el.style.left = left + 'px';
  el.style.top  = top + 'px';
}

function updateTooltips(nodeId, rectElement) {
  const v = nodeVals[nodeId];
  const n = NMAP[nodeId];
  const bbox = rectElement.getBBox();
  const ctm = svgEl.getScreenCTM();
  const tl = svgEl.createSVGPoint(); tl.x = bbox.x; tl.y = bbox.y;
  const br = svgEl.createSVGPoint(); br.x = bbox.x + bbox.width; br.y = bbox.y + bbox.height;
  const stl = tl.matrixTransform(ctm);
  const sbr = br.matrixTransform(ctm);
  const r = { left: stl.x, top: stl.y, right: sbr.x, bottom: sbr.y };
  const baseFontSize = 12 * sc;
  tooltipLeft.style.fontSize  = Math.max(baseFontSize, 10) + 'px';
  tooltipRight.style.fontSize = Math.max(baseFontSize, 10) + 'px';
  const pad = Math.max(8, 12 * sc / 1.5);
  tooltipLeft.style.padding  = pad + 'px';
  tooltipRight.style.padding = pad + 'px';

  const showLeft = n.is_param && currentParamFull[nodeId];
  if (showLeft) {
    tooltipLeft.innerHTML = "<b>📦 Poids (" + nodeId + ")</b><pre>" + currentParamFull[nodeId] + "</pre>";
    tooltipLeft.style.display = 'block';
  } else {
    tooltipLeft.style.display = 'none';
  }

  const fwdVal = (v && v.fwd) ? valToText(v.fwd) : (FULL_VALS[nodeId] || '?');
  let rightHtml = "";
  if (FORMULAS[nodeId] && FORMULAS[nodeId] !== nodeId) {
    rightHtml += "<b>📐 Formule</b><pre>" + FORMULAS[nodeId] + "</pre>";
  }
  rightHtml += "<b>➡️ Forward (" + nodeId + ")</b><pre>" + fwdVal + "</pre>";
  if (v && v.bwd) rightHtml += "<b>🔻 Gradient (" + nodeId + ")</b><pre>" + valToText(v.bwd) + "</pre>";
  rightHtml += "<div class='ntbl-hint'>🖱️ Cliquer sur le nœud pour explorer en tableau</div>";
  tooltipRight.innerHTML = rightHtml;
  tooltipRight.style.display = 'block';

  // Les deux tooltips choisissent leur côté ENSEMBLE plutôt qu'indépendamment
  // (avant ce fix, chacun retombait sur son propre repli sans savoir ce que
  // l'autre avait décidé, et pouvait finir du même côté — ex. fc2_W et fc2_b).
  const spaceRight = window.innerWidth - r.right - 14;
  const spaceLeft  = r.left - 14;
  const rightGoesRight = spaceRight >= tooltipRight.offsetWidth || spaceRight >= spaceLeft;

  if (showLeft) {
    placeTooltip(tooltipRight, r, rightGoesRight ? 'right' : 'left');
    placeTooltip(tooltipLeft,  r, rightGoesRight ? 'left'  : 'right');
  } else {
    placeTooltip(tooltipRight, r, rightGoesRight ? 'right' : 'left');
  }
}

function buildGraphSVG() {
  const eLayer = document.createElementNS(SVGNS, 'g');
  eLayer.id = 'edge-layer';
  EDGES.forEach(([s, d]) => {
    const path = document.createElementNS(SVGNS, 'path');
    path.setAttribute('d',          polylinePath(edgePts(s, d)));
    path.setAttribute('class',      'edge edge-default');
    path.setAttribute('id',         'edge-' + s + '-' + d);
    path.setAttribute('marker-end', 'url(#arr-default)');
    eLayer.appendChild(path);
  });
  svgEl.appendChild(eLayer);

  const nLayer = document.createElementNS(SVGNS, 'g');
  nLayer.id = 'node-layer';
  NODES.forEach(n => {
    const g = document.createElementNS(SVGNS, 'g');
    const rect = document.createElementNS(SVGNS, 'rect');
    rect.setAttribute('x', n.x); rect.setAttribute('y', n.y);
    rect.setAttribute('width', n.w); rect.setAttribute('height', n.h);
    rect.setAttribute('rx', '8');
    rect.setAttribute('id', 'node-' + n.id);

    let cls = 'node-body ';
    cls += n.is_param ? 'node-param' : 'node-default';
    if (n.is_op) cls += ' node-op';
    rect.setAttribute('class', cls);

    rect.addEventListener('mouseenter', e => {
      hoveredNodeId = n.id;
      updateTooltips(n.id, e.target);
    });
    rect.addEventListener('mouseleave', () => {
      hoveredNodeId = null;
      tooltipLeft.style.display  = 'none';
      tooltipRight.style.display = 'none';
    });
    rect.addEventListener('click', () => openNodeTable(n.id));

    const sep = document.createElementNS(SVGNS, 'line');
    sep.setAttribute('x1', n.x + 10);   sep.setAttribute('x2', n.x + n.w - 10);
    sep.setAttribute('y1', n.y + 24);   sep.setAttribute('y2', n.y + 24);
    sep.setAttribute('stroke', '$sep_stroke'); sep.setAttribute('stroke-width', '1');
    sep.setAttribute('pointer-events', 'none');
    const mkT = (id, dy, cls) => {
      const t = document.createElementNS(SVGNS, 'text');
      t.setAttribute('x', n.x + n.w/2); t.setAttribute('y', n.y + dy);
      t.setAttribute('text-anchor', 'middle'); t.setAttribute('class', cls);
      if (id) t.setAttribute('id', id);
      return t;
    };
    // Le nœud n'affiche plus que 2 rangées statiques : nom de la fonction/
    // variable (SHORT_LABELS) et forme du résultat (INIT_VALS, ex. "[64×128]").
    // Les valeurs numériques (forward, poids, gradient) ne sont plus rendues
    // à l'intérieur du nœud : elles vivent dans le tooltip au survol (aperçu
    // rapide) et dans le panneau tableau au clic (exploration complète,
    // actualisée à chaque forward/backward — voir openNodeTable/renderAllNodeTables).
    const tLabel = mkT(null, 17, 'node-label');
    tLabel.textContent = trunc(SHORT_LABELS[n.id] || n.id);
    const tVal = mkT('val-' + n.id, 38, 'node-val');
    tVal.textContent = (INIT_VALS[n.id] && INIT_VALS[n.id] !== '?') ? INIT_VALS[n.id] : '';
    g.append(rect, sep, tLabel, tVal);
    nLayer.appendChild(g);
  });
  svgEl.appendChild(nLayer);
}

function setNC(id, cls) {
  const r = document.getElementById('node-' + id);
  if (r) {
    let fullCls = 'node-body ' + cls;
    const n = NMAP[id];
    if (n && n.is_op) fullCls += ' node-op';
    r.setAttribute('class', fullCls);
  }
}
function setEC(s, d, cls, arr) {
  const e = document.getElementById('edge-' + s + '-' + d);
  if (e) {
    e.setAttribute('class',      'edge ' + cls);
    e.setAttribute('marker-end', 'url(#' + arr + ')');
  }
}

// Rendu partagé de la relecture pas-à-pas d'une trace d'événements
// (LOG entier pour la version statique, snapshot.events pour la version
// animée). Avant refactor, seule la version animée rafraîchissait le
// tooltip au survol pendant la lecture (hoveredNodeId) ; c'est désormais
// le cas pour les deux. De même, la mise à jour du texte de valeur sur les
// nœuds pendant la relecture — présente seulement côté statique avant —
// s'applique maintenant aux deux.
function renderEventsUpTo(events, idx) {
  // Le corps du nœud est désormais statique (nom + forme, posés une fois
  // dans buildGraphSVG) : la relecture ne touche plus que la couleur d'état
  // (bordure) et les arêtes. Les valeurs numériques par étape (ev.val) sont
  // conservées dans nodeVals[...] pour le tooltip au survol et le panneau
  // tableau au clic (renderAllNodeTables) — plus jamais écrites dans le SVG.
  NODES.forEach(n => setNC(n.id, n.is_param ? 'node-param' : 'node-default'));
  EDGES.forEach(([s, d]) => setEC(s, d, 'edge-default', 'arr-default'));

  const logPanel = document.getElementById('log-panel');
  if (idx < 0) { if (logPanel) logPanel.innerHTML = ''; refreshNodeTable(); return; }

  let lastFwd = null;
  for (let i = 0; i <= idx; i++) {
    const ev = events[i];
    // node-fwd (vert) et node-bwd (rouge) marquent désormais un nœud aussi
    // bien "en cours" que "terminé" dans leur phase respective — avant ce
    // fix, un nœud forward terminé retombait sur node-default (neutre) et un
    // nœud backward terminé prenait node-done, qui était VERT dans les deux
    // thèmes malgré son nom neutre : la distinction forward/backward par
    // couleur ne tenait donc pas au-delà du tout dernier nœud forward.
    if (ev.phase === 'forward') {
      setNC(ev.node, 'node-fwd');
      if (ev.status !== 'starting') { lastFwd = ev.node; nodeVals[ev.node].fwd = ev.val || FULL_VALS[ev.node] || ''; }
    } else {
      setNC(ev.node, 'node-bwd');
      if (ev.status !== 'starting') nodeVals[ev.node].bwd = ev.val || '';
    }
    EDGES.forEach(([s, d]) => {
      if (d === ev.node) {
        const [ec, ea] = ev.phase === 'forward' ? ['edge-fwd', 'arr-fwd'] : ['edge-bwd', 'arr-bwd'];
        setEC(s, d, ec, ea);
      }
    });
  }
  if (lastFwd) {
    setNC(lastFwd, 'node-final');
    EDGES.forEach(([s, d]) => { if (d === lastFwd) setEC(s, d, 'edge-final', 'arr-final'); });
  }
  if (logPanel) {
    logPanel.innerHTML = events.slice(0, idx + 1).slice().reverse().map(ev =>
      '<div class="log-entry log-' + ev.phase + '">' +
      '<b>' + ev.phase.toUpperCase() + '</b> ' + ev.node + ' ' +
      (ev.status === 'starting' ? '→ computing…' : ': ' + trunc(valToText(ev.val), 60)) +
      '</div>'
    ).join('');
  }
  if (hoveredNodeId) {
    const rectEl = document.getElementById('node-' + hoveredNodeId);
    if (rectEl) updateTooltips(hoveredNodeId, rectEl);
  }
  refreshNodeTable();
}

// ── Panneau "tableaux dynamiques" (clic sur un/des nœud(s)) ─────────────────
// Caché par défaut. Chaque clic sur un nœud AJOUTE une carte (pas de
// remplacement) ; chaque carte a son propre ✕ pour se fermer individuellement,
// et "Tout fermer" vide le panneau entier pour rendre l'espace au graphe.
// Se réactualise à chaque étape forward/backward (refreshNodeTable, appelé
// en fin de renderEventsUpTo) et à chaque changement d'époque côté animé.
//
// Valeur forward affichée par carte, par ordre de priorité :
//   1. Grille structurée de CETTE étape (nodeVals[id].fwd, si un event forward
//      a été loggé pour ce nœud dans le pas courant — dispatch.jl le fait
//      nativement pour les nœuds calculés par une règle).
//   2. Texte brut de CETTE étape (nodeVals[id].fwd est une chaîne : c'est le
//      cas des feuilles comme X/Y quand un notebook injecte leur valeur
//      manuellement dans le log — backward_graph! ne trace rien nativement).
//      Avant ce fix, une valeur-chaîne était silencieusement ignorée et le
//      tableau retombait sur la grille figée de fin d'entraînement : c'est
//      ce qui faisait paraître X/Y "gelés" au clic.
//   3. Repli statique : grille de poids de l'époque courante (currentParamGrid)
//      puis grille finale globale (FULL_GRIDS) si rien d'autre n'est connu.
function fwdContentHTML(nodeId, v) {
  if (isGridVal(v.fwd)) return gridToTableHTML(v.fwd);
  if (v.fwd) return '<pre class="ntbl-raw">' + v.fwd + '</pre>';
  const fallback = currentParamGrid[nodeId] || FULL_GRIDS[nodeId];
  return gridToTableHTML(fallback, 'pas encore calculé');
}
// Quand une cellule de poids est sélectionnée (sel !== null), le contenu
// backward devient cliquable : cliquer dessus affiche la courbe d'évolution
// du gradient de CETTE cellule (voir toggleGradPlot) — remplace l'ancien
// bouton "+ Courbe du gradient" par une interaction directe sur la valeur
// rouge déjà affichée, comme demandé.
function bwdContentHTML(nodeId, v, clickable) {
  const attrs = clickable
    ? ' class="ntbl-raw ntbl-raw-clickable" onclick="toggleGradPlot(\\'' + nodeId + '\\', true)" title="Cliquer pour tracer l\\'évolution du gradient"'
    : ' class="ntbl-raw"';
  if (isGridVal(v.bwd)) return '<div' + attrs + '>' + gridToTableHTML(v.bwd) + '</div>';
  if (v.bwd) return '<pre' + attrs + '>' + v.bwd + '</pre>';
  return null;
}

function openNodeTable(nodeId) {
  const isNew = !openNodeIds.includes(nodeId);
  if (isNew) openNodeIds.push(nodeId);
  const panel = document.getElementById('node-table-panel');
  if (panel) { panel.style.display = 'flex'; panel.style.height = panelHeight + 'px'; }
  renderAllNodeTables();
  if (!isNew) {
    const card = document.querySelector('.ntbl-card[data-node="' + nodeId + '"]');
    if (card) card.scrollIntoView({ behavior: 'smooth', inline: 'nearest' });
  }
}
function closeNodeTable(nodeId) {
  openNodeIds = openNodeIds.filter(id => id !== nodeId);
  delete activeCellByNode[nodeId];
  if (openNodeIds.length === 0) {
    const panel = document.getElementById('node-table-panel');
    if (panel) panel.style.display = 'none';
  } else {
    renderAllNodeTables();
  }
}
function closeAllNodeTables() {
  openNodeIds = [];
  activeCellByNode = {};
  const panel = document.getElementById('node-table-panel');
  if (panel) panel.style.display = 'none';
}
// Rafraîchit le CONTENU des cartes déjà ouvertes (valeurs, graphes) sans
// toucher à la structure DOM du panneau lui-même. Appelé à chaque étape
// forward/backward et à chaque changement d'époque — donc potentiellement
// plusieurs fois par seconde pendant la lecture automatique. Avant ce fix,
// ce chemin appelait renderAllNodeTables() qui reconstruit tout le HTML des
// cartes à chaque fois : ça réinitialisait le défilement des tableaux et,
// pire, interrompait un redimensionnement de carte en cours (resize:both)
// puisque l'élément DOM en cours de glissement disparaissait sous la souris.
// Ici, les éléments <div class="ntbl-card">/<canvas> ne sont JAMAIS recréés
// tant que le nœud reste ouvert — seul leur CONTENU (innerHTML du tableau,
// données du graphe) est mis à jour.
function refreshNodeTable() {
  const bodyEl = document.getElementById('ntbl-body');
  if (!bodyEl) return;
  openNodeIds.forEach(nodeId => {
    const card = bodyEl.querySelector('.ntbl-card[data-node="' + nodeId + '"]');
    if (card) updateCardContent(nodeId, card);
  });
}

// Historique complet (tous les snapshots, pas seulement jusqu'à l'étape
// courante) de la cellule (r,c) du paramètre nodeId — ne s'applique qu'au
// viewer animé (SNAPSHOTS n'existe pas côté trace statique, cf. plus bas).
function buildHistoryFrom(gridKey, nodeId, r, c) {
  if (typeof SNAPSHOTS === 'undefined') return null;
  const xs = [], ys = [];
  SNAPSHOTS.forEach((s, i) => {
    const g = s[gridKey] && s[gridKey][nodeId];
    if (g && g.rows && g.rows[r] && typeof g.rows[r][c] === 'number') {
      xs.push(i); ys.push(g.rows[r][c]);
    }
  });
  return xs.length ? { xs, ys } : null;
}
function buildWeightHistory(nodeId, r, c) { return buildHistoryFrom('params_grid', nodeId, r, c); }
function buildWeightGradHistory(nodeId, r, c) { return buildHistoryFrom('grads_grid', nodeId, r, c); }

function currentGlobalSnapIdx() {
  if (typeof EPOCHS === 'undefined') return -1;
  const epochVal = EPOCHS[currentEpochIdx];
  const snap = epochSnapshots[epochVal] && epochSnapshots[epochVal][currentSnapIdxInEpoch];
  return snap ? SNAPSHOTS.indexOf(snap) : -1;
}

// Crée le graphe s'il n'existe pas encore pour cette clé, sinon met juste à
// jour ses données en place (chart.update()) — pas de destroy/recreate à
// chaque étape, pour ne pas perturber une interaction en cours sur la carte.
function syncHistoryChart(canvas, key, hist, color, fillColor) {
  if (!canvas || !hist) return;
  const curIdx = currentGlobalSnapIdx();
  const ci = hist.xs.indexOf(curIdx);
  const curPoint = ci >= 0 ? [{ x: curIdx, y: hist.ys[ci] }] : [];
  const existing = weightCharts[key];
  if (existing) {
    existing.data.datasets[0].data = hist.xs.map((x, i) => ({ x, y: hist.ys[i] }));
    existing.data.datasets[1].data = curPoint;
    existing.update('none');
    return;
  }
  weightCharts[key] = new Chart(canvas.getContext('2d'), {
    data: {
      datasets: [
        { type: 'line', data: hist.xs.map((x, i) => ({ x, y: hist.ys[i] })),
          borderColor: color, backgroundColor: fillColor,
          borderWidth: 2, pointRadius: 0, tension: .25, fill: true, parsing: false },
        { type: 'scatter', data: curPoint, showLine: false,
          borderColor: 'transparent', backgroundColor: '#e06c75',
          pointRadius: 5, pointHoverRadius: 6, parsing: false }
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false, animation: false, parsing: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { type: 'linear', ticks: { color: '#6b7280', font: { size: 9 }, maxTicksLimit: 6 },
             grid: { color: '#2d3143' } },
        y: { ticks: { color: '#6b7280', font: { size: 9 } }, grid: { color: '#2d3143' } }
      }
    }
  });
}

function destroyChartsFor(nodeId) {
  Object.keys(weightCharts).forEach(k => {
    if (k.startsWith(nodeId + ':')) { weightCharts[k].destroy(); delete weightCharts[k]; }
  });
}

// Bascule l'affichage de la courbe de gradient pour la cellule active d'un
// nœud, indépendamment de la sélection de cellule elle-même (on peut la
// retirer sans désélectionner le poids, comme demandé). Rafraîchissement
// léger (refreshNodeTable) : ne change pas l'ensemble des cartes ouvertes.
function toggleGradPlot(nodeId, show) {
  const sel = activeCellByNode[nodeId];
  if (!sel) return;
  sel.showGrad = show;
  refreshNodeTable();
}

// ── Redimensionnement des cartes (CSS resize:both) : Chart.js n'écoute que
// les évènements 'resize' de la fenêtre, pas le redimensionnement manuel
// d'un conteneur — on relance .resize() sur les graphes de la carte via
// ResizeObserver pour que le canvas suive la taille choisie à la souris.
// Attaché une seule fois à la création de la carte (voir renderAllNodeTables).
function attachCardResizeObserver(card) {
  if (typeof ResizeObserver === 'undefined') return;
  const ro = new ResizeObserver(() => {
    card.querySelectorAll('canvas').forEach(cv => {
      const ch = Chart.getChart ? Chart.getChart(cv) : null;
      if (ch) ch.resize();
    });
  });
  ro.observe(card);
}

// Squelette DOM d'une carte : créé UNE SEULE FOIS par nœud ouvert (voir
// renderAllNodeTables). Son contenu (tableaux, graphes) est ensuite rempli/
// actualisé par updateCardContent, qui ne remplace jamais la carte ni ses
// conteneurs (.ntbl-table-wrap, canvas) — seulement leur innerHTML/données.
function cardShellHTML(nodeId) {
  return '<div class="ntbl-card-header">' +
      '<span>' + nodeId + ' — ' + trunc(SHORT_LABELS[nodeId] || nodeId, 24) + '</span>' +
      '<button class="ntbl-close" onclick="closeNodeTable(\\'' + nodeId + '\\')" title="Fermer">✕</button>' +
    '</div>' +
    '<div class="ntbl-card-body">' +
      '<div class="ntbl-section"><h4>Valeur (forward)<span class="ntbl-plot-hint" data-role="fwd-hint"></span></h4>' +
        '<div class="ntbl-table-wrap" data-role="fwd-table"></div>' +
      '</div>' +
      '<div class="ntbl-section" data-role="bwd-section" style="display:none;">' +
        '<h4>Gradient (backward)<span class="ntbl-plot-hint" data-role="bwd-hint"></span></h4>' +
        '<div class="ntbl-table-wrap" data-role="bwd-table"></div>' +
      '</div>' +
      '<div class="ntbl-plot-slot" data-role="plot-slot" style="display:none;"></div>' +
    '</div>';
}

// Remplit/actualise le CONTENU d'une carte déjà en place dans le DOM. Le
// scroll de chaque tableau est restauré après réécriture de son innerHTML
// pour ne pas donner l'impression que le défilement "saute" à chaque étape.
function updateCardContent(nodeId, card) {
  const v = nodeVals[nodeId] || {};
  const canPlot = !!(NMAP[nodeId] && NMAP[nodeId].is_param) && typeof SNAPSHOTS !== 'undefined';
  const sel = activeCellByNode[nodeId];

  const fwdWrap = card.querySelector('[data-role="fwd-table"]');
  const fwdTop = fwdWrap.scrollTop, fwdLeft = fwdWrap.scrollLeft;
  fwdWrap.innerHTML = fwdContentHTML(nodeId, v);
  fwdWrap.scrollTop = fwdTop; fwdWrap.scrollLeft = fwdLeft;

  const fwdHint = card.querySelector('[data-role="fwd-hint"]');
  if (fwdHint) fwdHint.textContent = canPlot ? ' · cliquer une cellule → évolution' : '';

  if (canPlot) {
    fwdWrap.querySelectorAll('td[data-r]').forEach(td => {
      td.addEventListener('click', () => {
        const r = parseInt(td.dataset.r, 10), c = parseInt(td.dataset.c, 10);
        const cur = activeCellByNode[nodeId];
        if (cur && cur.r === r && cur.c === c) delete activeCellByNode[nodeId];
        else activeCellByNode[nodeId] = { r, c, showGrad: false };
        updateCardContent(nodeId, card);
      });
    });
    if (sel) {
      const activeTd = fwdWrap.querySelector('td[data-r="' + sel.r + '"][data-c="' + sel.c + '"]');
      if (activeTd) activeTd.classList.add('ntbl-cell-active');
    }
  }

  const bwdSection = card.querySelector('[data-role="bwd-section"]');
  const bwdWrap = card.querySelector('[data-role="bwd-table"]');
  const bwdHtml = bwdContentHTML(nodeId, v, canPlot && !!sel);
  if (bwdHtml) {
    bwdSection.style.display = '';
    const bwdTop = bwdWrap.scrollTop, bwdLeft = bwdWrap.scrollLeft;
    bwdWrap.innerHTML = bwdHtml;
    bwdWrap.scrollTop = bwdTop; bwdWrap.scrollLeft = bwdLeft;
    const bwdHint = card.querySelector('[data-role="bwd-hint"]');
    if (bwdHint) bwdHint.textContent = (canPlot && sel) ? ' · cliquer la valeur → évolution' : '';
  } else {
    bwdSection.style.display = 'none';
  }

  const plotSlot = card.querySelector('[data-role="plot-slot"]');
  if (canPlot && sel) {
    plotSlot.style.display = 'flex';
    updatePlotSlot(nodeId, sel, plotSlot);
  } else {
    plotSlot.style.display = 'none';
    if (plotSlot.dataset.structKey) {
      destroyChartsFor(nodeId);
      plotSlot.innerHTML = '';
      plotSlot.dataset.structKey = '';
    }
  }
}

// Contenu de la zone de tracé : la STRUCTURE (quels blocs/canvas existent)
// n'est reconstruite que si la cellule sélectionnée ou l'affichage du
// gradient change ; sinon on se contente d'actualiser les données des
// graphes déjà en place (syncHistoryChart), côte à côte (pas empilés).
function updatePlotSlot(nodeId, sel, plotSlot) {
  const structKey = sel.r + ':' + sel.c + ':' + (sel.showGrad ? 1 : 0);
  if (plotSlot.dataset.structKey !== structKey) {
    destroyChartsFor(nodeId);
    plotSlot.dataset.structKey = structKey;
    plotSlot.innerHTML =
      '<div class="ntbl-plot-block" data-role="val-plot">' +
        '<div class="ntbl-plot-title">' + nodeId + '[' + (sel.r+1) + ',' + (sel.c+1) + '] — valeur</div>' +
        '<canvas></canvas>' +
      '</div>' +
      (sel.showGrad ?
        '<div class="ntbl-plot-block ntbl-plot-grad" data-role="grad-plot-wrap">' +
          '<div class="ntbl-plot-header">' +
            '<span class="ntbl-plot-title">Gradient</span>' +
            '<button class="ntbl-close" onclick="toggleGradPlot(\\'' + nodeId + '\\', false)" title="Retirer la courbe de gradient">✕</button>' +
          '</div>' +
          '<div data-role="grad-plot-body"></div>' +
        '</div>' : '');
  }

  const valHist = buildWeightHistory(nodeId, sel.r, sel.c);
  syncHistoryChart(plotSlot.querySelector('[data-role="val-plot"] canvas'),
                    nodeId + ':' + sel.r + ':' + sel.c + ':val',
                    valHist, '#d4a373', 'rgba(212,163,115,.14)');

  if (sel.showGrad) {
    const gradHist = buildWeightGradHistory(nodeId, sel.r, sel.c);
    const gradBody = plotSlot.querySelector('[data-role="grad-plot-body"]');
    if (gradBody) {
      if (gradHist) {
        let gradCanvas = gradBody.querySelector('canvas');
        if (!gradCanvas) { gradBody.innerHTML = '<canvas></canvas>'; gradCanvas = gradBody.querySelector('canvas'); }
        syncHistoryChart(gradCanvas, nodeId + ':' + sel.r + ':' + sel.c + ':grad',
                          gradHist, '#e06c75', 'rgba(224,108,117,.14)');
      } else if (!gradBody.querySelector('.ntbl-empty')) {
        gradBody.innerHTML = '<div class="ntbl-empty">Aucun gradient enregistré pour cette cellule</div>';
      }
    }
  }
}

// Change l'ENSEMBLE des cartes ouvertes (ajout/suppression de nœuds) : seule
// fonction qui crée ou retire des éléments .ntbl-card. Une carte déjà
// présente n'est jamais recréée — updateCardContent se charge de son
// contenu. C'est ce qui permet de redimensionner une carte ou de faire
// défiler un tableau sans interruption pendant la lecture automatique,
// puisque seule cette fonction (ouverture/fermeture d'un nœud) touche à
// l'existence des éléments DOM ; les rafraîchissements par étape passent
// tous par refreshNodeTable/updateCardContent.
function renderAllNodeTables() {
  const bodyEl  = document.getElementById('ntbl-body');
  const titleEl = document.getElementById('ntbl-title');
  if (!bodyEl) return;
  if (titleEl) {
    titleEl.textContent = openNodeIds.length === 0
      ? 'Cliquez sur un nœud pour explorer ses valeurs'
      : openNodeIds.length + ' tableau' + (openNodeIds.length > 1 ? 'x' : '') +
        ' ouvert' + (openNodeIds.length > 1 ? 's' : '');
  }

  bodyEl.querySelectorAll('.ntbl-card').forEach(card => {
    const nodeId = card.getAttribute('data-node');
    if (!openNodeIds.includes(nodeId)) {
      destroyChartsFor(nodeId);
      card.remove();
    }
  });

  openNodeIds.forEach(nodeId => {
    let card = bodyEl.querySelector('.ntbl-card[data-node="' + nodeId + '"]');
    if (!card) {
      card = document.createElement('div');
      card.className = 'ntbl-card';
      card.setAttribute('data-node', nodeId);
      card.innerHTML = cardShellHTML(nodeId);
      bodyEl.appendChild(card);
      attachCardResizeObserver(card);
    }
    updateCardContent(nodeId, card);
  });
}

// ── Redimensionnement du panneau tableau (poignée du haut) ──────────────────
(function() {
  const panel  = document.getElementById('node-table-panel');
  const handle = document.getElementById('ntblResize');
  if (!panel || !handle) return;
  let resizing = false, startY = 0, startHeight = panelHeight;
  handle.addEventListener('mousedown', e => {
    resizing = true; startY = e.clientY;
    startHeight = parseInt(getComputedStyle(panel).height, 10) || panelHeight;
    e.preventDefault();
  });
  document.addEventListener('mousemove', e => {
    if (!resizing) return;
    panelHeight = Math.min(500, Math.max(100, startHeight - (e.clientY - startY)));
    panel.style.height = panelHeight + 'px';
  });
  document.addEventListener('mouseup', () => { resizing = false; });
})();

let sc = 1, tx = 0, ty = 0, pan = false, px = 0, py = 0;
const wrap = document.getElementById('canvas-wrap');
function applyT() { svgEl.style.transform = 'translate(' + tx + 'px,' + ty + 'px) scale(' + sc + ')'; }
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
"""
end

# ═════════════════════════════════════════════════════════════════════════════
# save_interactive_graph — trace d'une seule passe forward/backward
# ═════════════════════════════════════════════════════════════════════════════
function save_interactive_graph(graph::NeuroGraph, log::ExecutionLog,
                                filepath::String; title="NeuroDSL Trace")
    ns = graph.active_ns
    d  = _graph_viz_json(graph, ns)
    theme = _VIZ_LIGHT_THEME

    log_json = JSON.json([Dict(
        :node   => e[:node],
        :phase  => e[:phase],
        :status => e[:status],
        :val    => e[:value]
    ) for e in log.events])

    graph_css = _viz_graph_css(theme)
    shared_js = _viz_shared_js(theme.marker_colors, theme.sep_stroke)

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
$graph_css
</style>
</head>
<body>

<div id="sidebar">
  <h2>$title</h2>
  <div class="controls">
    <button class="btn" onclick="step(-1)">◀ Prev</button>
    <button class="btn" id="playBtn" onclick="togglePlay()">▶ Play</button>
    <button class="btn" onclick="step(1)">Next ▶</button>
    <button class="btn" onclick="reset()">↺ Reset</button>
  </div>
  <div id="status">Step : 0 / 0</div>
  <div id="log-panel"></div>
</div>

<div id="graph-area">
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
  $(_viz_table_panel_html())
</div>

<script>
const NODES_RAW = $(d.nodes_json);
const EDGES     = $(d.edges_json);
const LOG       = $log_json;
const FULL_GRIDS = $(d.full_grids_json);
const INIT_VALS = $(d.init_json);
const FULL_VALS = $(d.full_json);
const FORMULAS  = $(d.formulas_json);
const SHORT_LABELS = $(d.short_labels_json);

$shared_js

buildGraphSVG();
const nodeVals = Object.fromEntries(
  NODES.map(n => [n.id, { fwd: FULL_VALS[n.id] || '?', bwd: '' }])
);

let step_i = -1, playing = false, timer = null;

function step(dir) {
  step_i = Math.max(-1, Math.min(LOG.length - 1, step_i + dir));
  updateUI();
}
function updateUI() {
  document.getElementById('status').textContent = 'Step : ' + (step_i + 1) + ' / ' + LOG.length;
  renderEventsUpTo(LOG, step_i);
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
function reset() {
  step_i = -1;
  playing = false;
  const btn = document.getElementById('playBtn');
  btn.textContent = '▶ Play';
  if (timer) clearInterval(timer);
  updateUI();
}

requestAnimationFrame(zoomFit);
</script>
</body>
</html>
"""

    write(filepath, html)
    println("✅ Interactive Trace exported → $filepath")
end

# ═════════════════════════════════════════════════════════════════════════════
# save_interactive_graph_animated — trace d'entraînement avec slider d'époques
# ═════════════════════════════════════════════════════════════════════════════
function save_interactive_graph_animated(
        graph::NeuroGraph,
        snapshots::Vector{TrainingSnapshot},
        filepath::String;
        title="NeuroDSL Training Trace",
        losses::Vector{Float32}=Float32[])

    ns = graph.active_ns
    d  = _graph_viz_json(graph, ns)
    theme = _VIZ_DARK_THEME

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
      :params_full => Dict(string(k) => format_tensor_preview(v) for (k,v) in s.params),
      :params_grid => Dict(string(k) => format_tensor_grid(v) for (k,v) in s.params),
      :grads_grid  => Dict(string(k) => format_tensor_grid(v) for (k,v) in s.grads)
  ) for s in snapshots])

    losses_json = JSON.json([Float64(l) for l in losses])

    epoch_to_snapshots = Dict{Int,Vector{TrainingSnapshot}}()
    for s in snapshots
        push!(get!(epoch_to_snapshots, s.epoch, TrainingSnapshot[]), s)
    end
    epochs = sort(collect(keys(epoch_to_snapshots)))
    epochs_json = JSON.json(epochs)

    graph_css = _viz_graph_css(theme)
    shared_js = _viz_shared_js(theme.marker_colors, theme.sep_stroke)

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
#topbar {
  background:#1e2030; border-bottom:1px solid #2d3143;
  padding:10px 20px; display:flex; align-items:center; gap:16px;
  flex-shrink:0; flex-wrap:wrap;
}
#topbar h1 { font-size:0.95rem; font-weight:700; color:#d4a373; white-space:nowrap; }
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
#main { display:flex; flex:1; overflow:hidden; }
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
#log-panel {
  flex:1; overflow-y:auto; font-family:'Consolas',monospace; font-size:10px;
  border:1px solid #2d3143; padding:6px; background:#12141c; border-radius:6px;
  color:#d1d5db;
}
.log-entry { padding:2px 5px; border-radius:3px; margin-bottom:2px; line-height:1.5; }
.log-fwd { background:#1a2530; color:#7aa2f7; }
.log-bwd { background:#2e1a1a; color:#e06c75; }
#epoch-badge {
  position:absolute; top:12px; left:12px;
  background:#1e2030; border:1px solid #3b3f53;
  border-radius:8px; padding:8px 14px; font-size:0.8rem;
  pointer-events:none;
}
#epoch-badge .ep  { color:#d4a373; font-weight:700; font-size:1rem; }
#epoch-badge .los { color:#6a9b6a; font-size:0.75rem; }
$graph_css
</style>
</head>
<body>

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
    <button class="btn" onclick="reset()">↺ Reset</button>
    <span id="stepStatus">–</span>
  </div>
</div>

<div id="main">
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
  <div id="graph-area">
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
    $(_viz_table_panel_html())
  </div>
</div>

<script>
const NODES_RAW    = $(d.nodes_json);
const EDGES        = $(d.edges_json);
const SNAPSHOTS    = $snaps_json;
const LOSSES       = $losses_json;
const FULL_GRIDS   = $(d.full_grids_json);
const INIT_VALS    = $(d.init_json);
const FULL_VALS    = $(d.full_json);
const FORMULAS     = $(d.formulas_json);
const SHORT_LABELS = $(d.short_labels_json);
const EPOCHS       = $epochs_json;

$shared_js

buildGraphSVG();
const nodeVals = Object.fromEntries(NODES.map(n => [n.id, { fwd: '', bwd: '' }]));

const epochSnapshots = {};
EPOCHS.forEach(e => { epochSnapshots[e] = SNAPSHOTS.filter(s => s.epoch == e); });

let currentEpochIdx = 0;
let currentSnapIdxInEpoch = 0;
let stepIdx = -1;
let playing = false;
let timer = null;

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

let lossChart = null;
function buildLossChart() {
  if (!LOSSES.length) return;
  const ctx2 = document.getElementById('lossChart').getContext('2d');

  // Une perte qui descend sur plusieurs ordres de grandeur (ex. 300 -> 0.01)
  // s'écrase visuellement sur une échelle linéaire : la chute initiale domine
  // tout le graphe et la descente réelle des dernières étapes paraît plate.
  // Passage automatique en échelle logarithmique quand l'amplitude le justifie.
  const positiveLosses = LOSSES.filter(l => l > 0);
  const lossMin = positiveLosses.length ? Math.min(...positiveLosses) : 0;
  const lossMax = positiveLosses.length ? Math.max(...positiveLosses) : 0;
  const useLogScale = lossMin > 0 && (lossMax / lossMin) > 20;

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
        y:{ type: useLogScale ? 'logarithmic' : 'linear',
            ticks:{color:'#6b7280',font:{size:9}},
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

function renderStep() {
  const epochVal = EPOCHS[currentEpochIdx];
  const snap = epochSnapshots[epochVal][currentSnapIdxInEpoch];
  if (!snap) return;
  currentParamFull = snap.params_full || {};
  currentParamGrid = snap.params_grid || {};
  document.getElementById('badge-epoch').textContent = 'Epoch ' + snap.epoch + '  |  iter ' + snap.iter;
  document.getElementById('badge-loss').textContent = 'Loss : ' + snap.loss.toFixed(6);
  const total = snap.events.length;
  document.getElementById('stepStatus').textContent =
    stepIdx < 0 ? '0 / ' + total : (stepIdx+1) + ' / ' + total;

  renderEventsUpTo(snap.events, stepIdx);
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
function reset() {
  // Remettait seulement l'étape courante à -1 : l'époque et l'itération
  // restaient sur leur dernière position visitée, ce qui ne correspond pas
  // à ce qu'on attend d'un "Reset" complet.
  stepIdx = -1;
  currentEpochIdx = 0;
  currentSnapIdxInEpoch = 0;
  slider.value = 0;
  epochInput.value = EPOCHS[0];
  playing = false;
  const btn = document.getElementById('playBtn');
  btn.textContent = '▶ Play';
  btn.classList.add('primary');
  if (timer) clearInterval(timer);
  updateLossMarker(0);
  renderStep();
}

// ── Redimensionnement sidebar (largeur) ──
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

// ── Redimensionnement hauteur courbe de perte ──
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
