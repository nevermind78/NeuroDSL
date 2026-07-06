# ══════════════════════════════════════════════════════════════════════════════
# serialization.jl — Sauvegarde/chargement d'un NeuroGraph sur disque
#
# Sans rapport avec `checkpoint.jl` (qui gère le recalcul d'activations EN RAM
# pendant un seul backward, pas la persistance entre sessions Julia).
#
# Format : deux fichiers par appel, "<prefix>.json" (manifeste, dépendance JSON
# déjà déclarée) + "<prefix>.bin" (tas d'octets Float32 brut, `Base.write`/
# `Base.read!` -- même pattern que `_oracle_snapshot_bytes` de
# `test/test_oracle_replay.jl`, zéro nouvelle dépendance).
#
# Portée volontaire (v1) :
#   - Seules les valeurs des nœuds FEUILLES (paramètres/entrées, i.e. sans
#     règle) sont persistées. Tout nœud de sortie de règle est reconstruit via
#     `addrule!` seul (valid=false) -- l'appelant doit rappeler `demand!` après
#     chargement. Évite toute ambiguïté sur la fraîcheur d'une activation en
#     cache ; cohérent avec la philosophie du moteur ("tout ce qui est dérivé
#     est bon marché à recalculer").
#   - `gradient`, `valid`, `backwarded` ne sont jamais persistés (état
#     transitoire -- le chargement traite le graphe comme fraîchement
#     reconstruit).
#   - `aux_data` n'est pas persisté (Dict{Symbol,Any} ouvert, aucun site
#     d'écriture actuel dans src/ -- limitation connue, pas gérée en v1).
#   - `on_change` doit être `nothing` au moment de la sauvegarde -- erreur
#     explicite sinon (aucun call site de src/ n'assigne jamais une vraie
#     fonction ici ; on refuse de corrompre silencieusement une closure).
# ══════════════════════════════════════════════════════════════════════════════

"""
    AdamWState

État d'optimiseur AdamW, keyé par SYMBOLE de paramètre (pas par position, à la
différence de l'usage actuel dans `notebook/notebook.ipynb`, qui apparie
`m1s`/`m2s` à `params(g;...)` par position -- fragile car l'ordre d'itération
d'un `Dict` n'est pas garanti stable). `t` est le compteur de pas AdamW
(correction de biais), jamais persisté ailleurs dans le code actuel.
"""
struct AdamWState
    t  :: Int
    m1 :: Dict{Symbol, Array{Float32}}
    m2 :: Dict{Symbol, Array{Float32}}
end

"""
    extend_adamw_state!(state, g, ns, new_param_syms) -> AdamWState

Ajoute des moments à zéro pour des paramètres nouvellement créés (ex. après
`insert_block!`, `src/graph_surgery.jl`) -- garde les moments existants
inchangés (keyé par symbole, pas par position, donc aucun risque de décalage).
Mutation en place des `Dict`s internes ; `state` lui-même reste le même objet.
"""
function extend_adamw_state!(state::AdamWState, g::NeuroGraph, ns::Symbol, new_param_syms)
    for sym in new_param_syms
        haskey(state.m1, sym) && continue
        shape = size(g.nodes[ns][sym].value)
        state.m1[sym] = zeros(Float32, shape...)
        state.m2[sym] = zeros(Float32, shape...)
    end
    return state
end

const _ATOM_TYPE_NAMES = Dict(Datom => "Datom", Quantom => "Quantom")
const _ATOM_TYPE_FROM_NAME = Dict("Datom" => Datom, "Quantom" => Quantom)

function _atom_type_name(t::Type{<:NeurAtom})
    haskey(_ATOM_TYPE_NAMES, t) || error("❌ save_graph! : atom_type inconnu : $t")
    return _ATOM_TYPE_NAMES[t]
end

function _atom_type_from_name(name::AbstractString)
    haskey(_ATOM_TYPE_FROM_NAME, name) || error("❌ load_graph! : atom_type inconnu dans le manifeste : $name")
    return _ATOM_TYPE_FROM_NAME[name]
end

_device_tag(dev) = dev isa Backend.CUDADevice ? "cuda" : "cpu"

"""
    save_graph!(g, ns, path_prefix; opt_state=nothing)

Écrit `"<path_prefix>.json"` + `"<path_prefix>.bin"`. Voir le banner du fichier
pour la portée exacte (valeurs de feuilles uniquement, pas de gradients/valid).
"""
function save_graph!(g::NeuroGraph, ns::Symbol, path_prefix::AbstractString;
                      opt_state::Union{Nothing,AdamWState}=nothing)
    haskey(g.nodes, ns) || error("❌ save_graph! : namespace :$ns introuvable dans le graphe")

    nodes_meta = Vector{Dict{String,Any}}()
    rules_meta = Vector{Dict{String,Any}}()
    opt_meta = nothing

    open(path_prefix * ".bin", "w") do bin
        for (sym, nd) in g.nodes[ns]
            nd.on_change === nothing ||
                error("❌ save_graph! : le nœud :$sym a un `on_change` non-nothing -- " *
                      "sérialisation d'une closure non supportée")
            is_leaf = !haskey(g.rules[ns], sym)
            meta = Dict{String,Any}(
                "name"      => string(sym),
                "is_param"  => nd.is_param,
                "atom_type" => _atom_type_name(nd.atom_type),
                "watchers"  => string.(nd.watchers),
            )
            if is_leaf
                nd.value === nothing &&
                    error("❌ save_graph! : le nœud feuille :$sym n'a pas de valeur (jamais `set!`)")
                arr = Array(nd.value)
                offset = position(bin)
                nbytes = write(bin, arr)
                meta["shape"]       = collect(size(arr))
                meta["blob_offset"] = offset
                meta["blob_nbytes"] = nbytes
            end
            push!(nodes_meta, meta)
        end

        for (_, rule) in g.rules[ns]
            push!(rules_meta, Dict{String,Any}(
                "output"    => string(rule.output),
                "inputs"    => string.(rule.inputs),
                "op"        => string(rule.op),
                "attrs"     => Dict(string(k) => v for (k, v) in rule.attrs),
                "atom_type" => _atom_type_name(rule.atom_type),
            ))
        end

        if opt_state !== nothing
            params_meta = Vector{Dict{String,Any}}()
            for (psym, m1arr) in opt_state.m1
                haskey(opt_state.m2, psym) ||
                    error("❌ save_graph! : AdamWState.m2 n'a pas d'entrée pour :$psym")
                m2arr = opt_state.m2[psym]
                size(m1arr) == size(m2arr) ||
                    error("❌ save_graph! : m1/m2 de forme différente pour :$psym")
                off1 = position(bin); n1 = write(bin, m1arr)
                off2 = position(bin); n2 = write(bin, m2arr)
                push!(params_meta, Dict{String,Any}(
                    "name" => string(psym), "shape" => collect(size(m1arr)),
                    "m1_offset" => off1, "m1_nbytes" => n1,
                    "m2_offset" => off2, "m2_nbytes" => n2,
                ))
            end
            opt_meta = Dict{String,Any}("t" => opt_state.t, "params" => params_meta)
        end
    end

    manifest = Dict{String,Any}(
        "format_version" => 1,
        "namespace"      => string(ns),
        "device"         => _device_tag(g.device),
        "nodes"          => nodes_meta,
        "rules"          => rules_meta,
        "optimizer"      => opt_meta,
    )
    open(path_prefix * ".json", "w") do io
        JSON.print(io, manifest)
    end
    return nothing
end

"""
    load_graph!(g, ns, path_prefix; overwrite=false) -> Union{Nothing,AdamWState}

Reconstruit le namespace `ns` de `g` depuis `"<path_prefix>.json"`/`.bin`.
Erreur si `ns` existe déjà et contient des nœuds, sauf `overwrite=true`.
"""
function load_graph!(g::NeuroGraph, ns::Symbol, path_prefix::AbstractString; overwrite::Bool=false)
    if haskey(g.nodes, ns) && !isempty(g.nodes[ns]) && !overwrite
        error("❌ load_graph! : le namespace :$ns existe déjà dans le graphe " *
              "(utilisez overwrite=true pour l'écraser)")
    end

    manifest = JSON.parsefile(path_prefix * ".json")
    manifest["format_version"] == 1 ||
        error("❌ load_graph! : format_version non supporté : $(manifest["format_version"])")

    manifest_device = get(manifest, "device", nothing)
    if manifest_device !== nothing && manifest_device != _device_tag(g.device)
        @warn "load_graph! : le manifeste a été sauvegardé pour le device \"$manifest_device\", " *
              "chargement vers \"$(_device_tag(g.device))\" -- purement informatif, `set!` gère la conversion."
    end

    _ensure_namespace!(g, ns)
    empty!(g.nodes[ns]); empty!(g.rules[ns])
    g._topo_cache[ns] = nothing
    g._consumers_cache[ns] = nothing

    io = open(path_prefix * ".bin", "r")
    opt_state = nothing
    try
        leaf_syms = Set{Symbol}()
        for nm in manifest["nodes"]
            haskey(nm, "shape") || continue
            sym = Symbol(nm["name"])
            push!(leaf_syms, sym)
            shape = Tuple(Int.(nm["shape"]))
            arr = Array{Float32}(undef, shape...)
            seek(io, nm["blob_offset"])
            read!(io, arr)
            set!(g, sym, arr; is_param=nm["is_param"],
                 atom_type=_atom_type_from_name(nm["atom_type"]), namespace=ns)
            g.nodes[ns][sym].watchers = Symbol.(nm["watchers"])
        end

        for rm in manifest["rules"]
            attrs = Dict{Symbol,Any}(Symbol(k) => v for (k, v) in rm["attrs"])
            rule = GraphRule(Symbol(rm["output"]), Symbol.(rm["inputs"]), Symbol(rm["op"]);
                             attrs=attrs, namespace=ns, atom_type=_atom_type_from_name(rm["atom_type"]))
            addrule!(g, rule)
        end

        for nm in manifest["nodes"]
            sym = Symbol(nm["name"])
            sym in leaf_syms && continue
            haskey(g.nodes[ns], sym) ||
                error("❌ load_graph! : nœud :$sym absent après reconstruction des règles (manifeste incohérent)")
            nm["is_param"] &&
                error("❌ load_graph! : nœud de sortie de règle :$sym marqué is_param=true (manifeste incohérent)")
            g.nodes[ns][sym].watchers = Symbol.(nm["watchers"])
        end

        if manifest["optimizer"] !== nothing
            om = manifest["optimizer"]
            m1 = Dict{Symbol,Array{Float32}}(); m2 = Dict{Symbol,Array{Float32}}()
            for pm in om["params"]
                psym = Symbol(pm["name"])
                shape = Tuple(Int.(pm["shape"]))
                a1 = Array{Float32}(undef, shape...); seek(io, pm["m1_offset"]); read!(io, a1)
                a2 = Array{Float32}(undef, shape...); seek(io, pm["m2_offset"]); read!(io, a2)
                m1[psym] = a1; m2[psym] = a2
            end
            opt_state = AdamWState(om["t"], m1, m2)
        end
    finally
        close(io)
    end
    return opt_state
end

"""
    save_all_graph!(g, dir_prefix; opt_states=Dict())

Sauvegarde TOUS les namespaces de `g` dans `dir_prefix` (un couple `.json`/
`.bin` par namespace, plus `"graph_manifest.json"` pour la liste des
namespaces et `active_ns`). `opt_states` associe optionnellement un
`AdamWState` par namespace.
"""
function save_all_graph!(g::NeuroGraph, dir_prefix::AbstractString;
                          opt_states::Dict{Symbol,AdamWState}=Dict{Symbol,AdamWState}())
    mkpath(dir_prefix)
    for ns in namespaces(g)
        save_graph!(g, ns, joinpath(dir_prefix, string(ns)); opt_state=get(opt_states, ns, nothing))
    end
    open(joinpath(dir_prefix, "graph_manifest.json"), "w") do io
        JSON.print(io, Dict("namespaces" => string.(namespaces(g)), "active_ns" => string(g.active_ns)))
    end
    return nothing
end

"""
    load_all_graph!(g, dir_prefix; overwrite=false) -> Dict{Symbol,AdamWState}

Charge tous les namespaces listés dans `"<dir_prefix>/graph_manifest.json"`,
restaure `active_ns`, et retourne les `AdamWState` trouvés (namespaces sans
optimiseur sauvegardé simplement absents du dict retourné).
"""
function load_all_graph!(g::NeuroGraph, dir_prefix::AbstractString; overwrite::Bool=false)
    manifest = JSON.parsefile(joinpath(dir_prefix, "graph_manifest.json"))
    result = Dict{Symbol,AdamWState}()
    for ns_str in manifest["namespaces"]
        ns = Symbol(ns_str)
        opt = load_graph!(g, ns, joinpath(dir_prefix, ns_str); overwrite=overwrite)
        opt !== nothing && (result[ns] = opt)
    end
    activate!(g, Symbol(manifest["active_ns"]))
    return result
end
