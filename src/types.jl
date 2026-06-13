
# CtxStore - alias unique pour le dict de contexte
const CtxStore = Dict{Symbol, Any}

abstract type NeurAtom end
abstract type Datom   <: NeurAtom end
abstract type Quantom <: NeurAtom end

# Structure avec deux paramètres de type pour la performance interne
# Structure avec un seul paramètre de type pour la performance interne.
# La valeur est toujours soit nothing, soit un AbstractArray{Float32}.
mutable struct GraphNode{F <: AbstractFloat}
    name      :: Symbol
    value     :: Union{Nothing, AbstractArray{F}}
    gradient  :: Union{Nothing, AbstractArray{F}}
    valid     :: Bool
    backwarded :: Bool   
    atom_type :: Type{<:NeurAtom}
    is_param  :: Bool
    namespace :: Symbol
    aux_data  :: Dict{Symbol,Any}
    watchers::Vector{Symbol}          # nœuds qui m'observent
    on_change::Union{Nothing,Function} # callback quand invalidé
end

function GraphNode(name::Symbol, value::Union{Nothing, AbstractArray};
                   atom_type = Quantom, is_param = false,
                   namespace = :default, valid = true)
    F_type = (value isa AbstractArray && eltype(value) <: AbstractFloat) ? eltype(value) : Float32
    return GraphNode{F_type}(name, value, nothing, valid, false,  # backwarded = false
                             atom_type, is_param, namespace, Dict{Symbol,Any}(),
                             Symbol[], nothing)
end

is_backpropable(n::GraphNode) = n.atom_type <: Quantom

struct GraphRule
    output    :: Symbol
    inputs    :: Vector{Symbol}
    op        :: Symbol
    attrs     :: Dict{Symbol, Any}
    namespace :: Symbol
    atom_type :: Type{<:NeurAtom}
end

GraphRule(output, inputs, op; attrs=Dict{Symbol,Any}(), namespace=:default, atom_type=Quantom) =
    GraphRule(output, inputs, op, attrs, namespace, atom_type)

# On utilise GraphNode{Float32} (sans spécifier le 2ème paramètre)
# pour permettre la covariance dans le dictionnaire.
mutable struct NeuroGraph
    nodes         :: Dict{Symbol, Dict{Symbol, GraphNode{Float32}}}

    rules         :: Dict{Symbol, Dict{Symbol, GraphRule}}
    grad_registry :: Dict{Symbol, Function}
    _topo_cache   :: Dict{Symbol, Union{Nothing, Vector{Symbol}}}
    active_ns     :: Symbol
    device        :: Union{Backend.CPUDevice, Backend.CUDADevice}
end

function NeuroGraph(; namespace::Symbol = :default, device = Backend.active_device())
    g = NeuroGraph(
        Dict{Symbol, Dict{Symbol, GraphNode{Float32}}}(),  # ← retour à l'original
        Dict{Symbol, Dict{Symbol, GraphRule}}(),
        Dict{Symbol, Function}(),
        Dict{Symbol, Union{Nothing, Vector{Symbol}}}(),
        namespace, device)
    _ensure_namespace!(g, namespace)
    return g
end
# Structure pour enregistrer la trace d'exécution
mutable struct ExecutionLog
    events::Vector{Dict{Symbol, Any}}
end
ExecutionLog() = ExecutionLog([])

# Fonction utilitaire pour enregistrer un événement
function log_event!(log::ExecutionLog, node, phase, status, value=nothing)
    push!(log.events, Dict(
        :node => node,
        :phase => phase,      # "forward" ou "backward"
        :status => status,    # "starting" ou "finished"
        :value => value,      # Résumé de la valeur
        :time => now()
    ))
end