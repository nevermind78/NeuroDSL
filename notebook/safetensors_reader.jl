# ══════════════════════════════════════════════════════════════════════════════
# safetensors_reader.jl — lecteur minimal, écrit à la main, du format
# `.safetensors` (poids HuggingFace) -- AUCUNE nouvelle dépendance : le format
# est {8 octets uint64 little-endian = longueur N de l'en-tête} + {N octets
# d'en-tête JSON} + {octets bruts des tenseurs, contigus, décalages donnés par
# l'en-tête} -- JSON est déjà une dépendance du projet (serialization.jl
# l'utilise déjà pour le format de checkpoint propre à NeuroDSL). Même
# discipline que serialization.jl : lecture directe d'octets, pas de
# bibliothèque tierce pour le format de fichier lui-même.
#
# Référence du format (spec publique HuggingFace safetensors) :
#   octets [0:8)   : N, uint64 little-endian, longueur de l'en-tête JSON
#   octets [8:8+N) : en-tête JSON UTF-8 -- dict {nom_tenseur => {dtype, shape,
#                    data_offsets:[start,end]}}, plus une clé optionnelle
#                    "__metadata__" (pas un tenseur, à ignorer)
#   octets [8+N:.) : données brutes de chaque tenseur, décalage RELATIF au
#                    début de cette zone (donc décalage ABSOLU = 8+N+start)
#
# dtypes couverts : BF16, F16, F32 (ceux qu'on a réellement besoin de lire
# pour les checkpoints ciblés cette session -- pas une lecture générique de
# TOUS les dtypes safetensors possibles).
# ══════════════════════════════════════════════════════════════════════════════

using JSON

"""
    SafetensorsFile

`header::Dict{String,Any}` -- en-tête JSON brut (sans `__metadata__`).
`data_start::Int` -- décalage absolu (1-indexé, convention Julia) du début de
la zone de données, i.e. `8 + N + 1`.
`path::String` -- fichier source, ré-ouvert à chaque lecture de tenseur (pas
de descripteur de fichier gardé ouvert entre deux appels -- plus simple, coût
négligeable face à la taille des tenseurs eux-mêmes).
"""
struct SafetensorsFile
    header::Dict{String,Any}
    data_start::Int
    path::String
end

"""
    open_safetensors(path) -> SafetensorsFile

Lit uniquement l'en-tête JSON (quelques dizaines de Ko même pour un modèle de
plusieurs Go) -- ne charge AUCUNE donnée de tenseur en mémoire.
"""
function open_safetensors(path::AbstractString)
    io = open(path, "r")
    n = read(io, UInt64)  # little-endian natif sur toute plateforme x86/ARM courante
    header_bytes = read(io, Int(n))
    close(io)
    header = JSON.parse(String(header_bytes))
    delete!(header, "__metadata__")
    return SafetensorsFile(header, 8 + Int(n) + 1, String(path))
end

const _DTYPE_BYTES = Dict("F32" => 4, "F16" => 2, "BF16" => 2, "I64" => 8, "I32" => 4)

"""
    bf16_to_f32(u::UInt16) -> Float32

bfloat16 -> Float32 : bf16 EST les 16 bits de poids fort d'un Float32 (même
exposant/mantisse tronquée, pas de conversion arithmétique -- juste un
décalage de bits). `reinterpret` direct, aucune bibliothèque nécessaire.
"""
bf16_to_f32(u::UInt16) = reinterpret(Float32, UInt32(u) << 16)

"""
    read_tensor(st::SafetensorsFile, name::String) -> Array{Float32}

Lit et convertit UN tenseur en Float32, avec la forme PyTorch d'origine
(row-major) réinterprétée correctement en tableau Julia (column-major) --
PyTorch stocke un tenseur de forme (d1,...,dn) en row-major ; relire les
mêmes octets bruts avec `reshape` en Julia (column-major) donnerait la
TRANSPOSÉE de ce qu'on veut pour un tenseur 2D -- corrigé ici en inversant
l'ordre des dimensions à `reshape` puis en transposant le résultat 2D. Pour
un tenseur 1D (biais, gamma de norme), aucune correction nécessaire.
"""
function read_tensor(st::SafetensorsFile, name::String)
    haskey(st.header, name) || error("safetensors: tenseur introuvable : $name")
    info = st.header[name]
    dtype = info["dtype"]::String
    shape = Int.(info["shape"])
    offs  = Int.(info["data_offsets"])  # [start, end) relatifs, en octets
    nbytes = offs[2] - offs[1]
    haskey(_DTYPE_BYTES, dtype) || error("safetensors: dtype non couvert : $dtype (tenseur $name)")
    nelem = isempty(shape) ? 1 : prod(shape)
    nelem * _DTYPE_BYTES[dtype] == nbytes ||
        error("safetensors: taille incohérente pour $name : $nbytes octets, $nelem éléments en $dtype")

    io = open(st.path, "r")
    seek(io, st.data_start - 1 + offs[1])
    raw = read(io, nbytes)
    close(io)

    vals = if dtype == "F32"
        reinterpret(Float32, raw)
    elseif dtype == "F16"
        Float32.(reinterpret(Float16, raw))
    elseif dtype == "BF16"
        bf16_to_f32.(reinterpret(UInt16, raw))
    else
        error("dtype $dtype non atteignable ici (déjà filtré plus haut)")
    end
    vals = Vector{Float32}(vals)

    isempty(shape) && return vals  # scalaire
    length(shape) == 1 && return vals
    length(shape) == 2 && return permutedims(reshape(vals, reverse(shape)...), (2, 1))
    error("read_tensor : rang $(length(shape)) non géré (seuls rang 1 et 2 attendus pour ce chargeur)")
end

"""
    tensor_names(st::SafetensorsFile) -> Vector{String}
"""
tensor_names(st::SafetensorsFile) = collect(keys(st.header))

# ══════════════════════════════════════════════════════════════════════════════
# Support des checkpoints "sharded" (plusieurs fichiers .safetensors + un
# model.safetensors.index.json qui mappe chaque nom de tenseur à son fichier
# -- HuggingFace découpe automatiquement au-delà d'une certaine taille, ex.
# Qwen2.5-7B-Instruct en 4 shards). Additif : ne touche à rien de ce qui
# précède (le lecteur 1-fichier reste utilisé tel quel pour les checkpoints
# non shardés, ex. Qwen2.5-1.5B-Instruct).
# ══════════════════════════════════════════════════════════════════════════════

"""
    ShardedSafetensorsFile

`weight_map::Dict{String,String}` -- nom de tenseur -> nom de fichier shard
(tel que lu dans l'index HuggingFace).
`shards::Dict{String,SafetensorsFile}` -- fichier shard -> en-tête déjà parsé
(un seul `open_safetensors` par shard, pas par tenseur).
"""
struct ShardedSafetensorsFile
    weight_map :: Dict{String,String}
    shards     :: Dict{String,SafetensorsFile}
end

"""
    open_safetensors_sharded(index_path) -> ShardedSafetensorsFile

Lit `model.safetensors.index.json` (clé `"weight_map"`, format HuggingFace
standard) puis ouvre l'en-tête de chaque shard référencé UNE fois (pas à
chaque lecture de tenseur) -- même discipline que `open_safetensors` : aucune
donnée de tenseur chargée ici, seulement les en-têtes JSON.
"""
function open_safetensors_sharded(index_path::AbstractString)
    idx = JSON.parsefile(index_path)
    weight_map = Dict{String,String}(String(k) => String(v) for (k, v) in idx["weight_map"])
    dir = dirname(index_path)
    shard_files = unique(values(weight_map))
    shards = Dict{String,SafetensorsFile}(
        f => open_safetensors(joinpath(dir, f)) for f in shard_files
    )
    return ShardedSafetensorsFile(weight_map, shards)
end

function read_tensor(st::ShardedSafetensorsFile, name::String)
    haskey(st.weight_map, name) || error("safetensors (sharded): tenseur introuvable : $name")
    return read_tensor(st.shards[st.weight_map[name]], name)
end

tensor_names(st::ShardedSafetensorsFile) = collect(keys(st.weight_map))
haskey_tensor(st::ShardedSafetensorsFile, name::String) = haskey(st.weight_map, name)
haskey_tensor(st::SafetensorsFile, name::String) = haskey(st.header, name)

"""
    open_any_safetensors(model_dir) -> SafetensorsFile or ShardedSafetensorsFile

Détecte automatiquement le format présent dans `model_dir` : si
`model.safetensors.index.json` existe, ouvre en mode sharded ; sinon retombe
sur le fichier unique `model.safetensors` (comportement historique inchangé).
"""
function open_any_safetensors(model_dir::AbstractString)
    idx_path = joinpath(model_dir, "model.safetensors.index.json")
    if isfile(idx_path)
        return open_safetensors_sharded(idx_path)
    end
    return open_safetensors(joinpath(model_dir, "model.safetensors"))
end
