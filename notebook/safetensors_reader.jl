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
