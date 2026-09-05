
"""
    GradPool

Pool de tampons pour les gradients/scratch intermédiaires du backward,
réutilisés À L'INTÉRIEUR d'une seule passe de `backward_graph!` (pas entre
passes -- voir l'expérience rejetée de persistance dans `aux_data`, qui
transformait de la mémoire transitoire en résidence permanente et empirait
les épisodes forward-seuls). Conçu pour éliminer par construction, pas par
discipline, les trois failles trouvées par audit sur une première ébauche
(pool dynamique + comptage de références par `objectid`) :

- **Pas de compteur de références du tout.** Un buffer prêté a exactement
  UN propriétaire à tout instant (une variable locale, un `nd.gradient`, ou
  un slot scratch) ; la propriété est TRANSFÉRÉE d'un possesseur à l'autre,
  jamais dupliquée. `lent` (ci-dessous) est le seul registre, détenu PAR le
  pool -- pas un dict externe indexé par `objectid` qui pourrait hériter un
  compte fantôme après un `release!`/`acquire!` sur le même objet physique.
- **Type distinct de `BufferPool`** (`src/liveness.jl`) : aucune fonction
  partagée (`acquire!`/`release!` restent réservées à `BufferPool`). Passer
  un `GradPool` là où un `BufferPool` est attendu est un `MethodError` sûr,
  pas un bug latent -- crucial parce qu'utiliser `BufferPool`/`acquire!`
  pour les gradients armerait par erreur `_POOLED_EXECUTION_SEEN[]`
  (le kill-switch de `CTX_REBUILD`, backward.jl) et ferait régresser
  `train_step` d'environ 125 Mo à ~142 Mo.
- **`grad_release!` n'accepte que les buffers nés dans CE pool** (présents
  dans `lent`) -- tout le reste (buffer étranger, valeur forward, résidu
  d'un `ctx_store` persistant threadé par l'appelant) est un no-op compté
  dans `n_foreign`, jamais injecté dans les listes libres. Ferme à la fois
  la fuite `prune_frozen` (un gradient calculé puis jamais assigné n'est
  simplement jamais entré dans le pool, voir `_gbuf!`) et le risque de
  "release étranger" (corruption d'un buffer encore référencé ailleurs).
"""
mutable struct GradPool
    buckets  :: Dict{Tuple, Vector{AbstractArray{Float32}}}
    lent     :: IdDict{AbstractArray{Float32}, Nothing}
    device   :: Union{Backend.CPUDevice, Backend.CUDADevice}
    n_alloc  :: Int
    n_hits   :: Int
    n_foreign:: Int
end

GradPool(dev::Union{Backend.CPUDevice, Backend.CUDADevice}) =
    GradPool(Dict{Tuple, Vector{AbstractArray{Float32}}}(), IdDict{AbstractArray{Float32}, Nothing}(),
             dev, 0, 0, 0)

"""
    grad_acquire!(gp, shape) -> AbstractArray{Float32}

Emprunte un tampon de forme `shape`, l'enregistre comme prêté. N'arme
JAMAIS `_POOLED_EXECUTION_SEEN[]` (contrairement à `BufferPool.acquire!`) --
voir la docstring de `GradPool`.
"""
function grad_acquire!(gp::GradPool, shape::Tuple)
    key = shape
    if haskey(gp.buckets, key) && !isempty(gp.buckets[key])
        buf = pop!(gp.buckets[key])
        gp.n_hits += 1
    else
        buf = Backend.zeros32(gp.device, shape...)
        gp.n_alloc += 1
    end
    gp.lent[buf] = nothing
    return buf
end

"""
    grad_release!(gp, buf) -> Bool

Rend `buf` au pool s'il en est bien issu (présent dans `lent`) -- sinon
no-op compté (`n_foreign`), jamais ajouté aux listes libres. Retourne
`true` si effectivement rendu.
"""
function grad_release!(gp::GradPool, buf::AbstractArray{Float32})
    if haskey(gp.lent, buf)
        delete!(gp.lent, buf)
        key = size(buf)
        haskey(gp.buckets, key) || (gp.buckets[key] = AbstractArray{Float32}[])
        push!(gp.buckets[key], buf)
        return true
    end
    gp.n_foreign += 1
    return false
end

grad_owns(gp::GradPool, buf) = haskey(gp.lent, buf)

"""
    empty_grad_pool!(gp)

Vide les listes libres, libérant la mémoire GPU de façon synchrone
(`Backend.free!`) plutôt que d'attendre le GC. N'affecte pas les buffers
actuellement prêtés (`lent`) -- à n'appeler qu'entre deux passes de
backward, jamais pendant.
"""
function empty_grad_pool!(gp::GradPool)
    for (_, bucket) in gp.buckets
        for buf in bucket
            Backend.free!(gp.device, buf)
        end
        empty!(bucket)
    end
    return gp
end

# WeakKeyDict : un graphe jeté par l'appelant (ex. un notebook qui en
# reconstruit un par cellule) ne garde pas son GradPool -- et donc pas ses
# tampons GPU -- artificiellement en vie pour le reste du processus.
const _GRAD_POOLS = WeakKeyDict{NeuroGraph, GradPool}()

"""
    _grad_pool_for(g) -> GradPool

Pool persistant PAR GRAPHE (pas par passe -- les listes libres survivent
d'un `backward_graph!` à l'autre, c'est ce qui élimine les allocations
répétées ; seuls les emprunts `lent` sont bornés à une seule passe).
"""
function _grad_pool_for(g::NeuroGraph)
    get!(_GRAD_POOLS, g) do
        GradPool(g.device)
    end
end
