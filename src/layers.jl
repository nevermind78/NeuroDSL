
# ── Embedding ─────────────────────────────────────────────────────────────────
struct Embedding; vocab_size::Int; dim::Int; end

function (m::Embedding)(g::NeuroGraph, idx_sym::Symbol, prefix::Symbol;
                        namespace=g.active_ns)
    en=Symbol(prefix,:_E); on=Symbol(prefix,:_out)
    k=1f0/sqrt(Float32(m.dim))
    E=(Backend.rand32(g.device,m.vocab_size,m.dim) .- 0.5f0) .* (2k)
    set!(g,en,E;is_param=true,namespace=namespace)
    addrule!(g,GraphRule(on,[en,idx_sym],:embedding;namespace=namespace))
    return on
end

# ── LayerNorm (= RMSNorm ici, style Llama) ────────────────────────────────────
struct LayerNorm; dim::Int; eps::Float32; end
LayerNorm(dim::Int; eps=1f-6) = LayerNorm(dim, Float32(eps))

function (m::LayerNorm)(g::NeuroGraph, input_sym::Symbol, prefix::Symbol;
                        namespace=g.active_ns)
    gn=Symbol(prefix,:_gamma); on=Symbol(prefix,:_out)
    set!(g,gn,Backend.ones32(g.device,m.dim);is_param=true,namespace=namespace)
    addrule!(g,GraphRule(on,[input_sym,gn],:rmsnorm;
             attrs=Dict{Symbol,Any}(:eps=>m.eps),namespace=namespace))
    return on
end

# ── Linear ────────────────────────────────────────────────────────────────────
struct Linear; in_features::Int; out_features::Int; has_bias::Bool; end
Linear(i,o;bias=true) = Linear(i,o,bias)

function (m::Linear)(g::NeuroGraph, input_sym::Symbol, prefix::Symbol;
                     namespace=g.active_ns)
    wn=Symbol(prefix,:_W); on=Symbol(prefix,:_out)
    k=1f0/sqrt(Float32(m.in_features))
    W=(Backend.rand32(g.device,m.out_features,m.in_features) .- 0.5f0) .* (2k)
    set!(g,wn,W;is_param=true,namespace=namespace)
    if m.has_bias
        bn=Symbol(prefix,:_b)
        b=(Backend.rand32(g.device,m.out_features) .- 0.5f0) .* (2k)
        set!(g,bn,b;is_param=true,namespace=namespace)
        addrule!(g,GraphRule(on,[input_sym,wn,bn],:linear;namespace=namespace))
    else
        addrule!(g,GraphRule(on,[input_sym,wn],:matmul;
                 attrs=Dict{Symbol,Any}(:trans_b=>true),namespace=namespace))
    end
    return on
end

# ── MultiHeadAttention ────────────────────────────────────────────────────────
# `batched` (défaut false, opt-in -- même discipline que tout le reste de
# cette session) : quand true, les 2 matmuls par tête (Q·Kᵀ et P·V) sont
# calculés en UN seul appel groupé (`:batched_qk`/`:batched_pv`,
# gemm_strided_batched, src/kernels.jl) au lieu de n_heads appels séparés --
# conçu avec Fable le 2026-07-10. `q_h`/`k_h`/`v_h` deviennent des vues
# zero-copy (`:view_cols`) sur `q_full`/`k_full`/`v_full`, et `sc_h`/`ao_h`
# des vues zero-copy (`:head_view`) sur les tenseurs groupés -- CHAQUE nœud
# reste individuellement adressable/patchable exactement comme avant
# (`patch_node!`, `sweep_patch_sites!`, `greedy_patch_search!` fonctionnent
# sans modification). `scale_mask`/`softmax` restent par tête, sur des vues
# (mesuré moins cher que de les batcher aussi, voir la conception Fable).
# `batched=false` émet exactement le graphe historique (:slice_cols partout,
# aucun changement de comportement) -- c'est le repli immédiat en cas de doute.
#
# `n_kv_heads`/`qkv_bias`/`use_rope`/`rope_theta` (2026-07-28, chargement d'un
# LLM réel -- Qwen2.5) : TOUS opt-in, défauts = comportement historique exact
# (`n_kv_heads=n_heads` -> MHA standard, `qkv_bias=false`, `use_rope=false`).
# Aucun call site existant (marker_task, induction, multihop, bidir_recall,
# tests) ne passe ces mots-clés -- donc aucun changement de comportement pour
# eux, vérifié en faisant tourner la suite de tests après ce commit.
#
# GQA (`n_kv_heads < n_heads`) : q_full garde sa largeur pleine
# (n_heads*d_head) ; k_full/v_full sont dimensionnés à n_kv_heads*d_head
# (PAS n_heads*d_head -- correspond à la forme réelle de k_proj/v_proj d'un
# checkpoint GQA, ce n'est pas qu'un choix de tranchage). Chaque tête K/V
# n'est PAS dupliquée en mémoire : le MÊME symbole de nœud est simplement
# référencé group_size=n_heads÷n_kv_heads fois dans les listes d'entrées de
# `:batched_qk`/`:batched_pv` (ou, en mode non batché, indexé group_size fois
# dans la boucle par tête) -- une `GraphRule.inputs` est un `Vector{Symbol}`
# ordinaire, rien n'empêche un même symbole d'y apparaître plusieurs fois, et
# `inputs_vals[i]` lit alors littéralement le MÊME objet Julia à chaque
# position répétée (zéro copie, zéro nouveau nœud). Convention de groupage
# alignée sur `repeat_kv` de HuggingFace : la tête KV d'indice j (1-indexé)
# dessert le bloc CONTIGU de têtes Q [(j-1)*group_size+1 : j*group_size].
# Sur le chemin CUDA batché, `_sibling_view_parent` (src/kernels.jl) ne
# reconnaîtra pas des vues répétées comme des tranches contiguës d'un même
# parent et retombera sur `_gather3` (déjà existant, pas de nouveau code) --
# c'est-à-dire une copie réelle, de taille proportionnelle à n_heads plutôt
# qu'à n_kv_heads, pour les tenseurs K/V *gatherés* uniquement ; correct, pas
# gratuit sur ce chemin précis, documenté ici pour ne pas prétendre le
# contraire.
#
# RoPE (`use_rope=true`) : appliqué à CHAQUE tranche Q et CHAQUE tranche K
# (jamais à V, jamais après le groupage GQA -- une seule fois par tête K
# réelle, pas par tête Q qui la partage) via l'op `:rope` déjà existante
# (`src/dispatch.jl`, convention "rotate-half", vérifiée algébriquement
# identique à celle de Llama/Qwen2 -- PAS la convention "interleaved" que
# certains modèles utilisent). `rope_theta` transite par `attrs[:theta]`
# (défaut 10000f0 dans le dispatcher si absent -- Qwen2.5 utilise 1e6).
struct MultiHeadAttention
    dim::Int; n_heads::Int; d_head::Int; batched::Bool
    n_kv_heads::Int; qkv_bias::Bool; use_rope::Bool; rope_theta::Float32
end
MultiHeadAttention(dim,n_heads; batched::Bool=false, n_kv_heads::Int=n_heads,
                    qkv_bias::Bool=false, use_rope::Bool=false, rope_theta::Real=10000f0) =
    MultiHeadAttention(dim, n_heads, dim÷n_heads, batched, n_kv_heads, qkv_bias, use_rope, Float32(rope_theta))

function (m::MultiHeadAttention)(g::NeuroGraph, x_sym::Symbol, prefix::Symbol;
                                 namespace=g.active_ns)
    m.n_heads % m.n_kv_heads == 0 ||
        error("MultiHeadAttention : n_heads=$(m.n_heads) doit être un multiple de n_kv_heads=$(m.n_kv_heads)")
    group_size = m.n_heads ÷ m.n_kv_heads
    kv_dim = m.n_kv_heads * m.d_head

    q_full = Linear(m.dim, m.dim, bias=m.qkv_bias)(g, x_sym, Symbol(prefix,:_q); namespace=namespace)
    k_full = Linear(m.dim, kv_dim, bias=m.qkv_bias)(g, x_sym, Symbol(prefix,:_k); namespace=namespace)
    v_full = Linear(m.dim, kv_dim, bias=m.qkv_bias)(g, x_sym, Symbol(prefix,:_v); namespace=namespace)

    slice_op = m.batched ? :view_cols : :slice_cols
    head_outputs = Symbol[]
    qh_syms = Symbol[]; pr_syms = Symbol[]
    kh_base_syms = Symbol[]; vh_base_syms = Symbol[]  # longueur n_kv_heads

    for h in 1:m.n_heads
        s = (h-1)*m.d_head + 1
        e =  h   *m.d_head
        qh = Symbol(prefix, :_q_h, h)
        addrule!(g, GraphRule(qh, [q_full], slice_op;
            attrs=Dict{Symbol,Any}(:start_col=>s,:end_col=>e), namespace=namespace))
        if m.use_rope
            qh_r = Symbol(qh, :_rope)
            addrule!(g, GraphRule(qh_r, [qh], :rope;
                attrs=Dict{Symbol,Any}(:theta=>m.rope_theta), namespace=namespace))
            qh = qh_r
        end
        push!(qh_syms, qh)
    end
    for h in 1:m.n_kv_heads
        s = (h-1)*m.d_head + 1
        e =  h   *m.d_head
        kh = Symbol(prefix, :_k_h, h)
        vh = Symbol(prefix, :_v_h, h)
        addrule!(g, GraphRule(kh, [k_full], slice_op;
            attrs=Dict{Symbol,Any}(:start_col=>s,:end_col=>e), namespace=namespace))
        addrule!(g, GraphRule(vh, [v_full], slice_op;
            attrs=Dict{Symbol,Any}(:start_col=>s,:end_col=>e), namespace=namespace))
        if m.use_rope
            kh_r = Symbol(kh, :_rope)
            addrule!(g, GraphRule(kh_r, [kh], :rope;
                attrs=Dict{Symbol,Any}(:theta=>m.rope_theta), namespace=namespace))
            kh = kh_r
        end
        push!(kh_base_syms, kh); push!(vh_base_syms, vh)
    end
    # Répète chaque symbole de tête KV (déjà RoPE'd si demandé) group_size
    # fois -- MHA standard (n_kv_heads==n_heads, group_size==1) redonne
    # exactement kh_base_syms/vh_base_syms tels quels, comportement historique
    # inchangé bit à bit.
    kh_syms = [kh_base_syms[(h-1)÷group_size + 1] for h in 1:m.n_heads]
    vh_syms = [vh_base_syms[(h-1)÷group_size + 1] for h in 1:m.n_heads]

    if m.batched
        sc3 = Symbol(prefix, :_sc3)
        addrule!(g, GraphRule(sc3, vcat(qh_syms, kh_syms), :batched_qk;
            attrs=Dict{Symbol,Any}(:d_head=>m.d_head), namespace=namespace))
    end

    for h in 1:m.n_heads
        sc_h = Symbol(prefix, :_sc_h, h)
        sk_h = Symbol(prefix, :_sk_h, h)
        pr_h = Symbol(prefix, :_pr_h, h)
        if m.batched
            addrule!(g, GraphRule(sc_h, [Symbol(prefix,:_sc3)], :head_view;
                attrs=Dict{Symbol,Any}(:head=>h), namespace=namespace))
        else
            addrule!(g, GraphRule(sc_h, [qh_syms[h], kh_syms[h]], :matmul;
                attrs=Dict{Symbol,Any}(:trans_b=>true), namespace=namespace))
        end
        addrule!(g, GraphRule(sk_h, [sc_h], :scale_mask;
            attrs=Dict{Symbol,Any}(:d_head=>m.d_head), namespace=namespace))
        addrule!(g, GraphRule(pr_h, [sk_h], :softmax; namespace=namespace))
        push!(pr_syms, pr_h)
    end

    if m.batched
        ao3 = Symbol(prefix, :_ao3)
        addrule!(g, GraphRule(ao3, vcat(pr_syms, vh_syms), :batched_pv;
            attrs=Dict{Symbol,Any}(:d_head=>m.d_head), namespace=namespace))
    end

    for h in 1:m.n_heads
        ao_h = Symbol(prefix, :_ao_h, h)
        if m.batched
            addrule!(g, GraphRule(ao_h, [Symbol(prefix,:_ao3)], :head_view;
                attrs=Dict{Symbol,Any}(:head=>h), namespace=namespace))
        else
            addrule!(g, GraphRule(ao_h, [pr_syms[h], vh_syms[h]], :matmul; namespace=namespace))
        end
        push!(head_outputs, ao_h)
    end

    concat_sym = Symbol(prefix, :_concat)
    addrule!(g, GraphRule(concat_sym, head_outputs, :hcat_heads; namespace=namespace))

    return Linear(m.dim, m.dim, bias=false)(g, concat_sym,
                  Symbol(prefix,:_output); namespace=namespace)
end

# ── LlamaBlock ────────────────────────────────────────────────────────────────
# `batched_attn` (défaut false) : transmis tel quel à `MultiHeadAttention` --
# voir sa docstring pour ce que ça change (opt-in, aucun effet par défaut).
# `n_kv_heads`/`qkv_bias`/`use_rope`/`rope_theta` (2026-07-28) : simple
# passe-plat vers `MultiHeadAttention`, mêmes défauts = même comportement
# historique tant qu'ils ne sont pas fixés explicitement.
struct LlamaBlock
    dim::Int; n_heads::Int; hidden_dim::Int; batched_attn::Bool
    n_kv_heads::Int; qkv_bias::Bool; use_rope::Bool; rope_theta::Float32
end
LlamaBlock(dim,n_heads,hidden_dim; batched_attn::Bool=false, n_kv_heads::Int=n_heads,
           qkv_bias::Bool=false, use_rope::Bool=false, rope_theta::Real=10000f0) =
    LlamaBlock(dim,n_heads,hidden_dim,batched_attn,n_kv_heads,qkv_bias,use_rope,Float32(rope_theta))

function (m::LlamaBlock)(g::NeuroGraph, x_sym::Symbol, prefix::Symbol;
                         namespace=g.active_ns)
    xn1=LayerNorm(m.dim)(g,x_sym,Symbol(prefix,:_norm1);namespace=namespace)
    ao=MultiHeadAttention(m.dim,m.n_heads; batched=m.batched_attn, n_kv_heads=m.n_kv_heads,
                           qkv_bias=m.qkv_bias, use_rope=m.use_rope, rope_theta=m.rope_theta)(
                           g,xn1,Symbol(prefix,:_mha);namespace=namespace)
    r1=Symbol(prefix,:_res1)
    addrule!(g,GraphRule(r1,[x_sym,ao],:add;namespace=namespace))

    xn2=LayerNorm(m.dim)(g,r1,Symbol(prefix,:_norm2);namespace=namespace)
    k=1f0/sqrt(Float32(m.dim))
    for (wname,sh) in [(:_mlp_w1,(m.hidden_dim,m.dim)),
                       (:_mlp_w2,(m.dim,m.hidden_dim)),
                       (:_mlp_w3,(m.hidden_dim,m.dim))]
        W=(Backend.rand32(g.device,sh...) .- 0.5f0) .* (2k)
        set!(g,Symbol(prefix,wname),W;is_param=true,namespace=namespace)
    end
    gt=Symbol(prefix,:_gate); up=Symbol(prefix,:_up)
    sg=Symbol(prefix,:_swiglu); mo=Symbol(prefix,:_mlp_out); os=Symbol(prefix,:_out)
    addrule!(g,GraphRule(gt,[xn2,Symbol(prefix,:_mlp_w1)],:matmul;
             attrs=Dict{Symbol,Any}(:trans_b=>true),namespace=namespace))
    addrule!(g,GraphRule(up,[xn2,Symbol(prefix,:_mlp_w3)],:matmul;
             attrs=Dict{Symbol,Any}(:trans_b=>true),namespace=namespace))
    addrule!(g,GraphRule(sg,[gt,up],:swiglu;namespace=namespace))
    addrule!(g,GraphRule(mo,[sg,Symbol(prefix,:_mlp_w2)],:matmul;
             attrs=Dict{Symbol,Any}(:trans_b=>true),namespace=namespace))
    addrule!(g,GraphRule(os,[r1,mo],:add;namespace=namespace))
    return os
end

# ── LlamaModel ────────────────────────────────────────────────────────────────
# `batched_attn` (défaut false) : transmis à chaque `LlamaBlock` -- opt-in,
# aucun effet sur le comportement/les tests existants tant qu'il n'est pas
# explicitement mis à `true`. `n_kv_heads`/`qkv_bias`/`use_rope`/`rope_theta`
# (2026-07-28) : même passe-plat, mêmes défauts = comportement historique.
struct LlamaModel; n_layers::Int; blocks::Vector{LlamaBlock}; dim::Int; end
LlamaModel(n,dim,nh,hd; batched_attn::Bool=false, n_kv_heads::Int=nh,
           qkv_bias::Bool=false, use_rope::Bool=false, rope_theta::Real=10000f0) =
    LlamaModel(n,[LlamaBlock(dim,nh,hd; batched_attn=batched_attn, n_kv_heads=n_kv_heads,
                              qkv_bias=qkv_bias, use_rope=use_rope, rope_theta=rope_theta)
                  for _ in 1:n], dim)

function (m::LlamaModel)(g::NeuroGraph, x_sym::Symbol; namespace=g.active_ns)
    cur=x_sym
    for i in 1:m.n_layers
        cur=m.blocks[i](g,cur,Symbol(:layer_,i);namespace=namespace)
    end
    return cur
end

# ══════════════════════════════════════════════════════════════════════════
# Décodage incrémental avec cache KV (2026-07-28)
# ══════════════════════════════════════════════════════════════════════════
# Motivé par `notebook/qwen2.ipynb` (conversation multi-tours) et une
# proposition de conception externe -- revue et REPRISE avec des
# modifications explicites, pas implémentée telle quelle (voir l'en-tête de
# `src/kv_cache.jl` pour le détail complet : pas de hiérarchie de types
# `Operator` dans ce dépôt -- dispatch par `Symbol`/`register_op!` comme
# tout le reste ; l'invalidation "ciblée" sur `:token_ids` couvre en réalité
# le MÊME ensemble de nœuds qu'`invalidate_all!` pour ce nœud racine, le
# vrai gain vient de la TAILLE des tenseurs, pas du choix de la fonction
# d'invalidation ; buffers alloués frais à chaque pas plutôt que des vues
# zero-copy pré-allouées, pour ne pas toucher `_VIEW_OPS`, un mécanisme
# central du dispatcher, pour un gain secondaire).
#
# CONÇU POUR RÉUTILISER LES POIDS D'UN `LlamaModel` DÉJÀ CONSTRUIT DANS LE
# MÊME NAMESPACE, PAS POUR EN CRÉER DE NOUVEAUX : `CachedMultiHeadAttention`/
# `CachedLlamaBlock`/`CachedLlamaModel` ne font AUCUN `set!(...;
# is_param=true)` -- ils reconstruisent, PAR CONVENTION DE NOMMAGE, les
# symboles de poids exacts que `MultiHeadAttention`/`LlamaBlock`/`Embedding`/
# `LayerNorm`/`Linear` (ci-dessus, et `load_qwen2.jl` pour Qwen) ont déjà
# créés, et leur ajoutent seulement de NOUVEAUX nœuds de calcul (jamais de
# nouveaux nœuds de poids) qui les lisent. Les deux graphes (« cache » et
# « recalcul complet ») partagent alors LITTÉRALEMENT les mêmes tenseurs de
# poids (même objet Julia) -- condition nécessaire pour qu'un test de
# parité entre les deux ait un sens : sinon on testerait deux modèles
# différents portant les mêmes poids PAR HASARD, pas deux chemins
# d'exécution du MÊME modèle.
#
# Convention de nommage reproduite EXACTEMENT depuis `Linear`/`LayerNorm`
# (lignes 29-48, 16-26 ci-dessus) et `MultiHeadAttention`/`LlamaBlock`
# (lignes 100-251 ci-dessus) : pour un bloc construit comme
# `LlamaBlock(...)(g, x, Symbol(:layer_,i))` --
#   poids Q/K/V   : Symbol(prefix,:_mha_q_W) / :_mha_q_b (si qkv_bias), etc.
#   sortie MHA    : Symbol(prefix,:_mha_output_W)  (jamais de biais)
#   normes        : Symbol(prefix,:_norm1_gamma) / :_norm2_gamma
#   MLP           : Symbol(prefix,:_mlp_w1/_w2/_w3)
# Si cette convention change un jour dans `MultiHeadAttention`/`LlamaBlock`,
# ce code cassera BRUYAMMENT (`KeyError` sur `g.nodes[ns][...]`) plutôt que
# de silencieusement lire un mauvais tenseur -- volontairement aucune valeur
# de repli ici.
#
# IMPORTANT -- CONSTRUCTION UNE SEULE FOIS, PAS PAR PAS DE DÉCODAGE : chaque
# structure ci-dessous s'appelle UNE fois pour construire tout le graphe de
# décodage (mêmes symboles de sortie à chaque fois qu'on les lirait -- ce
# N'EST PAS une boucle qui rappelle le functor à chaque token). Le décodage
# lui-même se fait ensuite par `set!`+`invalidate_all!`+`demand!` répétés sur
# CE graphe déjà construit, exactement comme la génération sans cache
# existante (`notebook/real_llm.ipynb`) -- seule différence : `x_sym` porte
# 1 ligne (le nouveau token) au lieu de toute la séquence, et les nœuds
# `:kv_cache_append` accumulent l'historique dans leur `aux_data` d'un
# `demand!` à l'autre (même `GraphNode`, jamais recréé).
struct CachedMultiHeadAttention
    dim::Int; n_heads::Int; d_head::Int
    n_kv_heads::Int; qkv_bias::Bool; use_rope::Bool; rope_theta::Float32
end
CachedMultiHeadAttention(dim,n_heads; n_kv_heads::Int=n_heads, qkv_bias::Bool=false,
                          use_rope::Bool=false, rope_theta::Real=10000f0) =
    CachedMultiHeadAttention(dim, n_heads, dim÷n_heads, n_kv_heads, qkv_bias, use_rope, Float32(rope_theta))

"""
    (m::CachedMultiHeadAttention)(g, x_sym, mha_prefix, cur_step_sym, pos_sym; namespace)

`x_sym` porte exactement 1 ligne (le nouveau token, déjà normalisé).
`mha_prefix` doit être le symbole `prefix` du `MultiHeadAttention` DÉJÀ
construit dont on réutilise les poids (ex. `:layer_1_mha`, produit par
`LlamaBlock` via `Symbol(block_prefix,:_mha)`). `cur_step_sym`/`pos_sym`
sont des nœuds `(1,)` réglés par l'appelant avant chaque `demand!`
(`cur_step`=nombre de tokens après ce pas, 1-indexé ; `pos`=position
0-indexée du nouveau token=`cur_step-1`) -- PARTAGÉS entre toutes les
couches/têtes d'un même pas (un seul `set!` par pas, pas par tête/couche).
"""
function (m::CachedMultiHeadAttention)(g::NeuroGraph, x_sym::Symbol, mha_prefix::Symbol,
                                        cur_step_sym::Symbol, pos_sym::Symbol; namespace=g.active_ns)
    m.n_heads % m.n_kv_heads == 0 ||
        error("CachedMultiHeadAttention : n_heads=$(m.n_heads) doit être un multiple de n_kv_heads=$(m.n_kv_heads)")
    group_size = m.n_heads ÷ m.n_kv_heads

    qW = Symbol(mha_prefix,:_q_W); kW = Symbol(mha_prefix,:_k_W); vW = Symbol(mha_prefix,:_v_W)
    oW = Symbol(mha_prefix,:_output_W)

    q_full = Symbol(mha_prefix,:_dec_q); k_full = Symbol(mha_prefix,:_dec_k); v_full = Symbol(mha_prefix,:_dec_v)
    if m.qkv_bias
        qb = Symbol(mha_prefix,:_q_b); kb = Symbol(mha_prefix,:_k_b); vb = Symbol(mha_prefix,:_v_b)
        addrule!(g, GraphRule(q_full, [x_sym,qW,qb], :linear; namespace=namespace))
        addrule!(g, GraphRule(k_full, [x_sym,kW,kb], :linear; namespace=namespace))
        addrule!(g, GraphRule(v_full, [x_sym,vW,vb], :linear; namespace=namespace))
    else
        addrule!(g, GraphRule(q_full, [x_sym,qW], :matmul; attrs=Dict{Symbol,Any}(:trans_b=>true), namespace=namespace))
        addrule!(g, GraphRule(k_full, [x_sym,kW], :matmul; attrs=Dict{Symbol,Any}(:trans_b=>true), namespace=namespace))
        addrule!(g, GraphRule(v_full, [x_sym,vW], :matmul; attrs=Dict{Symbol,Any}(:trans_b=>true), namespace=namespace))
    end

    # Têtes Q : tranche + RoPE à la position ABSOLUE du nouveau token (voir
    # `:rope_at_pos`, src/kv_cache.jl -- l'op `:rope` existant supposerait
    # à tort pos=0 puisque x_sym a toujours 1 seule ligne ici).
    qh_syms = Symbol[]
    for h in 1:m.n_heads
        s=(h-1)*m.d_head+1; e=h*m.d_head
        qh = Symbol(mha_prefix,:_dec_q_h,h)
        addrule!(g, GraphRule(qh, [q_full], :slice_cols;
            attrs=Dict{Symbol,Any}(:start_col=>s,:end_col=>e), namespace=namespace))
        if m.use_rope
            qh_r = Symbol(qh,:_rope)
            addrule!(g, GraphRule(qh_r, [qh,pos_sym], :rope_at_pos;
                attrs=Dict{Symbol,Any}(:theta=>m.rope_theta), namespace=namespace))
            qh = qh_r
        end
        push!(qh_syms, qh)
    end

    # Têtes K/V RÉELLES (n_kv_heads, pas n_heads) : tranche + RoPE (K
    # seulement, jamais V) + append dans le cache. Le nœud `:kv_cache_append`
    # (`kh_cache`/`vh_cache`) porte le MÊME symbole à CHAQUE pas de décodage
    # -- c'est ce qui fait persister `aux_data[:history]` d'un `demand!` au
    # suivant (voir `src/kv_cache.jl`).
    kh_hist_syms = Symbol[]; vh_hist_syms = Symbol[]  # longueur n_kv_heads
    for h in 1:m.n_kv_heads
        s=(h-1)*m.d_head+1; e=h*m.d_head
        kh = Symbol(mha_prefix,:_dec_k_h,h)
        vh = Symbol(mha_prefix,:_dec_v_h,h)
        addrule!(g, GraphRule(kh, [k_full], :slice_cols;
            attrs=Dict{Symbol,Any}(:start_col=>s,:end_col=>e), namespace=namespace))
        addrule!(g, GraphRule(vh, [v_full], :slice_cols;
            attrs=Dict{Symbol,Any}(:start_col=>s,:end_col=>e), namespace=namespace))
        if m.use_rope
            kh_r = Symbol(kh,:_rope)
            addrule!(g, GraphRule(kh_r, [kh,pos_sym], :rope_at_pos;
                attrs=Dict{Symbol,Any}(:theta=>m.rope_theta), namespace=namespace))
            kh = kh_r
        end
        kh_cache = Symbol(mha_prefix,:_kcache_h,h)
        vh_cache = Symbol(mha_prefix,:_vcache_h,h)
        addrule!(g, GraphRule(kh_cache, [kh, cur_step_sym], :kv_cache_append; namespace=namespace))
        addrule!(g, GraphRule(vh_cache, [vh, cur_step_sym], :kv_cache_append; namespace=namespace))
        push!(kh_hist_syms, kh_cache); push!(vh_hist_syms, vh_cache)
    end
    # GQA : chaque historique K/V RÉEL est référencé group_size fois -- même
    # convention (répétition de SYMBOLE, pas de copie) que
    # `MultiHeadAttention` non-caché ci-dessus (lignes 155-160).
    kh_syms = [kh_hist_syms[(h-1)÷group_size + 1] for h in 1:m.n_heads]
    vh_syms = [vh_hist_syms[(h-1)÷group_size + 1] for h in 1:m.n_heads]

    # Attention par tête : Q (1,d_head) contre l'historique K (cur_step,
    # d_head) -> scores (1,cur_step) ; PAS de masque causal (`:scale_no_mask`,
    # pas `:scale_mask` -- voir sa docstring dans src/kv_cache.jl : l'
    # historique ne contient QUE des positions passées, rien à masquer).
    head_outputs = Symbol[]
    for h in 1:m.n_heads
        sc_h = Symbol(mha_prefix,:_dec_sc_h,h); sk_h = Symbol(mha_prefix,:_dec_sk_h,h)
        pr_h = Symbol(mha_prefix,:_dec_pr_h,h); ao_h = Symbol(mha_prefix,:_dec_ao_h,h)
        addrule!(g, GraphRule(sc_h, [qh_syms[h], kh_syms[h]], :matmul;
            attrs=Dict{Symbol,Any}(:trans_b=>true), namespace=namespace))
        addrule!(g, GraphRule(sk_h, [sc_h], :scale_no_mask;
            attrs=Dict{Symbol,Any}(:d_head=>m.d_head), namespace=namespace))
        addrule!(g, GraphRule(pr_h, [sk_h], :softmax; namespace=namespace))
        addrule!(g, GraphRule(ao_h, [pr_h, vh_syms[h]], :matmul; namespace=namespace))
        push!(head_outputs, ao_h)
    end

    concat_sym = Symbol(mha_prefix,:_dec_concat)
    addrule!(g, GraphRule(concat_sym, head_outputs, :hcat_heads; namespace=namespace))
    out_sym = Symbol(mha_prefix,:_dec_output)
    addrule!(g, GraphRule(out_sym, [concat_sym, oW], :matmul;
        attrs=Dict{Symbol,Any}(:trans_b=>true), namespace=namespace))
    return out_sym
end

# `CachedLlamaBlock`/`CachedLlamaModel` : même relation à `LlamaBlock`/
# `LlamaModel` que `CachedMultiHeadAttention` à `MultiHeadAttention` --
# réutilisent les poids (normes + MLP) par convention de nommage, ajoutent
# seulement les nouveaux nœuds de calcul du pas de décodage courant.
struct CachedLlamaBlock
    dim::Int; n_heads::Int; hidden_dim::Int
    n_kv_heads::Int; qkv_bias::Bool; use_rope::Bool; rope_theta::Float32
end
CachedLlamaBlock(dim,n_heads,hidden_dim; n_kv_heads::Int=n_heads, qkv_bias::Bool=false,
                  use_rope::Bool=false, rope_theta::Real=10000f0) =
    CachedLlamaBlock(dim,n_heads,hidden_dim,n_kv_heads,qkv_bias,use_rope,Float32(rope_theta))

function (m::CachedLlamaBlock)(g::NeuroGraph, x_sym::Symbol, block_prefix::Symbol,
                                cur_step_sym::Symbol, pos_sym::Symbol; namespace=g.active_ns)
    n1_gamma = Symbol(block_prefix,:_norm1_gamma)
    xn1 = Symbol(block_prefix,:_dec_norm1)
    addrule!(g, GraphRule(xn1, [x_sym, n1_gamma], :rmsnorm; attrs=Dict{Symbol,Any}(:eps=>1f-6), namespace=namespace))

    mha = CachedMultiHeadAttention(m.dim, m.n_heads; n_kv_heads=m.n_kv_heads, qkv_bias=m.qkv_bias,
                                    use_rope=m.use_rope, rope_theta=m.rope_theta)
    ao = mha(g, xn1, Symbol(block_prefix,:_mha), cur_step_sym, pos_sym; namespace=namespace)

    r1 = Symbol(block_prefix,:_dec_res1)
    addrule!(g, GraphRule(r1, [x_sym, ao], :add; namespace=namespace))

    n2_gamma = Symbol(block_prefix,:_norm2_gamma)
    xn2 = Symbol(block_prefix,:_dec_norm2)
    addrule!(g, GraphRule(xn2, [r1, n2_gamma], :rmsnorm; attrs=Dict{Symbol,Any}(:eps=>1f-6), namespace=namespace))

    w1=Symbol(block_prefix,:_mlp_w1); w2=Symbol(block_prefix,:_mlp_w2); w3=Symbol(block_prefix,:_mlp_w3)
    gt=Symbol(block_prefix,:_dec_gate); up=Symbol(block_prefix,:_dec_up)
    sg=Symbol(block_prefix,:_dec_swiglu); mo=Symbol(block_prefix,:_dec_mlp_out); os=Symbol(block_prefix,:_dec_out)
    addrule!(g, GraphRule(gt,[xn2,w1],:matmul; attrs=Dict{Symbol,Any}(:trans_b=>true), namespace=namespace))
    addrule!(g, GraphRule(up,[xn2,w3],:matmul; attrs=Dict{Symbol,Any}(:trans_b=>true), namespace=namespace))
    addrule!(g, GraphRule(sg,[gt,up],:swiglu; namespace=namespace))
    addrule!(g, GraphRule(mo,[sg,w2],:matmul; attrs=Dict{Symbol,Any}(:trans_b=>true), namespace=namespace))
    addrule!(g, GraphRule(os,[r1,mo],:add; namespace=namespace))
    return os
end

struct CachedLlamaModel; n_layers::Int; blocks::Vector{CachedLlamaBlock}; dim::Int; end
CachedLlamaModel(n,dim,nh,hd; n_kv_heads::Int=nh, qkv_bias::Bool=false, use_rope::Bool=false, rope_theta::Real=10000f0) =
    CachedLlamaModel(n, [CachedLlamaBlock(dim,nh,hd; n_kv_heads=n_kv_heads, qkv_bias=qkv_bias,
                                           use_rope=use_rope, rope_theta=rope_theta) for _ in 1:n], dim)

function (m::CachedLlamaModel)(g::NeuroGraph, x_sym::Symbol, cur_step_sym::Symbol, pos_sym::Symbol; namespace=g.active_ns)
    cur = x_sym
    for i in 1:m.n_layers
        cur = m.blocks[i](g, cur, Symbol(:layer_,i), cur_step_sym, pos_sym; namespace=namespace)
    end
    return cur
end

"""
    build_cached_decode_graph!(g; n_layers, dim, n_heads, hidden_dim, vocab_size,
        n_kv_heads=n_heads, qkv_bias=false, use_rope=false, rope_theta=10000f0,
        namespace=g.active_ns,
        embed_E_sym=:tok_E, final_norm_gamma_sym=:final_norm_gamma, lm_head_W_sym=:lm_head_W,
        token_id_sym=:dec_token_id, cur_step_sym=:dec_cur_step, pos_sym=:dec_pos) -> logits_sym

Construit, UNE SEULE FOIS, le graphe de décodage incrémental complet
(embedding -> `CachedLlamaModel` -> norme finale -> lm_head), en réutilisant
les poids déjà chargés sous `embed_E_sym`/`final_norm_gamma_sym`/
`lm_head_W_sym` et layer_i_* (créés par un `LlamaModel`/`Embedding`/
`LayerNorm`/`Linear` construits AVANT cet appel, dans le MÊME `namespace`
avec le MÊME `n_layers`). Appelle `NeuroDSL._register_kv_cache_ops!()`
en interne (idempotent) -- pas besoin de l'appeler séparément.

`_register_kv_cache_ops!` doit avoir accès à `CUSTOM_OPS`/`CUSTOM_SHAPE_RULES`
définis dans `dispatch.jl`, ce qui est garanti par l'ordre d'`include` dans
`NeuroDSL.jl` (`kv_cache.jl` est inclus juste après `dispatch.jl`).

Retourne le symbole des logits `(1, vocab_size)` du nouveau token. L'appelant
règle `token_id_sym`/`cur_step_sym`/`pos_sym` via `set!` avant chaque
`demand!` (voir la docstring de `CachedMultiHeadAttention` pour la
convention exacte de `cur_step`/`pos`).
"""
function build_cached_decode_graph!(g::NeuroGraph; n_layers::Int, dim::Int, n_heads::Int, hidden_dim::Int, vocab_size::Int,
        n_kv_heads::Int=n_heads, qkv_bias::Bool=false, use_rope::Bool=false, rope_theta::Real=10000f0,
        namespace=g.active_ns,
        embed_E_sym::Symbol=:tok_E, final_norm_gamma_sym::Symbol=:final_norm_gamma, lm_head_W_sym::Symbol=:lm_head_W,
        token_id_sym::Symbol=:dec_token_id, cur_step_sym::Symbol=:dec_cur_step, pos_sym::Symbol=:dec_pos)
    _register_kv_cache_ops!()

    haskey(g.nodes[namespace], token_id_sym) ||
        set!(g, token_id_sym, [1]; atom_type=Datom, namespace=namespace)
    haskey(g.nodes[namespace], cur_step_sym) ||
        set!(g, cur_step_sym, Float32[1f0]; namespace=namespace)
    haskey(g.nodes[namespace], pos_sym) ||
        set!(g, pos_sym, Float32[0f0]; namespace=namespace)

    tok_emb = Symbol(:dec_tok_emb)
    addrule!(g, GraphRule(tok_emb, [embed_E_sym, token_id_sym], :embedding; namespace=namespace))

    model = CachedLlamaModel(n_layers, dim, n_heads, hidden_dim; n_kv_heads=n_kv_heads,
                              qkv_bias=qkv_bias, use_rope=use_rope, rope_theta=rope_theta)
    out = model(g, tok_emb, cur_step_sym, pos_sym; namespace=namespace)

    fn = Symbol(:dec_final_norm)
    addrule!(g, GraphRule(fn, [out, final_norm_gamma_sym], :rmsnorm; attrs=Dict{Symbol,Any}(:eps=>1f-6), namespace=namespace))
    logits = Symbol(:dec_logits)
    addrule!(g, GraphRule(logits, [fn, lm_head_W_sym], :matmul; attrs=Dict{Symbol,Any}(:trans_b=>true), namespace=namespace))
    return logits
end

# ══════════════════════════════════════════════════════════════════════════
# `prime_kv_cache_from_prefix!` (2026-07-29) -- corrige un vrai bug de
# performance trouvé en instrumentant `chat()` (notebook/qwen2.ipynb) :
# remplir le cache avec le préfixe (prompt ChatML, plusieurs dizaines de
# tokens) un token à la fois via `:kv_cache_append` payait ~0.5s PAR TOKEN
# DU PRÉFIXE (l'overhead fixe d'un `demand!` sur ~400 nœuds, répété une fois
# par token) -- pour un préfixe de 36 tokens, ~19-28s, alors qu'un seul
# passage avant batché sur les 36 tokens (le chemin `LlamaModel` existant,
# EXACTEMENT ce que le recalcul complet faisait déjà) coûte ~0.3-0.5s. Le
# cache KV incrémental n'a de sens QUE pour les tokens GÉNÉRÉS un par un
# (chacun dépend du précédent, pas de choix) -- le préfixe, lui, est connu
# en entier d'avance et doit être traité en un seul passage batché, comme
# n'importe quel forward pass normal.
#
# Ce que fait cette fonction : après un passage avant COMPLET (batché,
# causal) sur le préfixe dans `src_ns` (un `LlamaModel` standard -- PAS le
# graphe caché), copie directement le K (post-RoPE si `use_rope`) et le V
# (jamais RoPE'd) de CHAQUE tête KV RÉELLE de CHAQUE couche dans
# `aux_data[:history]` du nœud de cache correspondant de `dst_ns` (un
# `CachedLlamaModel` construit avec les MÊMES `n_layers`/`n_kv_heads`).
# BYPASSE `:kv_cache_append` pour tout le préfixe -- les tokens GÉNÉRÉS
# ensuite continuent d'utiliser le pas incrémental normal (`cur_step` reprend
# simplement à `length(prefix)+1`).
#
# Pourquoi c'est CORRECT, pas juste rapide (voir aussi
# `notebook/kv_cache_prefix_prime_gate.jl`, qui vérifie ceci numériquement,
# pas seulement par argument) : le tranchage K/V par tête et le RoPE sont
# des opérations PAR LIGNE/PAR POSITION, appliquées AVANT le matmul
# d'attention -- aucun mélange entre positions à ce stade. La ligne `t` du
# K/V calculé par le passage avant batché sur `[tok_1,...,tok_n]` est donc
# IDENTIQUE (même formule, même base RoPE) à ce qu'aurait produit un appel
# `:rope_at_pos` sur `tok_t` seul à la position `t-1` -- exactement ce que
# `:kv_cache_append` aurait accumulé ligne par ligne. Le masquage causal du
# passage batché garantit par ailleurs que la position `t` n'a vu que les
# positions `≤t`, comme le ferait un décodage vraiment incrémental.
#
# PRÉCONDITION (non vérifiée ici, à la charge de l'appelant) : `demand!` a
# déjà été appelé sur un nœud en aval du `LlamaModel` de `src_ns`, avec
# `:token_ids` réglé au préfixe EXACT dont on veut peupler le cache --
# sinon les nœuds K/V/RoPE de `src_ns` ne contiennent pas ce qu'on croit y
# copier (cette fonction ne déclenche AUCUN calcul elle-même, seulement des
# copies de valeurs déjà présentes).
#
# Retourne le nombre de tableaux K/V copiés (`2 * n_layers * n_kv_heads` si
# tout va bien) -- sert de vérification de complétude à l'appelant.
function prime_kv_cache_from_prefix!(g::NeuroGraph; src_ns::Symbol, dst_ns::Symbol,
                                      n_layers::Int, n_kv_heads::Int, use_rope::Bool=false)
    n = 0
    for i in 1:n_layers
        mha_prefix = Symbol(:layer_, i, :_mha)
        for h in 1:n_kv_heads
            k_src = use_rope ? Symbol(mha_prefix, :_k_h, h, :_rope) : Symbol(mha_prefix, :_k_h, h)
            v_src = Symbol(mha_prefix, :_v_h, h)
            k_dst = Symbol(mha_prefix, :_kcache_h, h)
            v_dst = Symbol(mha_prefix, :_vcache_h, h)

            haskey(g.nodes[src_ns], k_src) ||
                error("prime_kv_cache_from_prefix! : nœud :$k_src introuvable dans :$src_ns -- " *
                      "la couche $i a-t-elle bien été construite avec use_rope=$use_rope ?")
            haskey(g.nodes[dst_ns], k_dst) ||
                error("prime_kv_cache_from_prefix! : nœud de cache :$k_dst introuvable dans :$dst_ns -- " *
                      "build_cached_decode_graph! a-t-il été appelé avec le même n_layers/n_kv_heads ?")

            k_val = g.nodes[src_ns][k_src].value
            v_val = g.nodes[src_ns][v_src].value
            (k_val === nothing || v_val === nothing) &&
                error("prime_kv_cache_from_prefix! : :$k_src/:$v_src n'ont pas de valeur -- " *
                      "demand! a-t-il été appelé sur le graphe complet (namespace :$src_ns) AVANT ce priming ?")

            g.nodes[dst_ns][k_dst].aux_data[:history] = copy(k_val)
            g.nodes[dst_ns][v_dst].aux_data[:history] = copy(v_val)
            g.nodes[dst_ns][k_dst].aux_data[:kv_cache_active] = true
            g.nodes[dst_ns][v_dst].aux_data[:kv_cache_active] = true
            n += 2
        end
    end
    return n
end
