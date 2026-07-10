
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
struct MultiHeadAttention; dim::Int; n_heads::Int; d_head::Int; batched::Bool; end
MultiHeadAttention(dim,n_heads; batched::Bool=false) = MultiHeadAttention(dim, n_heads, dim÷n_heads, batched)

function (m::MultiHeadAttention)(g::NeuroGraph, x_sym::Symbol, prefix::Symbol;
                                 namespace=g.active_ns)
    q_full = Linear(m.dim, m.dim, bias=false)(g, x_sym, Symbol(prefix,:_q); namespace=namespace)
    k_full = Linear(m.dim, m.dim, bias=false)(g, x_sym, Symbol(prefix,:_k); namespace=namespace)
    v_full = Linear(m.dim, m.dim, bias=false)(g, x_sym, Symbol(prefix,:_v); namespace=namespace)

    slice_op = m.batched ? :view_cols : :slice_cols
    head_outputs = Symbol[]
    qh_syms = Symbol[]; kh_syms = Symbol[]; vh_syms = Symbol[]; pr_syms = Symbol[]

    for h in 1:m.n_heads
        s = (h-1)*m.d_head + 1
        e =  h   *m.d_head

        qh = Symbol(prefix, :_q_h, h)
        kh = Symbol(prefix, :_k_h, h)
        vh = Symbol(prefix, :_v_h, h)
        addrule!(g, GraphRule(qh, [q_full], slice_op;
            attrs=Dict{Symbol,Any}(:start_col=>s,:end_col=>e), namespace=namespace))
        addrule!(g, GraphRule(kh, [k_full], slice_op;
            attrs=Dict{Symbol,Any}(:start_col=>s,:end_col=>e), namespace=namespace))
        addrule!(g, GraphRule(vh, [v_full], slice_op;
            attrs=Dict{Symbol,Any}(:start_col=>s,:end_col=>e), namespace=namespace))
        push!(qh_syms, qh); push!(kh_syms, kh); push!(vh_syms, vh)
    end

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
struct LlamaBlock; dim::Int; n_heads::Int; hidden_dim::Int; batched_attn::Bool; end
LlamaBlock(dim,n_heads,hidden_dim; batched_attn::Bool=false) = LlamaBlock(dim,n_heads,hidden_dim,batched_attn)

function (m::LlamaBlock)(g::NeuroGraph, x_sym::Symbol, prefix::Symbol;
                         namespace=g.active_ns)
    xn1=LayerNorm(m.dim)(g,x_sym,Symbol(prefix,:_norm1);namespace=namespace)
    ao=MultiHeadAttention(m.dim,m.n_heads; batched=m.batched_attn)(g,xn1,Symbol(prefix,:_mha);namespace=namespace)
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
# explicitement mis à `true`.
struct LlamaModel; n_layers::Int; blocks::Vector{LlamaBlock}; dim::Int; end
LlamaModel(n,dim,nh,hd; batched_attn::Bool=false) =
    LlamaModel(n,[LlamaBlock(dim,nh,hd; batched_attn=batched_attn) for _ in 1:n],dim)

function (m::LlamaModel)(g::NeuroGraph, x_sym::Symbol; namespace=g.active_ns)
    cur=x_sym
    for i in 1:m.n_layers
        cur=m.blocks[i](g,cur,Symbol(:layer_,i);namespace=namespace)
    end
    return cur
end
