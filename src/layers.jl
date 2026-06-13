
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
struct MultiHeadAttention; dim::Int; n_heads::Int; d_head::Int; end
MultiHeadAttention(dim,n_heads) = MultiHeadAttention(dim, n_heads, dim÷n_heads)

function (m::MultiHeadAttention)(g::NeuroGraph, x_sym::Symbol, prefix::Symbol;
                                 namespace=g.active_ns)
    q_full = Linear(m.dim, m.dim, bias=false)(g, x_sym, Symbol(prefix,:_q); namespace=namespace)
    k_full = Linear(m.dim, m.dim, bias=false)(g, x_sym, Symbol(prefix,:_k); namespace=namespace)
    v_full = Linear(m.dim, m.dim, bias=false)(g, x_sym, Symbol(prefix,:_v); namespace=namespace)

    head_outputs = Symbol[]

    for h in 1:m.n_heads
        s = (h-1)*m.d_head + 1
        e =  h   *m.d_head

        qh = Symbol(prefix, :_q_h, h)
        kh = Symbol(prefix, :_k_h, h)
        vh = Symbol(prefix, :_v_h, h)
        addrule!(g, GraphRule(qh, [q_full], :slice_cols;
            attrs=Dict{Symbol,Any}(:start_col=>s,:end_col=>e), namespace=namespace))
        addrule!(g, GraphRule(kh, [k_full], :slice_cols;
            attrs=Dict{Symbol,Any}(:start_col=>s,:end_col=>e), namespace=namespace))
        addrule!(g, GraphRule(vh, [v_full], :slice_cols;
            attrs=Dict{Symbol,Any}(:start_col=>s,:end_col=>e), namespace=namespace))

        sc_h = Symbol(prefix, :_sc_h, h)
        sk_h = Symbol(prefix, :_sk_h, h)
        pr_h = Symbol(prefix, :_pr_h, h)
        ao_h = Symbol(prefix, :_ao_h, h)
        addrule!(g, GraphRule(sc_h, [qh, kh], :matmul;
            attrs=Dict{Symbol,Any}(:trans_b=>true), namespace=namespace))
        addrule!(g, GraphRule(sk_h, [sc_h], :scale_mask;
            attrs=Dict{Symbol,Any}(:d_head=>m.d_head), namespace=namespace))
        addrule!(g, GraphRule(pr_h, [sk_h], :softmax; namespace=namespace))
        addrule!(g, GraphRule(ao_h, [pr_h, vh], :matmul; namespace=namespace))
        push!(head_outputs, ao_h)
    end

    concat_sym = Symbol(prefix, :_concat)
    addrule!(g, GraphRule(concat_sym, head_outputs, :hcat_heads; namespace=namespace))

    return Linear(m.dim, m.dim, bias=false)(g, concat_sym,
                  Symbol(prefix,:_output); namespace=namespace)
end

# ── LlamaBlock ────────────────────────────────────────────────────────────────
struct LlamaBlock; dim::Int; n_heads::Int; hidden_dim::Int; end

function (m::LlamaBlock)(g::NeuroGraph, x_sym::Symbol, prefix::Symbol;
                         namespace=g.active_ns)
    xn1=LayerNorm(m.dim)(g,x_sym,Symbol(prefix,:_norm1);namespace=namespace)
    ao=MultiHeadAttention(m.dim,m.n_heads)(g,xn1,Symbol(prefix,:_mha);namespace=namespace)
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
struct LlamaModel; n_layers::Int; blocks::Vector{LlamaBlock}; dim::Int; end
LlamaModel(n,dim,nh,hd) = LlamaModel(n,[LlamaBlock(dim,nh,hd) for _ in 1:n],dim)

function (m::LlamaModel)(g::NeuroGraph, x_sym::Symbol; namespace=g.active_ns)
    cur=x_sym
    for i in 1:m.n_layers
        cur=m.blocks[i](g,cur,Symbol(:layer_,i);namespace=namespace)
    end
    return cur
end
