import Base: @kwdef

# Macros factices (pour que Julia accepte la syntaxe)
macro node(args...); end
macro rule(args...); end
macro snapshot(args...); end

@kwdef mutable struct GraphBuilder
    graph::NeuroGraph
    namespace::Symbol
    ctx::CtxStore                      = CtxStore()
    log::ExecutionLog                  = ExecutionLog()
    recorder::Union{Nothing,TrainingRecorder} = nothing
    memo::Dict{Any,Symbol}             = Dict{Any,Symbol}()
    rules::Dict{Symbol,Function}       = Dict{Symbol,Function}()
    formula_funcs::Dict{Symbol,Function} = Dict{Symbol,Function}()
    operators::Vector{Symbol}          = Symbol[]
    node_names::Vector{Symbol}         = Symbol[]
    snapshot_epoch::Int                = 0
end

_to_vector(x) = isa(x, Vector) ? x : []

function collect_rule_names(block)
    names = Symbol[]
    for e in _to_vector(block.args)
        if e isa Expr && e.head == :macrocall && e.args[1] == Symbol("@rule")
            def = e.args[3]
            if def.head == :function
                push!(names, def.args[1].args[1])
            elseif def.head == :(=) && def.args[1] isa Expr && def.args[1].head == :call
                push!(names, def.args[1].args[1])
            end
        end
    end
    return names
end

function collect_node_names(block)
    names = Symbol[]
    for e in _to_vector(block.args)
        if e isa Expr && e.head == :macrocall && e.args[1] == Symbol("@node")
            assign = e.args[3]
            if assign.head == :(=)
                push!(names, assign.args[1])
            end
        end
    end
    return names
end

function kw_args_from_expr(call_expr)
    kw = Any[]
    for a in call_expr.args[2:end]
        if a isa Expr && (a.head == :kw || a.head == :(=))
            key = a.args[1]
            val = a.args[2]
            push!(kw, (key, val))
        end
    end
    return kw
end

# ── Générateur d'expressions textuelles (inchangé) ─────────
function ast_to_string_expr(expr, rule_names, param_names=Symbol[])
    if expr isa Expr
        if expr.head == :call
            f = expr.args[1]
            if f isa Symbol
                if f in rule_names
                    args_str = [Expr(:call, :string, ast_to_string_expr(a, rule_names, param_names))
                                for a in expr.args[2:end] if !(a isa Expr && a.head in (:kw, :(=)))]
                    return Expr(:call, :string, QuoteNode(f), "(",
                                Expr(:call, :join, Expr(:vect, args_str...), ", "), ")")
                elseif f in (:wsum, :add)
                    pos_args = [a for a in expr.args[2:end] if !(a isa Expr && a.head in (:kw, :(=)))]
                    args_str = [Expr(:call, :string, ast_to_string_expr(a, rule_names, param_names))
                                for a in pos_args]
                    return Expr(:call, :join, Expr(:vect, args_str...), " + ")
                elseif f == :nsum
                    pos_args = [a for a in expr.args[2:end] if !(a isa Expr && a.head in (:kw, :(=)))]
                    args_str = [Expr(:call, :string, ast_to_string_expr(a, rule_names, param_names))
                                for a in pos_args]
                    return Expr(:call, :*, "Σ(", Expr(:call, :join, Expr(:vect, args_str...), ", "), ")")
                elseif f == :identity
                    pos_args = [a for a in expr.args[2:end] if !(a isa Expr && a.head in (:kw, :(=)))]
                    return ast_to_string_expr(pos_args[1], rule_names, param_names)
                else
                    return expr
                end
            end
        elseif expr.head == :if
            cond = expr.args[1]
            then_expr = ast_to_string_expr(expr.args[2], rule_names, param_names)
            else_expr = length(expr.args) >= 3 ? ast_to_string_expr(expr.args[3], rule_names, param_names) : ""
            return Expr(:if, cond, then_expr, else_expr)
        elseif expr.head == :&&
            return Expr(:&&, ast_to_string_expr(expr.args[1], rule_names, param_names),
                        ast_to_string_expr(expr.args[2], rule_names, param_names))
        elseif expr.head == :||
            return Expr(:||, ast_to_string_expr(expr.args[1], rule_names, param_names),
                        ast_to_string_expr(expr.args[2], rule_names, param_names))
        end
        return expr
    elseif expr isa QuoteNode
        return Expr(:call, :string, QuoteNode(expr.value))
    else
        return Expr(:call, :string, expr)
    end
end

function replace_rule_calls(expr, bsym, rule_names, node_names, param_names=Symbol[])
    if expr isa Expr
        if expr.head == :macrocall
            mac = expr.args[1]
            if mac == Symbol("@node")
                return process_node(expr, bsym, rule_names, node_names, param_names)
            elseif mac == Symbol("@rule")
                return process_rule(expr, bsym, rule_names, node_names)
            elseif mac == Symbol("@snapshot")
                return process_snapshot(expr, bsym)
            else
                return Expr(expr.head, [replace_rule_calls(a, bsym, rule_names, node_names, param_names) for a in expr.args]...)
            end
        elseif expr.head == :call
            f = expr.args[1]
            if f isa Symbol
                # 🔥 Opérateurs reconnus – liste fixe, pas de REGISTERED_OPS
                if f in (:wsum, :nsum, :identity, :add, :+)
                    op = f == :+ ? :add : f
                    pos_exprs = Any[]
                    kw_exprs = Any[]
                    for a in expr.args[2:end]
                        if a isa Expr && (a.head == :kw || a.head == :(=))
                            key = a.args[1]
                            val = a.args[2]
                            push!(kw_exprs, (key, val))
                        else
                            push!(pos_exprs, a)
                        end
                    end
                    new_pos_args = [replace_rule_calls(a, bsym, rule_names, node_names, param_names) for a in pos_exprs]
                    new_kw_args = [(k, replace_rule_calls(v, bsym, rule_names, node_names, param_names)) for (k,v) in kw_exprs]
                    return process_operator_call_full(op, new_pos_args, new_kw_args, bsym)
                elseif f in rule_names
                    new_args = [replace_rule_calls(a, bsym, rule_names, node_names, param_names) for a in expr.args[2:end]]
                    return :(call_rule($(bsym), $(QuoteNode(f)), $(new_args...)))
                else
                    new_args = [replace_rule_calls(a, bsym, rule_names, node_names, param_names) for a in expr.args[2:end]]
                    return Expr(:call, f, new_args...)
                end
            else
                new_args = [replace_rule_calls(a, bsym, rule_names, node_names, param_names) for a in expr.args[2:end]]
                return Expr(:call, f, new_args...)
            end
        else
            return Expr(expr.head, [replace_rule_calls(a, bsym, rule_names, node_names, param_names) for a in expr.args]...)
        end
    elseif expr isa Symbol
        if expr in node_names
            return QuoteNode(expr)
        elseif isempty(param_names)
            return esc(expr)
        else
            return expr
        end
    elseif expr isa QuoteNode || expr isa Number
        return expr
    else
        return expr
    end
end

function process_operator_call_full(op, pos_args, kw_args, bsym)
    attrs_expr = Expr(:call, :Dict, [Expr(:call, :(=>), QuoteNode(k), v) for (k,v) in kw_args]...)
    quote
        local in_syms = Symbol[$(pos_args...)]   # <-- la correction clé
        local key = ($(QuoteNode(op)), in_syms...)
        if haskey($(bsym).memo, key)
            $(bsym).memo[key]
        else
            local out_sym = Symbol(string($(QuoteNode(op)), "_", length($(bsym).memo)))
            addrule!($(bsym).graph, GraphRule(
                out_sym, in_syms, $(QuoteNode(op));
                namespace = $(bsym).namespace,
                attrs = $(esc(attrs_expr))
            ))
            local in_labels = [get($(bsym).graph.nodes[$(bsym).namespace][s].aux_data, :label, string(s)) for s in in_syms]
            local lhs_labels = [occursin(" = ", l) ? split(l, " = "; limit=2)[1] : l for l in in_labels]
            local op_str = $(QuoteNode(op))
            local expr_str = if op_str == :wsum
                string(out_sym, " = ", join(lhs_labels, " + "))
            elseif op_str == :nsum
                string(out_sym, " = nsum(", join(lhs_labels, ", "), ")")
            elseif op_str == :identity
                in_labels[1]
            elseif op_str == :add
                string(out_sym, " = ", join(lhs_labels, " + "))
            else
                string(out_sym, " = ", op_str, "(", join(lhs_labels, ", "), ")")
            end
            $(bsym).graph.nodes[$(bsym).namespace][out_sym].aux_data[:label] = expr_str
            $(bsym).memo[key] = out_sym
            out_sym
        end
    end
end

macro neuro(g_expr, args...)
    ns = :main
    capture = false
    block = nothing
    for a in args
        if a isa Expr && a.head == :(=)
            if a.args[1] == :ns
                ns = a.args[2]
            elseif a.args[1] == :capture
                capture = a.args[2]
            end
        elseif a isa Expr && a.head == :block
            block = a
        end
    end
    block === nothing && error("@neuro : bloc begin...end manquant")

    rule_names = collect_rule_names(block)
    node_names = collect_node_names(block)
    bsym = gensym("builder")

    quote
        local $bsym = GraphBuilder(
            graph = $(esc(g_expr)),
            namespace = $(esc(ns)),
            recorder = $(esc(capture)) ? TrainingRecorder() : nothing,
            operators = [:wsum, :nsum, :identity, :add],
            node_names = $(node_names)
        )
        $(process_block(block, bsym, rule_names, node_names))
        $bsym
    end
end

function process_block(block, bsym, rule_names, node_names)
    exprs = Expr[]
    for e in _to_vector(block.args)
        e isa LineNumberNode && continue
        push!(exprs, replace_rule_calls(e, bsym, rule_names, node_names))
    end
    Expr(:block, exprs...)
end

function process_rule(expr, bsym, rule_names, node_names)
    def = expr.args[3]
    if def.head == :function
        fname = def.args[1].args[1]
        args = _to_vector(def.args[1].args)
        param_names = args[2:end]
        body = def.args[2]
        new_body = transform_rule_body(body, bsym, fname, rule_names, node_names, param_names)
        rule_args = Expr(:tuple, bsym, param_names...)
        rule_func = Expr(:(->), rule_args, new_body)
        sig_str = string(fname, "(", join(param_names, ", "), ")")
        formula_func = :( (args...) -> $sig_str )
        rule_assign = Expr(:(=), Expr(:ref, Expr(:., bsym, QuoteNode(:rules)), QuoteNode(fname)), rule_func)
        formula_assign = Expr(:(=), Expr(:ref, Expr(:., bsym, QuoteNode(:formula_funcs)), QuoteNode(fname)), formula_func)
        return Expr(:block, rule_assign, formula_assign)
    elseif def.head == :(=) && def.args[1] isa Expr && def.args[1].head == :call
        fname = def.args[1].args[1]
        args = _to_vector(def.args[1].args)
        param_names = args[2:end]
        body = def.args[2]
        new_body = transform_rule_body(body, bsym, fname, rule_names, node_names, param_names)
        rule_args = Expr(:tuple, bsym, param_names...)
        rule_func = Expr(:(->), rule_args, new_body)
        sig_str = string(fname, "(", join(param_names, ", "), ")")
        formula_func = :( (args...) -> $sig_str )
        rule_assign = Expr(:(=), Expr(:ref, Expr(:., bsym, QuoteNode(:rules)), QuoteNode(fname)), rule_func)
        formula_assign = Expr(:(=), Expr(:ref, Expr(:., bsym, QuoteNode(:formula_funcs)), QuoteNode(fname)), formula_func)
        return Expr(:block, rule_assign, formula_assign)
    else
        error("@rule : définition de fonction invalide")
    end
end

function transform_rule_body(body, bsym, fname, rule_names, node_names, param_names)
    if body isa Expr && body.head == :block
        real_exprs = filter(e -> !(e isa LineNumberNode), body.args)
        if length(real_exprs) == 1
            return transform_rule_body(real_exprs[1], bsym, fname, rule_names, node_names, param_names)
        elseif isempty(real_exprs)
            error("Règle vide")
        else
            error("Le corps d'une règle doit contenir une seule expression.")
        end
    elseif body isa Expr && body.head == :if
        cond = body.args[1]
        then_branch = transform_rule_branch(body.args[2], bsym, fname, rule_names, node_names, param_names)
        else_branch = transform_rule_branch(body.args[3], bsym, fname, rule_names, node_names, param_names)
        return Expr(:if, cond, then_branch, else_branch)
    else
        return transform_rule_branch(body, bsym, fname, rule_names, node_names, param_names)
    end
end

function transform_rule_branch(branch, bsym, fname, rule_names, node_names, param_names)
    return replace_rule_calls(branch, bsym, rule_names, node_names, param_names)
end

function process_node(expr, bsym, rule_names, node_names, param_names=Symbol[])
    length(expr.args) < 3 && error("@node : syntaxe incorrecte")
    assign = expr.args[3]
    if !isa(assign, Expr) || assign.head != :(=)
        error("@node : affectation attendue")
    end
    target = assign.args[1]
    rhs = assign.args[2]
    if rhs isa Number
        return quote
            set!($(bsym).graph, $(QuoteNode(target)), Float32[$rhs]; namespace=$(bsym).namespace)
            $(bsym).graph.nodes[$(bsym).namespace][$(QuoteNode(target))].aux_data[:label] =
                string($(QuoteNode(target)), " = ", $rhs)
        end
    elseif rhs isa Expr && rhs.head == :vect
        return quote
            set!($(bsym).graph, $(QuoteNode(target)), Float32[$(rhs.args...)]; namespace=$(bsym).namespace)
            $(bsym).graph.nodes[$(bsym).namespace][$(QuoteNode(target))].aux_data[:label] =
                string($(QuoteNode(target)), " = [", join($rhs.args, ", "), "]")
        end
    elseif rhs isa Expr && rhs.head == :call
        op = rhs.args[1]
        if op in (:wsum, :nsum, :identity, :add, :+)
            op = op == :+ ? :add : op
            pos_args = [replace_rule_calls(a, bsym, rule_names, node_names, param_names)
                        for a in rhs.args[2:end] if !(a isa Expr && (a.head == :kw || a.head == :(=)))]
            kw_args = [(k, replace_rule_calls(v, bsym, rule_names, node_names, param_names))
                       for (k,v) in kw_args_from_expr(rhs)]
            transformed = process_operator_call_full(op, pos_args, kw_args, bsym)
            return quote
                local out_sym = $transformed
                if out_sym != $(QuoteNode(target))
                    addrule!($(bsym).graph, GraphRule(
                        $(QuoteNode(target)), Symbol[out_sym], :identity;
                        namespace=$(bsym).namespace))
                end
                local inner_label = get($(bsym).graph.nodes[$(bsym).namespace][out_sym].aux_data, :label, string(out_sym))
                local final_label = occursin(" = ", inner_label) ?
                    string($(QuoteNode(target)), inner_label[last(findfirst(" = ", inner_label)):end]) :
                    string($(QuoteNode(target)), " = ", inner_label)
                $(bsym).graph.nodes[$(bsym).namespace][$(QuoteNode(target))].aux_data[:label] = final_label
                $(QuoteNode(target))
            end
        elseif op in rule_names
            new_args = [replace_rule_calls(a, bsym, rule_names, node_names, param_names)
                        for a in rhs.args[2:end]]
            return quote
                local out = call_rule($(bsym), $(QuoteNode(op)), $(new_args...))
                if out != $(QuoteNode(target))
                    addrule!($(bsym).graph, GraphRule(
                        $(QuoteNode(target)), Symbol[out], :identity;
                        namespace=$(bsym).namespace))
                end
                local inner_label = get($(bsym).graph.nodes[$(bsym).namespace][out].aux_data, :label, string(out))
                local final_label = occursin(" = ", inner_label) ?
                    string($(QuoteNode(target)), inner_label[last(findfirst(" = ", inner_label)):end]) :
                    string($(QuoteNode(target)), " = ", inner_label)
                $(bsym).graph.nodes[$(bsym).namespace][$(QuoteNode(target))].aux_data[:label] = final_label
                $(QuoteNode(target))
            end
        else
            return :( set!($(bsym).graph, $(QuoteNode(target)), $(esc(rhs)); namespace=$(bsym).namespace) )
        end
    else
        error("@node : membre de droite non reconnu")
    end
end

function process_snapshot(expr, bsym)
    kwargs = Dict{Symbol,Any}()
    for a in expr.args[3:end]
        if a isa Expr && a.head == :(=)
            kwargs[a.args[1]] = a.args[2]
        end
    end
    epoch_expr = get(kwargs, :epoch, :($(bsym).snapshot_epoch += 1))
    loss_expr  = get(kwargs, :loss, 0.0)
    params_expr = get(kwargs, :params, :(Dict{Symbol,AbstractArray{Float32}}()))
    return :( record_snapshot!($(bsym), $epoch_expr, Float32($loss_expr), Dict{Symbol,AbstractArray{Float32}}($params_expr)) )
end

function record_snapshot!(builder::GraphBuilder, epoch::Int, loss::Float32, params::Dict)
    if builder.recorder !== nothing
        snap = TrainingSnapshot(epoch, 0, loss, builder.log, params)
        push!(builder.recorder.snapshots, snap)
    end
end

function call_rule(builder::GraphBuilder, fname::Symbol, args...)
    key = (fname, args...)
    if haskey(builder.memo, key)
        return builder.memo[key]
    end
    func = builder.rules[fname]
    result = func(builder, args...)
    # Créer un nœud identity pour représenter l'appel de règle
    call_sym = Symbol(fname, "_", length(builder.memo))
    addrule!(builder.graph, GraphRule(call_sym, Symbol[result], :identity; namespace=builder.namespace))
    label = string(fname, "(", join(args, ", "), ")")
    builder.graph.nodes[builder.namespace][call_sym].aux_data[:label] = label
    builder.graph.nodes[builder.namespace][call_sym].aux_data[:is_rule] = true
    builder.memo[key] = call_sym
    return call_sym
end