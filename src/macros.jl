
macro addrules(graph_expr, args...)
    ns_arg_expr = nothing
    body = nothing

    if length(args) == 2
        ns_arg_expr = args[1]
        body = args[2]
    elseif length(args) == 1
        body = args[1]
    else
        error("@addrules g begin...end  ou  @addrules g :ns begin...end")
    end

    stmts = filter(s -> !(s isa LineNumberNode), body.args)
    rule_calls = map(stmts) do stmt
        (stmt isa Expr && stmt.head == :(=)) || return :()
        lhs = stmt.args[1]
        rhs = stmt.args[2]
        (rhs isa Expr && rhs.head == :call) || return :()
        op_sym = QuoteNode(rhs.args[1])
        raw_args = rhs.args[2:end]
        inputs = [QuoteNode(a) for a in raw_args if a isa Symbol]
        kw_exprs = filter(a -> a isa Expr && (a.head == :kw || a.head == :(=)), raw_args)
        kw_pairs = [(e.args[1], e.args[2]) for e in kw_exprs]

        current_rule_ns_expr = if ns_arg_expr === nothing
                                   :($(esc(graph_expr)).active_ns)
                               else
                                   esc(ns_arg_expr)
                               end

        quote
            let _g = $(esc(graph_expr))
                _ns = $current_rule_ns_expr
                _ensure_namespace!(_g, _ns)
                addrule!(_g, GraphRule($(QuoteNode(lhs)), Symbol[$(inputs...)], $op_sym;
                    attrs=Dict{Symbol,Any}($([:($(QuoteNode(k))=>$(esc(v))) for (k,v) in kw_pairs]...)),
                    namespace=_ns, atom_type=Quantom))
            end
        end
    end

    print_ns_val_expr = if ns_arg_expr === nothing
                            :($(esc(graph_expr)).active_ns)
                        else
                            esc(ns_arg_expr)
                        end

    quote
        $(rule_calls...)
        let _g = $(esc(graph_expr))
            _ns_to_print = $print_ns_val_expr
            println("✅ @addrules [:", _ns_to_print, "] — ", length(_g.rules[_ns_to_print]), " règles")
        end
    end
end
