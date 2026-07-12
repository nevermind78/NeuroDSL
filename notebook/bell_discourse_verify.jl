using NeuroDSL, Printf

@defop scale_add out = attrs[:factor] * inputs[1] + inputs[2]
@defop identity  out = inputs[1]
@defop nsum (dev, out, inputs, attrs, out_sym, nd, ctx) -> begin
    copyto!(out, inputs[1])
    for i in 2:length(inputs)
        out .+= inputs[i]
    end
end

function build_stirling_graph(g, n)
    builder = @neuro g ns=:stirling operators=[:scale_add] begin
        @node one  = 1.0
        @node zero = 0.0

        @rule stirling(n::Int, k::Int) = begin
            if n == 0 && k == 0
                :one
            elseif k == 0 || k > n
                :zero
            else
                scale_add(factor=k, stirling(n-1, k), stirling(n-1, k-1))
            end
        end

        all_terms = [stirling(n, i) for i in 1:n]
        @node bell = nsum(all_terms...)
    end
    return builder
end

n = 6
g = NeuroGraph(namespace=:stirling, device=Backend.CPUDevice())
builder = build_stirling_graph(g, n)

log = ExecutionLog()
ctx = CtxStore()
NeuroDSL.demand!(g, :bell; ctx_store=ctx, namespace=:stirling, log=log)

bell_val = Array(NeuroDSL.node(g, :bell; namespace=:stirling).value)[]
@printf "Bell number B_%d = %d\n" n Int(bell_val)

n_nodes = length(g.nodes[:stirling])
println("Total nodes in graph: ", n_nodes)

save_interactive_graph(g, log, joinpath(@__DIR__, "stirling_bell_6_verify.html"); title = "Stirling & Bell numbers (n=$n)")
println("Saved -> notebook/stirling_bell_6_verify.html")
