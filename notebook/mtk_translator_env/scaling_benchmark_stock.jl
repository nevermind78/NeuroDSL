t0 = time()
function logmsg(s)
    println(s, "  [+", round(time() - t0, digits=1), "s]")
    flush(stdout)
end
import Pkg
Pkg.activate(@__DIR__; io=devnull)
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using LinearAlgebra
logmsg("using done (STOCK Symbolics: eager scalarized dot derivative)")

ns = [3, 6, 10, 15, 20, 30]
results = Tuple{Int, Float64, Bool}[]

for n in ns
    @parameters L = 1.0
    @variables p(t)[1:n] = ones(n) ./ sqrt(n)
    @variables v(t)[1:n] = zeros(n)
    @variables lambda(t) = 1.0
    pv = collect(p); vv = collect(v)
    F = zeros(n)
    eqs = [D.(pv) .~ vv; D.(vv) .~ -lambda .* pv .+ F; dot(p, p) ~ L^2]
    @named sys = ODESystem(eqs, t)
    tstart = time()
    ok = true
    try
        simplified = ModelingToolkit.mtkcompile(sys)
        @assert length(equations(simplified)) > 0
    catch e
        ok = false
        logmsg("n=$n THREW: $(sprint(showerror, e))")
    end
    dt = time() - tstart
    push!(results, (n, dt, ok))
    logmsg("n=$n : mtkcompile took $(round(dt, digits=3))s, succeeded=$ok")
    dt > 90 && (logmsg("aborting sweep: n=$n already took >90s"); break)
end

open(joinpath(@__DIR__, "scaling_stock_results.tsv"), "w") do io
    println(io, "n\tseconds\tsucceeded")
    for (n, dt, ok) in results
        println(io, "$n\t$dt\t$ok")
    end
end
logmsg("done")
