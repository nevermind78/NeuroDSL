using Test

# Déterminer le chemin absolu du dossier src et test
src_dir = joinpath(@__DIR__, "..", "src")
test_dir = @__DIR__

using NeuroDSL

@testset "NeuroDSL v4" begin
    include(joinpath(test_dir, "test_graph_api.jl"))
    include(joinpath(test_dir, "test_kernels.jl"))
    include(joinpath(test_dir, "test_backward.jl"))
    include(joinpath(test_dir, "test_compiler.jl"))
    include(joinpath(test_dir, "test_layers.jl"))
    include(joinpath(test_dir, "test_runtime_optimizations.jl"))
    include(joinpath(test_dir, "test_backward_sparse.jl"))
end
