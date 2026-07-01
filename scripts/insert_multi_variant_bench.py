import json
from pathlib import Path

path = Path('notebook/notebook.ipynb')
with path.open('r', encoding='utf-8') as f:
    nb = json.load(f)

# New multi-variant benchmark cell content (Julia)
new_source = [
    'using BenchmarkTools, Printf, Statistics, Random, NeuroDSL\n',
    '\n',
    '# Multi-Variant Professional Benchmarks for NeuroDSL\n',
    '# 1) Larger workload (scaling), 2) Plain-Julia baseline (forward),\n',
    '# 3) Allocation checks, 4) Stable BenchmarkTools settings + results summary\n',
    '\n',
    'function mk_graph(D, depth, batch=1)\n',
    '    g = NeuroGraph(device=Backend.CPUDevice())\n',
    '    NeuroDSL.set!(g, :x, randn(Float32, batch, D))\n',
    '    NeuroDSL.set!(g, :y, zeros(Float32, batch, D); atom_type=NeuroDSL.Datom)\n',
    '    prev = :x\n',
    '    for i in 1:depth\n',
    '        w = Symbol(:W, i)\n',
    '        NeuroDSL.set!(g, w, randn(Float32, D, D) .* 0.01f0; is_param=true)\n',
    '        out = Symbol(:h, i)\n',
    '        NeuroDSL.addrule!(g, GraphRule(out, [prev, w], :matmul; attrs=Dict(:trans_b=>true)))\n',
    '        prev = out\n',
    '    end\n',
    '    NeuroDSL.addrule!(g, :loss, [prev, :y], :mse_loss)\n',
    '    return g\n',
    'end\n',
    '\n',
    "# Plain-Julia baseline forward (one layer equivalent)\n",
    'function baseline_forward(H, W, y)\n',
    '    Y = H * W\n',
    '    return sum((Y .- y).^2)\n',
    'end\n',
    '\n',
    'function run_variants()\n',
    '    rng = MersenneTwister(1234)\n',
    '    configs = [ (64, 8, 4), (128, 16, 8), (256, 24, 16) ] # (D, depth, batch)\n',
    '    for (D, depth, batch) in configs\n',
    '        println("--- Config D=$D depth=$depth batch=$batch ---")\n',
    '        g = mk_graph(D, depth, batch)\n',
    '        ctx = CtxStore()\n',
    '        # Warmup\n',
    '        for i in 1:10\n',
    '            NeuroDSL.demand!(g, :loss; ctx_store=ctx)\n',
    '            NeuroDSL.backward_graph!(g, :loss; ctx_store=ctx)\n',
    '        end\n',
    '        GC.gc()\n',
    '        # Measure NeuroDSL forward+backward allocation and time\n',
    '        alloc_nb = @allocated begin NeuroDSL.demand!(g, :loss; ctx_store=ctx); NeuroDSL.backward_graph!(g, :loss; ctx_store=ctx) end\n',
    '        b_nb = @benchmark begin NeuroDSL.demand!($g, :loss; ctx_store=$ctx); NeuroDSL.backward_graph!($g, :loss; ctx_store=$ctx) end samples=50 evals=3\n',
    '        println("NeuroDSL allocated: ", alloc_nb, " bytes")\n',
    '        println(b_nb)\n',
    '        # Baseline forward (single-layer mini-problem)\n',
    '        H = randn(Float32, batch, D)\n',
    '        W = randn(Float32, D, D)\n',
    '        y = zeros(Float32, batch, D)\n',
    '        alloc_base = @allocated baseline_forward(H, W, y)\n',
    '        b_base = @benchmark baseline_forward($H, $W, $y) samples=50 evals=3\n',
    '        println("Baseline forward allocated:", alloc_base, " bytes")\n',
    '        println(b_base)\n',
    '        println()\n',
    '    end\n',
    'end\n',
    '\n',
    'run_variants()\n',
]

new_cell = {
    'cell_type': 'code',
    'execution_count': None,
    'metadata': {},
    'outputs': [],
    'source': new_source,
}

# Insert after index 31 (so it is visible near existing benchmarks)
insert_at = 32
nb['cells'].insert(insert_at, new_cell)
with path.open('w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f'Inserted multi-variant benchmark cell at index {insert_at}')
