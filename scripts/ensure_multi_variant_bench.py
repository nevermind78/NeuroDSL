import json
from pathlib import Path
p = Path('notebook/notebook.ipynb')
nb = json.loads(p.read_text(encoding='utf-8'))
search_terms = ['function mk_graph(', 'run_variants()', 'Multi-Variant Professional Benchmarks for NeuroDSL']
found = False
for cell in nb['cells']:
    if cell.get('cell_type')=='code':
        src=''.join(cell.get('source', []))
        if any(term in src for term in search_terms):
            found = True
            break
if found:
    print('already present')
else:
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
        '        alloc_nb = @allocated begin NeuroDSL.demand!(g, :loss; ctx_store=ctx); NeuroDSL.backward_graph!(g, :loss; ctx_store=ctx) end\n',
        '        b_nb = @benchmark begin NeuroDSL.demand!($g, :loss; ctx_store=$ctx); NeuroDSL.backward_graph!($g, :loss; ctx_store=$ctx) end samples=50 evals=3\n',
        '        println("NeuroDSL allocated: ", alloc_nb, " bytes")\n',
        '        println(b_nb)\n',
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
    new_cell = {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':new_source}
    # Insert after the top marker if present, else at 0
    insert_at = 1
    nb['cells'].insert(insert_at, new_cell)
    p.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
    print('inserted at', insert_at)
