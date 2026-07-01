import json
from pathlib import Path

path = Path('notebook/notebook.ipynb')
with path.open('r', encoding='utf-8') as f:
    nb = json.load(f)

new_source = [
    'using BenchmarkTools, NeuroDSL\n',
    'using Printf\n',
    '\n',
    '# Comprehensive NeuroDSL benchmark cell\n',
    '# Warmup, allocation measurement, and stable BenchmarkTools settings\n',
    'g = NeuroGraph(device=Backend.CPUDevice())\n',
    'H = randn(Float32, 10, 5)\n',
    'W = randn(Float32, 8, 5)\n',
    'set!(g, :H, H; is_param=true)\n',
    'set!(g, :W, W; is_param=true)\n',
    'addrule!(g, GraphRule(:Y, [:H, :W], :matmul; attrs=Dict(:trans_b=>true)))\n',
    'addrule!(g, GraphRule(:loss, [:Y], :sum_matrix))\n',
    '\n',
    'function bench_fwd(g, ctx)\n',
    '    invalidate_all!(g)\n',
    '    empty!(ctx)\n',
    '    demand!(g, :loss; ctx_store=ctx)\n',
    'end\n',
    '\n',
    'function bench_bwd(g, ctx)\n',
    '    invalidate_all!(g)\n',
    '    empty!(ctx)\n',
    '    demand!(g, :loss; ctx_store=ctx)\n',
    '    backward_graph!(g, :loss; ctx_store=ctx)\n',
    'end\n',
    '\n',
    'ctx = CtxStore()\n',
    '\n',
    '# Warmup phase\n',
    'for i in 1:20\n',
    '    bench_fwd(g, ctx)\n',
    '    bench_bwd(g, ctx)\n',
    '    GC.gc()\n',
    'end\n',
    '\n',
    'println("Lancement du Benchmark complet...")\n',
    'alloc_fwd = @allocated bench_fwd(g, ctx)\n',
    'alloc_bwd = @allocated bench_bwd(g, ctx)\n',
    '\n',
    'b_fwd = @benchmark bench_fwd($g, $ctx) samples=100 evals=3\n',
    'b_bwd = @benchmark bench_bwd($g, $ctx) samples=100 evals=3\n',
    '\n',
    '@printf("Allocated FWD = %d bytes\\n", alloc_fwd)\n',
    '@printf("Allocated BWD = %d bytes\\n", alloc_bwd)\n',
    '\n',
    'display(b_fwd)\n',
    'display(b_bwd)\n',
]
new_cell = {
    'cell_type': 'code',
    'execution_count': None,
    'metadata': {},
    'outputs': [],
    'source': new_source,
}

insert_index = 34
nb['cells'].insert(insert_index, new_cell)
with path.open('w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print(f'Inserted comprehensive benchmark cell at index {insert_index}')
