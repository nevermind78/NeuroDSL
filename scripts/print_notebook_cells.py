import json
from pathlib import Path
p = Path('notebook/notebook.ipynb')
nb = json.loads(p.read_text(encoding='utf-8'))
for i, cell in enumerate(nb['cells']):
    typ = cell.get('cell_type')
    first = ''
    if typ == 'code':
        src = ''.join(cell.get('source', []))
        first = src.splitlines()[0] if src.splitlines() else ''
    elif typ == 'markdown':
        src = ''.join(cell.get('source', []))
        first = (src.splitlines()[0] if src.splitlines() else '')
    marker = ''
    for m in ['NEURODSL BENCHMARKS (Inserted for visibility)', 'Comprehensive NeuroDSL benchmark cell', 'Multi-Variant Professional Benchmarks for NeuroDSL', 'Lancement du Benchmark complet...']:
        if m in src:
            marker = m
            break
    print(i, typ, first[:120].replace('\n','\\n'), '>>', marker)
