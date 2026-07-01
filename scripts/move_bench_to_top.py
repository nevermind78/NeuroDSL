import json
from pathlib import Path

path = Path('notebook/notebook.ipynb')
with path.open('r', encoding='utf-8') as f:
    nb = json.load(f)

markers = ['Multi-Variant Professional Benchmarks for NeuroDSL',
           'Comprehensive NeuroDSL benchmark cell',
           'run_variants()']

found_idx = None
for i, cell in enumerate(nb['cells']):
    if cell.get('cell_type') == 'code':
        src = ''.join(cell.get('source', []))
        if any(m in src for m in markers):
            found_idx = i
            break

if found_idx is None:
    print('NOT FOUND')
else:
    cell = nb['cells'].pop(found_idx)
    nb['cells'].insert(0, cell)
    with path.open('w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f'MOVED from {found_idx} to 0')
    print('FIRST LINES:\n', ''.join(nb['cells'][0].get('source', [])[:8]))
