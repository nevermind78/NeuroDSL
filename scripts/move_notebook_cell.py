import json
from pathlib import Path

path = Path('notebook/notebook.ipynb')
with path.open('r', encoding='utf-8') as f:
    nb = json.load(f)

source_text = 'Comprehensive NeuroDSL benchmark cell'
idx = None
for i, cell in enumerate(nb['cells']):
    if cell.get('cell_type') == 'code' and any(source_text in line for line in cell.get('source', [])):
        idx = i
        break

if idx is None:
    raise ValueError('Benchmark cell not found')

cell = nb['cells'].pop(idx)
insert_at = 31
nb['cells'].insert(insert_at, cell)

with path.open('w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f'Moved benchmark cell from {idx} to {insert_at}')
