import json
from pathlib import Path

path = Path('notebook/notebook.ipynb')
with path.open('r', encoding='utf-8') as f:
    nb = json.load(f)
print('total cells', len(nb['cells']))
for i in range(25, 40):
    cell = nb['cells'][i]
    cell_type = cell.get('cell_type')
    print('INDEX', i, 'TYPE', cell_type, 'EXEC', cell.get('execution_count'))
    if cell_type == 'code':
        src = ''.join(cell.get('source', []))
        print(src[:400].replace('\n', '\\n'))
        if 'Benchmark' in src or 'Lancement du Benchmark' in src or 'Comprehensive' in src:
            print('*** MATCH ***')
    else:
        print('--- MARKDOWN or other ---')
    print('-----')
