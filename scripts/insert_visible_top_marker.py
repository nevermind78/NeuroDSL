import json
from pathlib import Path

p = Path('notebook/notebook.ipynb')
nb = json.loads(p.read_text(encoding='utf-8'))

marker = {
    'cell_type': 'markdown',
    'metadata': {'language': 'markdown'},
    'source': [
        '## NEURODSL BENCHMARKS (Inserted for visibility)\n',
        '\n',
        '- JUMP TO: run the next code cell to execute the multi-variant benchmarks.\n',
        '- If you do not see the cell, please **close and reopen** the notebook editor to reload from disk.\n'
    ]
}

nb['cells'].insert(0, marker)
with p.open('w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print('Inserted visible markdown marker at top (index 0)')
