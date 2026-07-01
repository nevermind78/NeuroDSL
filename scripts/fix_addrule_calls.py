import json
from pathlib import Path
p = Path('notebook/notebook.ipynb')
nb = json.loads(p.read_text(encoding='utf-8'))
cell = nb['cells'][1]
src = cell.get('source', [])
old = '    NeuroDSL.addrule!(g, :loss, [prev, :y], :mse_loss)\n'
new = '    NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:loss, [prev, :y], :mse_loss))\n'
changed = False
for i,line in enumerate(src):
    if line == old:
        src[i] = new
        changed = True
        break
if not changed:
    # try to replace without exact indent
    for i,line in enumerate(src):
        if 'addrule!(g, :loss' in line and ':mse_loss' in line:
            src[i] = line.replace('NeuroDSL.addrule!(g, :loss, [prev, :y], :mse_loss)', 'NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:loss, [prev, :y], :mse_loss))')
            changed = True
            break

if changed:
    cell['source'] = src
    nb['cells'][1] = cell
    p.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
    print('Patched cell 1: addrule! fixed')
else:
    print('No matching line found to patch')
