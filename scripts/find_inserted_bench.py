from pathlib import Path
path = Path('notebook/notebook.ipynb')
text = path.read_text(encoding='utf-8')
needle = 'Comprehensive NeuroDSL benchmark cell'
idx = text.find(needle)
print('found at', idx)
if idx != -1:
    start = max(0, idx-200)
    end = min(len(text), idx+400)
    print(text[start:end])
else:
    for other in ['Lancement du Benchmark complet...', 'samples=100 evals=3', '@printf("Allocated FWD']:
        print(other, '->', text.find(other))
