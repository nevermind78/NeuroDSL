#!/usr/bin/env bash
# Pilote de bench_fusion_memory.jl : UN BRAS PAR PROCESSUS.
#
# L'isolation par processus est indispensable ici : les deux graphes (fusionné et
# non fusionné) coexistant dans un même processus, les buffers du premier bras
# gonfleraient la baseline mémoire du second et la comparaison n'aurait aucun
# sens. C'est la même leçon que bench_prune_frozen.jl, mais avec un enjeu plus
# direct puisque la quantité mesurée EST la mémoire.
#
# USAGE : bash notebook/bench_fusion_memory_driver.sh
set -u
cd "$(dirname "$0")/.."
LOG="notebook/bench_fusion_memory_results.txt"
: > "$LOG"

for d in cpu gpu; do
  for dim in 32 128 512; do
    for arm in unfused fused; do
      printf '  %-3s dim=%-4s %-8s ... ' "$d" "$dim" "$arm"
      line=$(julia --project=. notebook/bench_fusion_memory.jl "$d" "$dim" "$arm" 2>/dev/null | grep '^RESULT')
      if [ -z "$line" ]; then echo "ECHEC"; continue; fi
      echo "$line" >> "$LOG"; echo "ok"
    done
  done
done

python - "$LOG" <<'PYEOF'
import sys
rows={}
for ln in open(sys.argv[1]):
    if not ln.startswith('RESULT'): continue
    d=dict(kv.split('=',1) for kv in ln.split()[1:])
    rows[(d['dev'], int(d['dim']), d['arm'])]=d

print(f"\n{'device':<7}{'dim':>5}{'regles':>16}{'noeuds act.':>14}"
      f"{'activations (Mo)':>19}{'gain':>8}{'   baseline GPU (Mo)':>21}{'gain':>8}")
print('-'*100)
for dv in ('cpu','gpu'):
    for dim in (32,128,512):
        u=rows.get((dv,dim,'unfused')); f=rows.get((dv,dim,'fused'))
        if not u or not f: continue
        ua,fa=int(u['act_bytes'])/1024**2, int(f['act_bytes'])/1024**2
        ub,fb=float(u['baseline_mb']), float(f['baseline_mb'])
        ga=100*(ua-fa)/ua if ua else 0
        gb=(100*(ub-fb)/ub) if ub else 0
        bl = f"{ub:8.2f}->{fb:6.2f}" if dv=='gpu' else "         n/a"
        gbs = f"{gb:6.1f}%" if dv=='gpu' else "     -"
        print(f"{dv:<7}{dim:>5}{u['rules']+'->'+f['rules']:>16}{u['n_act']+'->'+f['n_act']:>14}"
              f"{f'{ua:8.3f}->{fa:6.3f}':>19}{ga:7.1f}%{bl:>21}{gbs:>8}")
        for tag,r in (('unfused',u),('fused',f)):
            if r['stable']!='true': print(f"    ATTENTION pic instable : {dv} dim={dim} {tag}")
print("\nactivations = octets residents des valeurs de noeuds NON parametres (structurel, deterministe)")
print("baseline GPU = memoire residente du pool CUDA apres warm-up (pic absolu == baseline : graphe persistant)")
PYEOF
