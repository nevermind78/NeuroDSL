#!/usr/bin/env bash
# Pilote de bench_retrieval_locality.jl : UN (rembourrage, cible) PAR PROCESSUS.
#
# `_ancestors_of!` MET EN CACHE son resultat par cible, et `topo_order!` aussi.
# Mesurer plusieurs cibles dans un meme processus melangerait des etats de cache
# differents ; un processus neuf par point supprime la question.
#
# Aucun chronometrage : la quantite mesuree est un comptage de noeuds. Ni horloge
# verrouillee ni GPU necessaires.
#
# USAGE : bash notebook/bench_retrieval_locality_driver.sh
set -u
cd "$(dirname "$0")/.."
LOG="notebook/bench_retrieval_locality_results.txt"
: > "$LOG"

run() {
  printf '  pad=%s %-7s ... ' "$1" "$2"
  line=$(julia --project=. notebook/bench_retrieval_locality.jl "$1" "$2" 2>/dev/null | grep '^RESULT')
  if [ -z "$line" ]; then echo "ECHEC"; return; fi
  echo "$line" >> "$LOG"; echo "ok"
}

for pad in 1 2 4; do
  for t in input layer1 layer6 output; do run "$pad" "$t"; done
done
echo "  -- repetitions pour le critere de determinisme --"
for i in 2 3 4 5; do run 4 layer1; done

python - "$LOG" <<'PYEOF'
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
rows=[]
for ln in open(sys.argv[1]):
    if not ln.startswith('RESULT'): continue
    rows.append(dict(kv.split('=',1) for kv in ln.split()[1:]))
grid={}
for r in rows: grid.setdefault((int(r['pad']), r['target']), []).append(r)

print(f"\n{'cible':<8}{'pad':>4}{'|V| total':>11}{'cone ancetre':>14}"
      f"{'ancien prefixe':>16}{'oracle':>8}{'accord':>8}{'gain':>9}")
print('-'*80)
for t in ('input','layer1','layer6','output'):
    for pad in (1,2,4):
        rs=grid.get((pad,t))
        if not rs: continue
        r=rs[0]
        print(f"{t:<8}{pad:>4}{r['n_total']:>11}{r['new_cost']:>14}"
              f"{r['old_cost']:>16}{r['oracle']:>8}{r['agree']:>8}{float(r['speedup']):>8.2f}x")

print("\n== CRITERES PRE-ENREGISTRES ==")
print(f"(c) accord avec l'oracle independant : "
      f"{sum(r['agree']=='true' for r in rows)}/{len(rows)}"
      f"  -> premiere verification de _ancestors_of! (zero test dans le depot)")
rep=grid.get((4,'layer1'),[])
s={r['new_cost'] for r in rep}
print(f"(a) determinisme sur {len(rep)} lancements de (pad=4,layer1) : "
      f"{'OUI' if len(s)==1 else 'NON'}  valeurs={sorted(s)}")
print("(b) invariance du cone ancetre sous rembourrage, ancien cout croissant :")
for t in ('layer1','layer6'):
    nw={p: grid[(p,t)][0]['new_cost'] for p in (1,2,4) if (p,t) in grid}
    od={p: grid[(p,t)][0]['old_cost'] for p in (1,2,4) if (p,t) in grid}
    ok = len(set(nw.values()))==1 and len(set(od.values()))>1
    print(f"    {t:<7} cone={nw}  prefixe={od}  -> {'CONFORME' if ok else 'NON CONFORME'}")
c=grid.get((1,'output'))
if c:
    sp=float(c[0]['speedup'])
    print(f"(d) controle negatif (sortie du modele) : gain={sp:.3f}x"
          f"  -> {'<= 1.10 : aucun gain, OUI' if sp<=1.10 else 'gain inattendu, NON'}")
PYEOF
