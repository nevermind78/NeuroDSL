#!/usr/bin/env bash
# Pilote de bench_invalidation_locality.jl : UN (rembourrage, site) PAR PROCESSUS.
#
# L'isolation est obligatoire ici, pas prudentielle : `_invalidate_downstream!`
# MUTE le graphe (`valid=false`). Mesurer deux sites de suite dans un même
# processus fausserait le second, puisqu'il partirait d'un graphe déjà
# partiellement invalide.
#
# Grille principale : 3 rembourrages x 4 sites.
# Plus 4 répétitions de (pad=1, layer6) pour porter ce point a 5 lancements
# independants -- le critere pre-enregistre (a) porte sur le determinisme du
# comptage, il suffit de l'etablir sur un point, la structure etant identique
# d'un lancement a l'autre.
#
# Aucun chronometrage : la quantite mesuree est un comptage de noeuds. Pas
# besoin d'horloge GPU verrouillee, pas besoin de GPU du tout.
#
# USAGE : bash notebook/bench_invalidation_locality_driver.sh
set -u
cd "$(dirname "$0")/.."
LOG="notebook/bench_invalidation_locality_results.txt"
: > "$LOG"

run() {
  printf '  pad=%s %-8s ... ' "$1" "$2"
  line=$(julia --project=. notebook/bench_invalidation_locality.jl "$1" "$2" 2>/dev/null | grep '^RESULT')
  if [ -z "$line" ]; then echo "ECHEC"; return; fi
  echo "$line" >> "$LOG"; echo "ok"
}

for pad in 1 2 4; do
  for site in input layer1 layer6 layer12; do run "$pad" "$site"; done
done
echo "  -- repetitions pour le critere de determinisme --"
for i in 2 3 4 5; do run 1 layer6; done

python - "$LOG" <<'PYEOF'
import sys, collections
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
rows=[]
for ln in open(sys.argv[1]):
    if not ln.startswith('RESULT'): continue
    rows.append(dict(kv.split('=',1) for kv in ln.split()[1:]))

grid={}
for r in rows: grid.setdefault((int(r['pad']), r['site']), []).append(r)

print(f"\n{'site':<9}{'pad':>4}{'|V| total':>11}{'invalidables':>14}"
      f"{'traverses':>11}{'oracle':>8}{'accord':>8}{'part des invalidables':>23}")
print('-'*90)
for site in ('input','layer1','layer6','layer12'):
    for pad in (1,2,4):
        rs=grid.get((pad,site))
        if not rs: continue
        r=rs[0]
        print(f"{site:<9}{pad:>4}{r['n_total']:>11}{r['n_invalidable']:>14}"
              f"{r['visited']:>11}{r['oracle']:>8}{r['agree']:>8}"
              f"{float(r['ratio_invalidable'])*100:>21.2f}%")

print("\n== CRITERES PRE-ENREGISTRES ==")
# (c) accord oracle sur 100% des sites
allagree = all(r['agree']=='true' for r in rows)
print(f"(c) accord avec l'oracle independant sur tous les points : {'OUI' if allagree else 'NON'}"
      f"  ({sum(r['agree']=='true' for r in rows)}/{len(rows)})")
# (a) determinisme
rep=grid.get((1,'layer6'),[])
cnt={r['visited'] for r in rep}
print(f"(a) comptage deterministe sur {len(rep)} lancements de (pad=1,layer6) : "
      f"{'OUI' if len(cnt)==1 else 'NON'}  valeurs={sorted(cnt)}")
# (b) invariance sous rembourrage
print("(b) invariance du comptage sous rembourrage (le critere decisif) :")
for site in ('layer1','layer6','layer12'):
    v={pad: grid[(pad,site)][0]['visited'] for pad in (1,2,4) if (pad,site) in grid}
    tot={pad: grid[(pad,site)][0]['n_total'] for pad in (1,2,4) if (pad,site) in grid}
    ok = len(set(v.values()))==1
    print(f"    {site:<8} traverses={v}  |V|={tot}  -> {'IDENTIQUE' if ok else 'A VARIE (FALSIFIE)'}")
# (d) controle negatif
c=grid.get((1,'input'))
if c:
    ri=float(c[0]['ratio_invalidable']); rm=float(c[0]['ratio_model'])
    print(f"(d) controle negatif (site = entree du graphe) :")
    print(f"    part des noeuds INVALIDABLES atteints : {ri*100:.2f}%  -> {'>= 95% : OUI' if ri>=0.95 else '< 95% : NON'}")
    print(f"    part de |V| brut (denominateur du critere tel qu'ecrit) : {rm*100:.2f}%"
          f"  -> {'>= 95%' if rm>=0.95 else '< 95%, cf. note sur les parametres-feuilles'}")
PYEOF
