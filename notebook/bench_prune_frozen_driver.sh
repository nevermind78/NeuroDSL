#!/usr/bin/env bash
# Pilote de bench_prune_frozen.jl : UN PROCESSUS PAR BRAS, N lancements
# indépendants par bras, agrégation min/médiane/max ENTRE lancements.
#
# Pourquoi ce pilote existe : une première version du banc faisait tourner les
# deux bras dans le même processus (le contrôle toujours en second, héritant de
# l'état mémoire du premier) et ne lançait qu'une fois. Le bras contrôle
# rapportait alors +1.0% puis +22.8% entre deux lancements -- non reproductible.
# Ici chaque bras part d'un processus neuf et chaque chiffre est agrégé.
#
# Chronométrage valide seulement horloge GPU VERROUILLÉE :
#   nvidia-smi -lgc 1402,1402     (terminal administrateur)
#   nvidia-smi -rgc               (à la fin, pour libérer l'horloge)
#
# USAGE : bash notebook/bench_prune_frozen_driver.sh [N_LANCEMENTS]
set -u
cd "$(dirname "$0")/.."

N="${1:-5}"
LOG="notebook/bench_prune_frozen_results.txt"
: > "$LOG"

echo "horloge GPU : $(nvidia-smi --query-gpu=clocks.gr --format=csv,noheader 2>/dev/null || echo inconnue)"
echo "N lancements par bras : $N"

for arm in favorable controle; do
  for i in $(seq 1 "$N"); do
    printf '  %-10s lancement %d/%d ... ' "$arm" "$i" "$N"
    line=$(julia --project=. notebook/bench_prune_frozen.jl "$arm" 2>/dev/null | grep '^RESULT')
    if [ -z "$line" ]; then echo "ECHEC"; continue; fi
    echo "$line" >> "$LOG"
    echo "ok"
  done
done

python - "$LOG" <<'PYEOF'
import sys, statistics as st
rows={}
for ln in open(sys.argv[1]):
    if not ln.startswith('RESULT'): continue
    d=dict(kv.split('=',1) for kv in ln.split()[1:])
    rows.setdefault(d['arm'],[]).append(d)

def agg(vals):
    v=sorted(float(x) for x in vals)
    # Étendue rapportée comme max et NON comme quantile : à n<=5, l'indice
    # round(0.9*(n-1)) EST le dernier, donc un "p90" ainsi calculé n'est qu'un
    # maximum sous une étiquette trompeuse.
    return min(v), st.median(v), v[-1]

for arm, rs in rows.items():
    print(f"\n=== {arm}  ({len(rs)} lancements indépendants)")
    pp={r['pruned_pct'] for r in rs}
    bw={(r['bw_full'], r['bw_pruned']) for r in rs}
    me=max(float(r['max_err']) for r in rs)
    print(f"  travail élagué        : {sorted(pp)}  "
          f"{'(DÉTERMINISTE)' if len(pp)==1 else '(VARIABLE !)'}")
    print(f"  nœuds backwarded      : {sorted(bw)}")
    print(f"  erreur max gradient   : {me:.3e}")
    for k,lab in (('t_full_ms','temps sans élagage '),('t_pruned_ms','temps avec élagage '),
                  ('gain_pct','GAIN EN TEMPS (%)  ')):
        mn,md,mx=agg([r[k] for r in rs])
        spread = (mx-mn)/md*100 if md else 0.0
        print(f"  {lab}  min={mn:8.4f}  médiane={md:8.4f}  max={mx:8.4f}   "
              f"étendue={spread:5.1f}% de la médiane")
PYEOF
