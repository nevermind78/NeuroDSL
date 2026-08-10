#!/usr/bin/env bash
# Pilote de bench_fusion.jl : un processus par (device, dim), N lancements,
# agrégation min/médiane/max ENTRE lancements.
#
# But : délimiter le RÉGIME dans lequel la fusion apporte quelque chose. La
# fusion appelle le GEMM du vendeur puis un épilogue élémentaire (src/dispatch.jl:340)
# -- elle n'économise aucun FLOP, seulement un nœud de graphe, son dispatch et son
# buffer. Le gain doit donc être grand à petite taille et s'évanouir à grande
# taille. Un pourcentage unique (l'article annonçait "37%") n'a pas de sens sans
# son régime.
#
# Chronométrage GPU valide seulement horloge verrouillée :
#   nvidia-smi -lgc 1402,1402   (terminal administrateur)   /   nvidia-smi -rgc
#
# USAGE : bash notebook/bench_fusion_driver.sh [N_LANCEMENTS]
set -u
cd "$(dirname "$0")/.."

N="${1:-3}"
LOG="notebook/bench_fusion_results.txt"
: > "$LOG"

echo "horloge GPU : $(nvidia-smi --query-gpu=clocks.gr --format=csv,noheader 2>/dev/null || echo inconnue)"
echo "N lancements par point : $N"

for d in cpu gpu; do
  for dim in 32 128 512; do
    for i in $(seq 1 "$N"); do
      printf '  %-3s dim=%-4s lancement %d/%d ... ' "$d" "$dim" "$i" "$N"
      line=$(julia --project=. notebook/bench_fusion.jl "$d" "$dim" 2>/dev/null | grep '^RESULT')
      if [ -z "$line" ]; then echo "ECHEC"; continue; fi
      echo "$line" >> "$LOG"; echo "ok"
    done
  done
done

python - "$LOG" <<'PYEOF'
import sys, statistics as st
rows={}
for ln in open(sys.argv[1]):
    if not ln.startswith('RESULT'): continue
    d=dict(kv.split('=',1) for kv in ln.split()[1:])
    rows.setdefault((d['dev'], int(d['dim'])),[]).append(d)

print(f"\n{'device':<7}{'dim':>5}{'règles':>12}{'fusions':>9}{'err max':>11}"
      f"{'non fus.(ms)':>14}{'fusionné(ms)':>14}{'GAIN médian':>13}{'  [min..max]':>16}")
print('-'*105)
for (dv,dim), rs in sorted(rows.items(), key=lambda k:(k[0][0], k[0][1])):
    g=sorted(float(r['gain_pct']) for r in rs)
    # Étendue rapportée comme [min..max] et NON comme un quantile : à N<=5
    # lancements, l'indice round(0.9*(N-1)) EST le dernier, donc un "p90"
    # calculé ainsi ne serait qu'un maximum sous une étiquette trompeuse.
    max_=g[-1]
    tu=st.median(float(r['t_unfused_ms']) for r in rs)
    tf=st.median(float(r['t_fused_ms']) for r in rs)
    err=max(float(r['max_err']) for r in rs)
    rr=f"{rs[0]['rules_unfused']}->{rs[0]['rules_fused']}"
    print(f"{dv:<7}{dim:>5}{rr:>12}{rs[0]['fused_applied']:>9}{err:>11.2e}"
          f"{tu:>14.4f}{tf:>14.4f}{st.median(g):>12.1f}%{f'  [{min(g):.1f}..{max_:.1f}]':>16}")
print("\nerr max = ecart max entre sortie fusionnee et non fusionnee (correction avant vitesse)")
PYEOF
