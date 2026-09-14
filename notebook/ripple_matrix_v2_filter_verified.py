"""Filtre ripple_matrix_v2_candidates.json en un jeu VERIFIE, a partir de la
calibration reelle (ripple_matrix_v2_calibration.json).

REGLE DE VERIFICATION, DISCLOSED COMME POST-HOC
------------------------------------------------
Cette regle a ete choisie APRES avoir vu les donnees de calibration (le
clivage net a moins de 1 nat / plus de 2.7 nats saute aux yeux une fois les
chiffres devant soi) -- ce n'est PAS une regle preenregistree a l'aveugle,
et ce script le dit sans detour plutot que de presenter la regle comme si
elle avait ete fixee d'avance :

  Un token de reponse hypothese est considere CONNU si :
    (a) c'est directement le top-1 du modele (gap = 0 nat), OU
    (b) son logit est a MOINS DE 1.0 NAT du top-1 reel (seuil rond, pas
        ajuste sur les donnees pour maximiser le nombre de triplets
        recuperes -- verifie a la main : le clivage reel est net entre
        0.994 et 2.743, un seuil a 1.5 ou 2.0 nats donnerait EXACTEMENT le
        meme resultat).
  Un triplet est VERIFIE seulement si Q1 ET Q2 passent tous les deux.

Un triplet dont le top-1 reel est un token de ponctuation/remplissage
(':', ' ', ' the') mais dont le token-reponse hypothese est a moins de 1 nat
est traite comme un ARTEFACT DE FORMAT, pas une lacune de connaissance --
exactement le defaut diagnostique qui avait fait perdre 3/5 triplets dans
v1 (ripple_matrix_experiment.jl), corrige ici en verifiant sur le LOGIT de
la bonne reponse plutot que sur l'identite du token top-1.

USAGE : python notebook/ripple_matrix_v2_filter_verified.py
ECRIT : notebook/ripple_matrix_v2_verified.json
"""
import io
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
GAP_THRESHOLD = 1.0  # nats -- rond, non ajuste (voir docstring)

cal = json.load(io.open(os.path.join(HERE, "ripple_matrix_v2_calibration.json"), encoding="utf-8"))
cands = json.load(io.open(os.path.join(HERE, "ripple_matrix_v2_candidates.json"), encoding="utf-8"))
by_name = {c["name"]: c for c in cands}

verified, blocked = [], []
for r in cal:
    c = by_name[r["name"]]
    gap1 = r["q1_top1_logit"] - r["hyp1_logit"]
    gap2 = r["q2_top1_logit"] - r["hyp2_logit"]
    known1 = gap1 < GAP_THRESHOLD
    known2 = gap2 < GAP_THRESHOLD
    reason1 = "direct top-1" if r["matches_hyp1"] else ("format (gap=%.3f)" % gap1 if known1 else "REJETE (gap=%.3f)" % gap1)
    reason2 = "direct top-1" if r["matches_hyp2"] else ("format (gap=%.3f)" % gap2 if known2 else "REJETE (gap=%.3f)" % gap2)
    print("%-32s Q1: %-22s  Q2: %-22s  -> %s" %
          (r["name"], reason1, reason2, "VERIFIE" if (known1 and known2) else "BLOQUE"))

    if known1 and known2:
        verified.append({
            "name": c["name"],
            "full_ids": c["full_ids"],
            "obj_pos": c["obj_pos"],
            "pos_Q1": c["pos_Q1"],
            "pos_Q2": c["pos_Q2"],
            "obj_first_id": c["obj_first_id"],
            "wrong_first_id": c["wrong_first_id"],
            "a1_first_id": r["hyp1_id"],   # le token semantiquement correct,
            "a2_first_id": r["hyp2_id"],   # verifie a <1 nat du top-1 reel
            "gap_Q1": gap1, "gap_Q2": gap2,
        })
    else:
        blocked.append({"name": c["name"], "gap_Q1": gap1, "gap_Q2": gap2,
                        "reason": "Q1" if not known1 else "Q2"})

print("\n%d verifies, %d bloques (sur %d candidats)." % (len(verified), len(blocked), len(cal)))
dest = os.path.join(HERE, "ripple_matrix_v2_verified.json")
io.open(dest, "w", encoding="utf-8").write(json.dumps(verified, ensure_ascii=False, indent=1))
print("Ecrit :", dest)
