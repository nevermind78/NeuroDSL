"""Construit ripple_matrix_v2_prompts.json : un pool ELARGI de triplets de
faits pour rejouer la matrice de replication du ripple-effect, en corrigeant
le defaut de conception diagnostique dans v1 (ripple_matrix_experiment.jl) :
3 triplets sur 5 avaient ete BLOQUES a la verification factuelle non pas
faute de connaissance du modele, mais parce que le token de reponse attendu
etait devine A L'AVANCE (ex. " Shakespeare" pour une reponse dont la
completion naturelle du modele est probablement " William" -- le prenom
avant le nom de famille) plutot que lu sur ce que le modele repond REELLEMENT.

CE QUI CHANGE, ET CE QUI NE CHANGE PAS
---------------------------------------
Ne change pas : le protocole causal lui-meme (patch position-masque sur les
28 couches vs baseline cosinus, memes seuils ratio>=2, meme regle
mediane>=3 ET 4/5>=2 pour "REPLICATES"), le principe de ne jamais reformuler
et relancer un triplet DEJA teste et bloque (les 3 triplets bloques de v1 ne
sont PAS repris ici sous un nom identique -- ce sont des faits, des
formulations et des paires differentes, nommees explicitement _v2).
Change : la verification factuelle devient EN DEUX TEMPS -- (1) calibration :
on lit ce que le modele repond REELLEMENT en contexte propre a Q1 et Q2, on
verifie SOI-MEME (dans ce script, avant tout calcul GPU) que c'est
semantiquement une reponse correcte au fait pose, PUIS on utilise CE token
comme "reponse attendue" pour la mesure causale -- au lieu de deviner un
token a l'avance et de bloquer si le modele en dit un autre qui serait
pourtant correct.

12 candidats, diversite de type de relation preservee (ville partagee, pays
partage, categorie partagee/symboles chimiques), reponses courtes a fort
risque d'etre un unique token BPE (villes, symboles chimiques, planetes) --
evite deliberement les noms de personnes complets (lecon tiree de l'echec
"Shakespeare" de v1).

USAGE
  python notebook/ripple_matrix_v2_build_prompts.py
ECRIT notebook/ripple_matrix_v2_candidates.json (candidats TOKENISES, pas
encore verifies -- la verification factuelle se fait cote Julia, avec le
vrai modele, pas ici).
"""
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from qwen_tokenize_prompts import encode  # noqa: E402  (encodeur deja verifie 3/3 cette session)

CANDIDATES = [
    # (name, full_text, obj_text, wrong_text, a1_text, a2_text)
    # full_text doit contenir EXACTEMENT UNE fois obj_text (avec son espace
    # de tete) au point de jonction ; a1_text/a2_text sont les reponses
    # HYPOTHESEES pour construire le prompt, mais la verification reelle se
    # fait cote Julia sur ce que le modele repond, pas sur ce id precis.
    ("big_ben_london_sahara_v2",
     "Big Ben is located in the city of London. In which city is "
     "Buckingham Palace located? The answer is London. What is the "
     "largest desert in the world? The answer is",
     " London", " Berlin", " London", " Sahara"),

    ("great_wall_china_helium_v2",
     "The Great Wall is located in the country of China. In which "
     "country is the Forbidden City located? The answer is China. "
     "What is the chemical symbol for helium? The answer is",
     " China", " Japan", " China", " He"),

    ("taj_mahal_india_neon_v2",
     "The Taj Mahal is located in the country of India. In which "
     "country is the Ganges River located? The answer is India. "
     "What is the chemical symbol for neon? The answer is",
     " India", " Nepal", " India", " Ne"),

    ("sydney_opera_australia_copper_v2",
     "The Sydney Opera House is located in the country of Australia. "
     "In which country is the Great Barrier Reef located? The answer "
     "is Australia. What is the chemical symbol for copper? The "
     "answer is",
     " Australia", " Canada", " Australia", " Cu"),

    ("kremlin_moscow_zinc_v2",
     "The Kremlin is located in the city of Moscow. In which city is "
     "Red Square located? The answer is Moscow. What is the chemical "
     "symbol for zinc? The answer is",
     " Moscow", " Berlin", " Moscow", " Zn"),

    ("colosseum_pantheon_jupiter_v2",
     "The Colosseum is located in the city of Rome. In which city is "
     "the Pantheon located? The answer is Rome. What is the largest "
     "planet in the solar system? The answer is",
     " Rome", " Athens", " Rome", " Jupiter"),

    ("pyramid_egypt_lithium_v2",
     "The Great Pyramid of Giza is located in the country of Egypt. "
     "In which country is the Sphinx located? The answer is Egypt. "
     "What is the chemical symbol for lithium? The answer is",
     " Egypt", " Sudan", " Egypt", " Li"),

    ("brandenburg_berlin_pacific_v2",
     "The Brandenburg Gate is located in the city of Berlin. In "
     "which city is the Berlin Wall located? The answer is Berlin. "
     "What is the largest ocean on Earth? The answer is",
     " Berlin", " Munich", " Berlin", " Pacific"),

    ("potassium_calcium_paris_v2",
     "The chemical symbol for potassium is K. What is the chemical "
     "symbol for calcium? The answer is Ca. What is the capital of "
     "France? The answer is",
     " K", " Ca", " Ca", " Paris"),

    ("machupicchu_nazca_mercury_v2",
     "Machu Picchu is located in the country of Peru. In which "
     "country is the Nazca Lines located? The answer is Peru. What "
     "is the smallest planet in the solar system? The answer is",
     " Peru", " Chile", " Peru", " Mercury"),

    ("acropolis_athens_iodine_v2",
     "The Acropolis is located in the city of Athens. In which city "
     "is the Parthenon located? The answer is Athens. What is the "
     "chemical symbol for iodine? The answer is",
     " Athens", " Rome", " Athens", " I"),

    ("cntower_toronto_argon_v2",
     "The CN Tower is located in the city of Toronto. In which city "
     "is the Rogers Centre located? The answer is Toronto. What is "
     "the chemical symbol for argon? The answer is",
     " Toronto", " Ottawa", " Toronto", " Ar"),
]

out = []
n_multitoken_obj = 0
for name, full_text, obj_text, wrong_text, a1_text, a2_text in CANDIDATES:
    full_ids = encode(full_text)
    obj_ids = encode(obj_text)
    wrong_ids = encode(wrong_text)
    a1_ids = encode(a1_text)
    a2_ids = encode(a2_text)

    # localise obj_pos : la position dans full_ids ou le PREMIER token de
    # obj_ids apparait, dans la fenetre attendue (juste apres "city/country
    # of"/"is"). On cherche la sous-sequence obj_ids dans full_ids.
    def find_sub(hay, needle):
        for i in range(len(hay) - len(needle) + 1):
            if hay[i:i + len(needle)] == needle:
                return i
        return -1

    obj_start = find_sub(full_ids, obj_ids)
    if obj_start < 0:
        print("!!! %s : objet introuvable dans full_ids -- verifier la formulation" % name)
        continue
    obj_pos = obj_start  # 0-indexe HF, position du PREMIER token de l'objet

    if len(obj_ids) != len(wrong_ids):
        n_multitoken_obj += 1
        print("[note] %s : obj_ids (%d tok) et wrong_ids (%d tok) de longueur differente "
              "-- corruption a une seule position restera bien definie (on ne remplace "
              "que le PREMIER token), mais a signaler." % (name, len(obj_ids), len(wrong_ids)))

    # positions Q1/Q2 : dernier token de full_ids EST la position de lecture
    # (juste apres "The answer is" - le modele complete a partir de la).
    # Il y a DEUX occurrences de ce pattern dans le texte (une pour Q1, deja
    # repondue dans le texte lui-meme, une pour Q2, en fin de texte a
    # completer) -- pos_Q1 = position du dernier token AVANT que la reponse
    # a Q1 soit ecrite dans le texte ; pos_Q2 = dernier token de la sequence
    # (fin de prompt, a completer).
    # On les retrouve en cherchant les DEUX sous-sequences "The answer is"
    # dans full_ids (tokenisees separement) puis en prenant la position
    # juste avant chaque reponse ecrite / juste a la fin.
    marker_ids = encode(" The answer is")
    positions = []
    i = 0
    while True:
        j = find_sub(full_ids[i:], marker_ids)
        if j < 0:
            break
        positions.append(i + j + len(marker_ids) - 1)  # position du dernier token du marqueur
        i = i + j + 1
    if len(positions) < 2:
        print("!!! %s : marqueur 'The answer is' trouve %d fois (attendu >=2) -- verifier" %
              (name, len(positions)))
        continue
    pos_Q1 = positions[0]
    pos_Q2 = len(full_ids) - 1  # fin de sequence = point de lecture pour Q2

    out.append({
        "name": name,
        "full_text": full_text,          # garde pour lisibilite humaine (v1 ne l'avait pas -- corrige ici)
        "full_ids": full_ids,
        "obj_pos": obj_pos,
        "pos_Q1": pos_Q1,
        "pos_Q2": pos_Q2,
        "obj_first_id": obj_ids[0],
        "wrong_first_id": wrong_ids[0],
        "a1_hypothesis_id": a1_ids[0],   # hypothese de depart -- PAS utilisee pour bloquer,
        "a2_hypothesis_id": a2_ids[0],   # seulement pour affichage/comparaison a la calibration
        "a1_hypothesis_text": a1_text,
        "a2_hypothesis_text": a2_text,
    })

dest = os.path.join(HERE, "ripple_matrix_v2_candidates.json")
io.open(dest, "w", encoding="utf-8").write(json.dumps(out, ensure_ascii=False, indent=1))
print("\n%d/%d candidats construits (%d avec obj/wrong de longueur de token differente)." %
      (len(out), len(CANDIDATES), n_multitoken_obj))
print("Ecrit :", dest)
