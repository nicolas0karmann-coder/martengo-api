# api.py — Backend Flask pour Martengo Prediction
# Déploiement : Railway / Render / Heroku

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from io import StringIO
import os
import re
import pickle
import requests as http_requests
from sklearn.ensemble import HistGradientBoostingClassifier

app = Flask(__name__)
CORS(app)

# ============================================================
# DONNÉES INITIALES
# ============================================================
CSV_INITIAL = """date,numero,note,rapport,rang_arrivee,score_cible
2026-01-13,8,16,4.6,1,7
2026-01-13,15,6,108,2,6
2026-01-13,10,16,5.8,3,5
2026-01-13,3,13,16,4,4
2026-01-13,2,13,8.2,5,3
2026-01-13,12,16,12,8,0
2026-01-13,9,15,7.6,8,0
2026-01-13,14,15,8.3,8,0
2026-01-13,11,13,15,8,0
2026-01-13,1,12,94,8,0
2026-01-13,7,12,26,8,0
2026-01-13,13,12,22,8,0
2026-01-13,6,6,148,8,0
2026-01-13,4,4,191,8,0
2026-01-13,16,4,162,8,0
2026-01-15,1,18,3.2,1,7
2026-01-15,6,15,2.9,2,6
2026-01-15,8,10,31,3,5
2026-01-15,3,12,20,4,4
2026-01-15,10,10,43,5,3
2026-01-15,4,14,32,8,0
2026-01-15,13,13,8.8,8,0
2026-01-15,12,12,18,8,0
2026-01-15,7,11,30,8,0
2026-01-15,5,9,64,8,0
2026-01-15,15,9,41,8,0
2026-01-15,14,8,105,8,0
2026-01-15,2,7,142,8,0
2026-01-15,9,7,41,8,0
2026-01-15,11,7,111,8,0
2026-01-15,16,7,71,8,0
2026-01-17,8,17,3.7,1,7
2026-01-17,3,10,59,2,6
2026-01-17,13,16,7.7,3,5
2026-01-17,11,15,11,4,4
2026-01-17,10,10,93,5,3
2026-01-17,5,18,4.8,8,0
2026-01-17,4,17,7.3,8,0
2026-01-17,7,15,26,8,0
2026-01-17,6,14,14,8,0
2026-01-17,14,14,42,8,0
2026-01-17,1,13,21,8,0
2026-01-17,9,9,58,8,0
2026-01-17,12,9,36,8,0
2026-01-17,2,8,77,8,0
2026-01-22,6,15,7.8,1,7
2026-01-22,1,11,6.3,2,6
2026-01-22,9,16,9.8,3,5
2026-01-22,8,12,28,4,4
2026-01-22,3,14,7.8,5,3
2026-01-22,11,15,5.8,8,0
2026-01-22,2,14,40,8,0
2026-01-22,13,13,34,8,0
2026-01-22,15,12,49,8,0
2026-01-22,5,11,22,8,0
2026-01-22,12,11,5.9,8,0
2026-01-22,16,11,22,8,0
2026-01-22,14,10,0,8,0
2026-01-22,10,9,120,8,0
2026-01-22,4,7,129,8,0
2026-01-22,7,3,187,8,0
2026-01-25,13,13,35,1,7
2026-01-25,11,17,5.7,2,6
2026-01-25,6,16,6.7,3,5
2026-01-25,10,16,10,4,4
2026-01-25,12,12,44,5,3
2026-01-25,18,18,6.5,8,0
2026-01-25,14,15,14,8,0
2026-01-25,5,14,38,8,0
2026-01-25,2,13,5.2,8,0
2026-01-25,9,12,38,8,0
2026-01-25,17,12,19,8,0
2026-01-25,1,10,79,8,0
2026-01-25,7,10,31,8,0
2026-01-25,8,10,103,8,0
2026-01-25,15,10,68,8,0
2026-01-25,16,10,107,8,0
2026-01-25,3,8,60,8,0
2026-01-25,4,6,125,8,0
2026-01-27,10,13,12,1,7
2026-01-27,11,17,10,2,6
2026-01-27,2,9,35,3,5
2026-01-27,14,12,7,4,4
2026-01-27,9,14,15,5,3
2026-01-27,5,10,74,6,2
2026-01-27,3,16,4.7,7,1
2026-01-27,1,13,29,8,0
2026-01-27,4,9,77,8,0
2026-01-27,6,11,24,8,0
2026-01-27,7,11,13,8,0
2026-01-27,8,9,111,8,0
2026-01-27,12,16,8.2,8,0
2026-01-27,13,15,6.5,8,0
2026-01-27,15,5,135,8,0
2026-01-27,16,9,0,8,0
2026-01-29,3,12,17,1,7
2026-01-29,10,17,6.4,2,6
2026-01-29,15,13,10,3,5
2026-01-29,16,13,8.7,4,4
2026-01-29,8,11,22,5,3
2026-01-29,4,8,93,6,2
2026-01-29,7,16,8.5,7,1
2026-01-29,12,19,4.7,8,0
2026-01-29,13,17,0,8,0
2026-01-29,9,13,0,8,0
2026-01-29,14,13,35,8,0
2026-01-29,5,12,0,8,0
2026-01-29,6,12,21,8,0
2026-01-29,1,9,5.7,8,0
2026-01-29,11,7,151,8,0
2026-01-29,2,4,112,8,0
2026-01-31,13,16,10,1,7
2026-01-31,8,16,11,2,6
2026-01-31,9,12,29,3,5
2026-01-31,1,11,30,4,4
2026-01-31,12,12,9.5,5,3
2026-01-31,16,16,14,6,2
2026-01-31,7,12,24,7,1
2026-01-31,11,16,3.7,8,0
2026-01-31,3,15,7.9,8,0
2026-01-31,6,13,31,8,0
2026-01-31,15,13,9.7,8,0
2026-01-31,14,12,17,8,0
2026-01-31,2,10,58,8,0
2026-01-31,5,9,100,8,0
2026-01-31,10,5,156,8,0
2026-01-31,4,4,114,8,0
2026-02-01,7,11,50,1,7
2026-02-01,11,16,4.9,2,6
2026-02-01,4,13,7.1,3,5
2026-02-01,6,14,14,4,4
2026-02-01,5,10,136,5,3
2026-02-01,15,16,16,6,2
2026-02-01,1,11,18,7,1
2026-02-01,14,18,6.6,8,0
2026-02-01,12,17,6.1,8,0
2026-02-01,9,16,6.4,8,0
2026-02-01,10,11,70,8,0
2026-02-01,13,11,75,8,0
2026-02-01,3,9,59,8,0
2026-02-01,8,8,67,8,0
2026-02-01,2,5,155,8,0
2026-02-03,3,15,3.8,1,7
2026-02-03,7,15,5.1,3,5
2026-02-03,2,14,17,8,0
2026-02-03,6,13,7.7,8,0
2026-02-03,8,13,7.2,8,0
2026-02-03,11,13,21,6,2
2026-02-03,4,12,14,4,4
2026-02-03,5,12,28,7,1
2026-02-03,1,10,76,8,0
2026-02-03,9,10,54,8,0
2026-02-03,12,10,22,8,0
2026-02-03,14,9,18,5,3
2026-02-03,10,8,63,2,6
2026-02-03,13,7,65,8,0
2026-02-05,7,18,5.1,3,5
2026-02-05,8,16,11,4,4
2026-02-05,9,15,3,1,7
2026-02-05,3,14,16,8,0
2026-02-05,4,13,18,6,2
2026-02-05,11,12,20,5,3
2026-02-05,14,12,7.1,2,6
2026-02-05,1,11,27,8,0
2026-02-05,2,11,77,8,0
2026-02-05,10,11,33,8,0
2026-02-05,13,10,30,7,1
2026-02-05,5,9,95,8,0
2026-02-05,12,8,45,8,0
2026-02-05,6,7,54,8,0
2026-02-05,16,7,63,8,0
2026-02-06,15,18,0,8,0
2026-02-06,16,16,0,7,1
2026-02-06,5,15,0,5,3
2026-02-06,10,15,0,8,0
2026-02-06,13,15,0,8,0
2026-02-06,14,15,0,2,6
2026-02-06,8,14,0,8,0
2026-02-06,4,13,0,8,0
2026-02-06,2,12,0,3,5
2026-02-06,7,12,0,4,4
2026-02-06,1,11,0,8,0
2026-02-06,9,11,0,1,7
2026-02-06,11,11,0,6,2
2026-02-06,12,11,0,8,0
2026-02-06,6,9,0,8,0
2026-02-06,3,3,0,8,0
2026-02-07,6,14,14,1,7
2026-02-07,3,14,20,2,6
2026-02-07,7,18,7.3,3,5
2026-02-07,15,10,41,4,4
2026-02-07,2,18,3.1,5,3
2026-02-07,4,11,45,6,2
2026-02-07,1,13,30,7,1
2026-02-07,5,14,8.6,8,0
2026-02-07,16,13,4.2,8,0
2026-02-07,8,10,33,8,0
2026-02-07,14,8,86,8,0
2026-02-07,9,7,109,8,0
2026-02-07,11,6,93,8,0
2026-02-07,13,6,191,8,0
2026-02-07,10,5,75,8,0
2026-02-07,12,4,183,8,0
2026-02-08,10,6,14,1,7
2026-02-08,5,18,4.3,2,6
2026-02-08,1,15,5,3,5
2026-02-08,4,15,10,4,4
2026-02-08,3,16,9,5,3
2026-02-08,2,14,19,6,2
2026-02-08,12,6,80,7,1
2026-02-08,6,17,11,8,0
2026-02-08,9,14,10,8,0
2026-02-08,7,12,21,8,0
2026-02-08,8,12,16,8,0
2026-02-08,11,7,102,8,0
2026-02-08,13,2,40,8,0
2026-02-10,12,10,25,1,7
2026-02-10,9,15,5.8,2,6
2026-02-10,5,13,27,3,5
2026-02-10,6,13,30,4,4
2026-02-10,11,16,10,5,3
2026-02-10,1,6,41,8,0
2026-02-10,2,10,47,8,0
2026-02-10,3,18,7.9,8,0
2026-02-10,4,5,69,8,0
2026-02-10,7,18,4,8,0
2026-02-10,8,10,13,8,0
2026-02-10,10,10,42,8,0
2026-02-10,13,16,5.3,8,0
2026-02-11,11,17,5.4,1,7
2026-02-11,3,8,20,2,6
2026-02-11,8,12,21,3,5
2026-02-11,9,12,13,4,4
2026-02-11,13,10,72,5,3
2026-02-11,10,17,22,8,0
2026-02-11,6,16,3.2,8,0
2026-02-11,7,0,0,8,0
2026-02-11,15,14,7,8,0
2026-02-11,16,14,17,8,0
2026-02-11,5,13,20,8,0
2026-02-11,4,9,65,8,0
2026-02-11,14,9,27,8,0
2026-02-11,12,8,27,8,0
2026-02-11,1,7,102,8,0
2026-02-11,2,7,104,8,0
"""

# ============================================================
# ÉTAT GLOBAL
# ============================================================
df         = None
model      = None
note_mean  = None
note_std   = None
modele_abs = None
note_mean_a = None
note_std_a  = None

FEATURES = [
    'note','rapport','log_rapport',
    'note_normalisee','inverse_rapport',
    'score_valeur','rapport_over_10','valeur_brute'
]
FEATURES_ABSOLU = FEATURES + ['ratio_note_rapport']

CSV_PATH = "historique_courses.csv"

# ============================================================
# LOGIQUE ML (identique aux blocs 3 & 5)
# ============================================================
def _enrichir(df_in, nm=None, ns=None):
    d = df_in.copy()
    d['log_rapport']   = np.log1p(d['rapport'])
    if nm is None:
        nm = d['note'].mean()
        ns = d['note'].std(); ns = ns if ns != 0 else 1.0
    d['note_normalisee']    = (d['note'] - nm) / ns
    d['inverse_rapport']    = 1.0 / (1.0 + d['rapport'])
    d['score_valeur']       = d['note'] / (1.0 + d['log_rapport'])
    d['rapport_over_10']    = np.maximum(0, d['rapport'] - 10)
    d['valeur_brute']       = d['note'] * d['log_rapport']
    d['ratio_note_rapport'] = d['note'] / (d['rapport'] + 1)
    return d, nm, ns


def _entrainer(df_source, features, target_col):
    d = df_source.sort_values('date')
    last = d['date'].max()
    train = d[d['date'] < last]
    clf = HistGradientBoostingClassifier(
        learning_rate=0.05, max_depth=6,
        max_iter=800, l2_regularization=0.5, random_state=42
    )
    clf.fit(train[features], train[target_col])
    return clf


def initialiser():
    global df, model, note_mean, note_std, modele_abs, note_mean_a, note_std_a

    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
    else:
        df = pd.read_csv(StringIO(CSV_INITIAL))
    df['date'] = pd.to_datetime(df['date'])

    # Modèle principal (cote >= 10)
    df_p = df.copy()
    df_p['target'] = ((df_p['rapport'] >= 10) & (df_p['rang_arrivee'] <= 3)).astype(int)
    df_p, note_mean, note_std = _enrichir(df_p)
    model = _entrainer(df_p, FEATURES, 'target')

    # Modèle absolu
    df_a = df.copy()
    df_a['target_absolu'] = (df_a['rang_arrivee'] <= 3).astype(int)
    df_a, note_mean_a, note_std_a = _enrichir(df_a)
    modele_abs = _entrainer(df_a, FEATURES_ABSOLU, 'target_absolu')

    print(f"✅ Modèles entraînés sur {len(df)} lignes / {df['date'].nunique()} courses")


# ============================================================
# ROUTES API
# ============================================================
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "courses": int(df['date'].nunique()) if df is not None else 0})


@app.route('/predict', methods=['POST'])
def predict():
    """
    Body JSON : { "chevaux": [{"numero":1,"note":15,"rapport":12.5}, ...] }
    """
    data = request.get_json()
    chevaux = data.get('chevaux', [])
    if not chevaux:
        return jsonify({"error": "Aucun cheval fourni"}), 400

    df_nc = pd.DataFrame(chevaux)
    df_nc, _, _ = _enrichir(df_nc, note_mean, note_std)

    # Modèle principal
    df_nc['proba_principal'] = model.predict_proba(df_nc[FEATURES])[:, 1]

    # Modèle absolu
    df_nc['proba_absolu'] = modele_abs.predict_proba(df_nc[FEATURES_ABSOLU])[:, 1]

    # Top 3 principal (cote > 10)
    candidats = df_nc[df_nc['rapport'] > 10].copy()
    candidats = candidats.sort_values(['proba_principal','rapport'], ascending=[False,False])
    top3_principal = candidats.head(3)['numero'].tolist()

    # Top 3 absolu
    tous = df_nc.sort_values(['proba_absolu','rapport'], ascending=[False,False])
    top3_absolu = tous.head(3)['numero'].tolist()

    # Résultat complet trié par proba absolu
    tous_list = []
    for _, row in tous.iterrows():
        tous_list.append({
            "numero":          int(row['numero']),
            "note":            float(row['note']),
            "rapport":         float(row['rapport']),
            "proba_principal": round(float(row['proba_principal']) * 100, 1),
            "proba_absolu":    round(float(row['proba_absolu']) * 100, 1),
            "top3_principal":  int(row['numero']) in top3_principal,
            "top3_absolu":     int(row['numero']) in top3_absolu,
        })

    return jsonify({
        "tous": tous_list,
        "top3_principal": top3_principal,
        "top3_absolu":    top3_absolu,
    })


@app.route('/ajouter', methods=['POST'])
def ajouter():
    """
    Body JSON : {
      "date": "2026-03-01",
      "chevaux": [{"numero":1,"note":15,"rapport":12.5,"rang_arrivee":2}, ...]
    }
    """
    global df, model, note_mean, note_std, modele_abs, note_mean_a, note_std_a

    data    = request.get_json()
    date    = data.get('date')
    chevaux = data.get('chevaux', [])

    if not date or not chevaux:
        return jsonify({"error": "date et chevaux requis"}), 400

    rows = []
    for c in chevaux:
        rows.append({
            "date":         pd.to_datetime(date),
            "numero":       c['numero'],
            "note":         c['note'],
            "rapport":      c['rapport'],
            "rang_arrivee": c['rang_arrivee'],
            "score_cible":  0,
        })

    df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    df.to_csv(CSV_PATH, index=False)

    # Réentraînement
    df_p = df.copy()
    df_p['target'] = ((df_p['rapport'] >= 10) & (df_p['rang_arrivee'] <= 3)).astype(int)
    df_p, note_mean, note_std = _enrichir(df_p)
    model = _entrainer(df_p, FEATURES, 'target')

    df_a = df.copy()
    df_a['target_absolu'] = (df_a['rang_arrivee'] <= 3).astype(int)
    df_a, note_mean_a, note_std_a = _enrichir(df_a)
    modele_abs = _entrainer(df_a, FEATURES_ABSOLU, 'target_absolu')

    return jsonify({
        "message":  f"Course du {date} ajoutée et modèles réentraînés",
        "nb_lignes": len(df),
        "nb_courses": int(df['date'].nunique()),
    })


# ============================================================
# MODELE PMU — Chargement
# ============================================================
# MODELE PMU — Globals
# ============================================================
_model_pmu           = None
_features_pmu        = None
_le_driver           = None
_le_entr             = None
_driver_stats        = None
_entr_stats          = None
_duo_stats           = None
_spec_dist           = None
_spec_disc           = None
_prior_pmu           = None
_k_bayes_pmu         = None
_target_mean_pmu     = None
_target_std_pmu      = None
_ferrage_map_pmu     = None
_avis_map_pmu        = {'POSITIF': 1, 'NEUTRE': 0, 'NEGATIF': -1}
_mediane_rapport_ref = 18.0

PMU_MODEL_PATH = "model_pmu.pkl"

DISC_MUSIQUE_MAP = {'a': 0, 'm': 1, 'p': 2, 'h': 3, 's': 4, 'c': 5}
DISCIPLINE_MAP   = {'TROT_ATTELE': 0, 'TROT_MONTE': 1, 'PLAT': 2, 'OBSTACLE': 3}
CORDE_MAP        = {'CORDE_A_GAUCHE': 0, 'CORDE_A_DROITE': 1}
SEXE_MAP         = {'MALES': 0, 'FEMELLES': 1, 'MIXTE': 2}


def _charger_modele_pmu():
    global _model_pmu, _features_pmu, _le_driver, _le_entr
    global _driver_stats, _entr_stats, _duo_stats, _spec_dist, _spec_disc
    global _prior_pmu, _k_bayes_pmu
    global _target_mean_pmu, _target_std_pmu, _ferrage_map_pmu, _mediane_rapport_ref

    if not os.path.exists(PMU_MODEL_PATH):
        print("⚠️  model_pmu.pkl introuvable — endpoint /notes_pmu désactivé")
        return False
    try:
        with open(PMU_MODEL_PATH, 'rb') as f:
            pmu = pickle.load(f)
        _model_pmu           = pmu['model']
        _features_pmu        = pmu['features']
        _le_driver           = pmu['le_driver']
        _le_entr             = pmu['le_entr']
        _driver_stats        = pmu['driver_stats']
        _entr_stats          = pmu.get('entr_stats')
        _duo_stats           = pmu.get('duo_stats')
        _spec_dist           = pmu.get('spec_dist')
        _spec_disc           = pmu.get('spec_disc')
        _prior_pmu           = pmu['prior']
        _k_bayes_pmu         = pmu['k_bayes']
        _target_mean_pmu     = pmu['target_mean']
        _target_std_pmu      = pmu['target_std']
        _ferrage_map_pmu     = pmu['ferrage_map']
        _mediane_rapport_ref = pmu.get('mediane_rapport_ref', 18.0)
        v = pmu.get('version', 1)
        print(f"✅ Modèle PMU v{v} chargé ({len(_features_pmu)} features, "
              f"{len(_driver_stats)} drivers"
              + (f", {len(_duo_stats)} duos" if _duo_stats is not None else "") + ")")
        return True
    except Exception as e:
        print(f"❌ Erreur chargement model_pmu.pkl : {e}")
        return False


# ── Parseur musique v3 (format réel : position + discipline) ─
def _parser_musique_api(musique):
    from collections import Counter
    if not musique:
        return {
            'mus_nb_courses': 0, 'mus_nb_victoires': 0, 'mus_nb_podiums': 0,
            'mus_moy_classement': 99, 'mus_derniere_place': 99, 'mus_regularite': 0,
            'mus_nb_disq': 0, 'mus_taux_disq': 0.0,
            'mus_nb_tombes': 0, 'mus_nb_arretes': 0,
            'mus_tendance': 0.0, 'mus_score_pondere': 0.0,
            'mus_disc_principale': -1, 'mus_nb_disciplines': 0,
        }
    clean   = re.sub(r'\(\d+\)', '', musique).strip()
    tokens  = re.findall(r'[0-9DATRdat][amphsc]', clean)
    if not tokens:
        return {
            'mus_nb_courses': 0, 'mus_nb_victoires': 0, 'mus_nb_podiums': 0,
            'mus_moy_classement': 99, 'mus_derniere_place': 99, 'mus_regularite': 0,
            'mus_nb_disq': 0, 'mus_taux_disq': 0.0,
            'mus_nb_tombes': 0, 'mus_nb_arretes': 0,
            'mus_tendance': 0.0, 'mus_score_pondere': 0.0,
            'mus_disc_principale': -1, 'mus_nb_disciplines': 0,
        }
    entries, nb_disq, nb_tombes, nb_arretes = [], 0, 0, 0
    for tok in tokens[:10]:
        pos, disc = tok[0], tok[1].lower()
        if pos.isdigit():
            place = 10 if pos == '0' else int(pos)
        elif pos.upper() == 'D':
            place = 15; nb_disq += 1
        elif pos.upper() == 'T':
            place = 15; nb_tombes += 1
        elif pos.upper() == 'A':
            place = 15; nb_arretes += 1
        elif pos.upper() == 'R':
            place = 12
        else:
            continue
        entries.append((place, disc))
    if not entries:
        return {
            'mus_nb_courses': 0, 'mus_nb_victoires': 0, 'mus_nb_podiums': 0,
            'mus_moy_classement': 99, 'mus_derniere_place': 99, 'mus_regularite': 0,
            'mus_nb_disq': 0, 'mus_taux_disq': 0.0,
            'mus_nb_tombes': 0, 'mus_nb_arretes': 0,
            'mus_tendance': 0.0, 'mus_score_pondere': 0.0,
            'mus_disc_principale': -1, 'mus_nb_disciplines': 0,
        }
    places      = [e[0] for e in entries]
    disciplines = [e[1] for e in entries]
    nb          = len(places)
    recentes    = places[:3]
    anciennes   = places[-3:] if nb >= 6 else places
    tendance    = round(float(np.mean(anciennes) - np.mean(recentes)), 2)
    poids       = [1.0 / (i + 1) for i in range(nb)]
    score_p     = round(sum(p*(10-min(pl,10)) for p,pl in zip(poids,places))/sum(poids), 3)
    disc_counter     = Counter(disciplines)
    disc_principale  = DISC_MUSIQUE_MAP.get(disc_counter.most_common(1)[0][0], -1)
    return {
        'mus_nb_courses':      nb,
        'mus_nb_victoires':    sum(1 for p in places if p == 1),
        'mus_nb_podiums':      sum(1 for p in places if p <= 3),
        'mus_moy_classement':  round(sum(places) / nb, 2),
        'mus_derniere_place':  places[0],
        'mus_regularite':      round(sum(1 for p in places if p <= 5) / nb, 2),
        'mus_nb_disq':         nb_disq,
        'mus_taux_disq':       round(nb_disq / nb, 2),
        'mus_nb_tombes':       nb_tombes,
        'mus_nb_arretes':      nb_arretes,
        'mus_tendance':        tendance,
        'mus_score_pondere':   score_p,
        'mus_disc_principale': disc_principale,
        'mus_nb_disciplines':  len(disc_counter),
    }


def _perf_vide():
    return {
        'perf_nb': 0, 'perf_moy_classement': 99, 'perf_derniere_place': 99,
        'perf_nb_top3': 0, 'perf_taux_top3': 0.0,
        'perf_moy_rk': 0.0, 'perf_moy_gains': 0.0, 'perf_regularite': 0.0,
    }


def _fetch_performances(date_str, r_num, c_num):
    url = (f"https://offline.turfinfo.api.pmu.fr/rest/client/7/programme"
           f"/{date_str}/R{r_num}/C{c_num}/performances-detaillees")
    try:
        resp = http_requests.get(url, timeout=5)
        if resp.status_code in (400, 404, 204):
            return {}
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return {}
    result = {}
    for cheval in data.get('performancesDetaillees', []):
        num_pmu = cheval.get('numPmu')
        perfs   = cheval.get('performances', [])[:5]
        if not perfs:
            result[num_pmu] = _perf_vide(); continue
        classements, temps_list, gains_list = [], [], []
        for perf in perfs:
            cl = perf.get('ordreArrivee') or perf.get('classement')
            if cl and cl <= 15:
                classements.append(cl)
            t = perf.get('tempsObtenu') or perf.get('reductionKilometrique')
            if t and t > 0: temps_list.append(t)
            g = perf.get('gainsCourse') or perf.get('gains') or 0
            if g: gains_list.append(g)
        nb = len(classements)
        result[num_pmu] = {
            'perf_nb':             nb,
            'perf_moy_classement': round(sum(classements)/nb, 2) if nb > 0 else 99,
            'perf_derniere_place': classements[0] if classements else 99,
            'perf_nb_top3':        sum(1 for c in classements if c <= 3),
            'perf_taux_top3':      round(sum(1 for c in classements if c<=3)/nb,2) if nb>0 else 0.0,
            'perf_moy_rk':         round(sum(temps_list)/len(temps_list),1) if temps_list else 0.0,
            'perf_moy_gains':      round(sum(gains_list)/len(gains_list),1) if gains_list else 0.0,
            'perf_regularite':     round(sum(1 for c in classements if c<=5)/nb,2) if nb>0 else 0.0,
        }
    return result


def _fetch_conditions(date_str, r_num, c_num):
    url = f"https://offline.turfinfo.api.pmu.fr/rest/client/7/programme/{date_str}"
    try:
        resp = http_requests.get(url, timeout=5)
        if resp.status_code in (400, 404, 204):
            return _cond_vides()
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return _cond_vides()
    for reunion in data.get('programme', {}).get('reunions', []):
        if reunion.get('numOfficiel') == r_num or reunion.get('numReunion') == r_num:
            for course in reunion.get('courses', []):
                if course.get('numOrdre') == c_num or course.get('numExterne') == c_num:
                    return {
                        'distance':       course.get('distance', 0) or 0,
                        'montant_prix':   course.get('montantPrix', 0) or 0,
                        'discipline':     DISCIPLINE_MAP.get(course.get('discipline',''), 0),
                        'corde':          CORDE_MAP.get(course.get('corde',''), 0),
                        'condition_sexe': SEXE_MAP.get(course.get('conditionSexe',''), 2),
                        'nb_partants':    course.get('nombreDeclaresPartants', 0) or 0,
                    }
    return _cond_vides()


def _cond_vides():
    return {'distance': 0, 'montant_prix': 0, 'discipline': 0,
            'corde': 0, 'condition_sexe': 2, 'nb_partants': 0}


def _proba_to_note_api(proba_series):
    notes_raw  = np.log1p(proba_series * 10)
    notes_norm = (notes_raw - notes_raw.mean()) / (notes_raw.std() + 1e-8)
    notes_cal  = notes_norm * _target_std_pmu + _target_mean_pmu
    return notes_cal.clip(1, 20).round(0).astype(int)


@app.route('/notes_pmu', methods=['GET'])
def notes_pmu():
    """
    Calcule les notes PMU pour une course donnée.
    Paramètres GET : date (DDMMYYYY), reunion (int), course (int)
    Ex: /notes_pmu?date=05032026&reunion=1&course=5
    """
    if _model_pmu is None:
        return jsonify({"error": "Modèle PMU non disponible"}), 503

    date_str = request.args.get('date', '')
    r_num    = request.args.get('reunion', '')
    c_num    = request.args.get('course', '')

    if not date_str or not r_num or not c_num:
        return jsonify({"error": "Paramètres requis : date, reunion, course"}), 400
    try:
        r_num = int(r_num); c_num = int(c_num)
    except ValueError:
        return jsonify({"error": "reunion et course doivent être des entiers"}), 400

    # ── Conditions de course & performances détaillées ───────
    conditions = _fetch_conditions(date_str, r_num, c_num)
    perfs_map  = _fetch_performances(date_str, r_num, c_num)

    # ── Participants ──────────────────────────────────────────
    url = (f"https://offline.turfinfo.api.pmu.fr/rest/client/7/programme"
           f"/{date_str}/R{r_num}/C{c_num}/participants")
    try:
        resp = http_requests.get(url, timeout=8)
        resp.raise_for_status()
        participants = resp.json().get('participants', [])
    except Exception as e:
        return jsonify({"error": f"Erreur API PMU : {str(e)}"}), 502

    if not participants:
        return jsonify({"error": "Aucun participant trouvé"}), 404

    # Médiane rapport de référence
    rapports_course = [
        p['dernierRapportReference'].get('rapport')
        for p in participants
        if p.get('dernierRapportReference') and p.get('statut') != 'NON_PARTANT'
    ]
    rapports_course  = [r for r in rapports_course if r]
    mediane_rr       = float(np.median(rapports_course)) if rapports_course else _mediane_rapport_ref

    rows = []
    for p in participants:
        if p.get('statut') == 'NON_PARTANT' or p.get('incident') == 'NON_PARTANT':
            continue
        mus        = _parser_musique_api(p.get('musique', ''))
        gains      = p.get('gainsParticipant', {}) or {}
        rk         = p.get('reductionKilometrique', 0) or 0
        num_pmu    = p.get('numPmu')
        nb_courses = p.get('nombreCourses', 0) or 0
        driver_nom = (p.get('driver', {}).get('nom', '')
                      if isinstance(p.get('driver'), dict) else str(p.get('driver', '')))
        entr_nom   = (p.get('entraineur', {}).get('nom', '')
                      if isinstance(p.get('entraineur'), dict) else str(p.get('entraineur', '')))
        nb_victoires = p.get('nombreVictoires', 0) or 0
        nb_places    = p.get('nombrePlaces', 0) or 0
        gains_car    = gains.get('gainsCarriere', 0) or 0
        gains_ann    = gains.get('gainsAnneeEnCours', 0) or 0

        rapport_ref = None
        if p.get('dernierRapportReference'):
            rapport_ref = p['dernierRapportReference'].get('rapport')
        if rapport_ref is None:
            rapport_ref = mediane_rr

        cote_app = None
        if p.get('dernierRapportDirect'):
            cote_app = p['dernierRapportDirect'].get('rapport')
        if cote_app is None and p.get('dernierRapportReference'):
            cote_app = p['dernierRapportReference'].get('rapport')

        perf = perfs_map.get(num_pmu, _perf_vide())

        row = {
            'numero':            num_pmu,
            'nom':               p.get('nom', ''),
            # Conditions course
            'distance':          conditions['distance'],
            'montant_prix':      conditions['montant_prix'],
            'discipline':        conditions['discipline'],
            'corde':             conditions['corde'],
            'condition_sexe':    conditions['condition_sexe'],
            'nb_partants':       conditions['nb_partants'],
            # Cheval
            'age':               p.get('age', 0) or 0,
            'deferre':           _ferrage_map_pmu.get(p.get('deferre', 'FERRE'), 0),
            'oeilleres':         1 if p.get('oeilleres') else 0,
            'driver':            driver_nom,
            'entraineur':        entr_nom,
            'nb_courses':        nb_courses,
            'nb_victoires':      nb_victoires,
            'nb_places':         nb_places,
            'gains_carriere':    gains_car,
            'gains_annee':       gains_ann,
            'reduction_km_corr': rk if rk > 0 else 72600,
            'cheval_etranger':   int(rk == 0 and nb_courses > 5),
            'avis_entraineur':   _avis_map_pmu.get(p.get('avisEntraineur', 'NEUTRE'), 0),
            'rapport_ref':       float(rapport_ref),
            'log_rapport_ref':   float(np.log1p(rapport_ref)),
            '_cote_app':         cote_app,
        }
        row.update(mus)
        row.update(perf)
        rows.append(row)

    df_nc = pd.DataFrame(rows)

    # ── Features dérivées ─────────────────────────────────────
    df_nc['ratio_victoires']  = df_nc['nb_victoires'] / (df_nc['nb_courses'] + 1)
    df_nc['ratio_places']     = df_nc['nb_places']    / (df_nc['nb_courses'] + 1)
    df_nc['gains_par_course'] = df_nc['gains_carriere'] / (df_nc['nb_courses'] + 1)
    df_nc['ratio_gains_rec']  = df_nc['gains_annee'] / (df_nc['gains_carriere'] + 1)
    df_nc['log_distance']     = np.log1p(df_nc['distance'])
    df_nc['log_montant_prix'] = np.log1p(df_nc['montant_prix'])
    df_nc['rang_cote_course'] = df_nc['rapport_ref'].rank(ascending=True, method='min')
    nb_ch = len(df_nc)
    df_nc['rang_cote_norme']  = (df_nc['rang_cote_course'] - 1) / (nb_ch - 1 + 1e-8)
    df_nc['accord_mus_perf']  = (
        (df_nc['mus_moy_classement'] < 5) & (df_nc['perf_moy_classement'] < 5)
    ).astype(int)
    df_nc['tranche_distance'] = pd.cut(
        df_nc['distance'], bins=[0, 1600, 2100, 2700, 9999],
        labels=['court', 'moyen', 'long', 'tres_long']
    ).astype(str)

    _fallback = _prior_pmu * _k_bayes_pmu / (_k_bayes_pmu + 1)

    # Driver
    top_drivers = set(_le_driver.classes_)
    df_nc['driver_enc'] = df_nc['driver'].apply(lambda x: x if x in top_drivers else 'AUTRE')
    df_nc['driver_id']  = _le_driver.transform(df_nc['driver_enc'])
    d_cols = ['driver', 'driver_win_rate_bayes', 'driver_n']
    if 'driver_place_rate_bayes' in _driver_stats.columns:
        d_cols += ['driver_place_rate_bayes', 'driver_disq']
    df_nc = df_nc.merge(_driver_stats[d_cols], on='driver', how='left')
    df_nc['driver_win_rate_bayes']   = df_nc['driver_win_rate_bayes'].fillna(_fallback)
    df_nc['driver_n']                = df_nc['driver_n'].fillna(0)
    if 'driver_place_rate_bayes' in df_nc.columns:
        df_nc['driver_place_rate_bayes'] = df_nc['driver_place_rate_bayes'].fillna(_fallback)
        df_nc['driver_disq']             = df_nc['driver_disq'].fillna(0)

    # Entraîneur
    top_entrs = set(_le_entr.classes_)
    df_nc['entraineur_enc'] = df_nc['entraineur'].apply(lambda x: x if x in top_entrs else 'AUTRE')
    df_nc['entraineur_id']  = _le_entr.transform(df_nc['entraineur_enc'])
    if _entr_stats is not None:
        df_nc = df_nc.merge(
            _entr_stats[['entraineur', 'entr_win_rate_bayes', 'entr_n']],
            on='entraineur', how='left')
    if 'entr_win_rate_bayes' not in df_nc.columns:
        df_nc['entr_win_rate_bayes'] = _fallback
        df_nc['entr_n'] = 0
    df_nc['entr_win_rate_bayes'] = df_nc['entr_win_rate_bayes'].fillna(_fallback)
    df_nc['entr_n']              = df_nc['entr_n'].fillna(0)

    # Duo
    if _duo_stats is not None:
        df_nc = df_nc.merge(
            _duo_stats[['nom', 'driver', 'duo_win_rate_bayes', 'duo_n']],
            on=['nom', 'driver'], how='left')
    if 'duo_win_rate_bayes' not in df_nc.columns:
        df_nc['duo_win_rate_bayes'] = _fallback
        df_nc['duo_n'] = 0
    df_nc['duo_win_rate_bayes'] = df_nc['duo_win_rate_bayes'].fillna(_fallback)
    df_nc['duo_n']              = df_nc['duo_n'].fillna(0)

    # Spécialisation distance
    if _spec_dist is not None:
        df_nc = df_nc.merge(
            _spec_dist[['nom', 'tranche_distance', 'spec_dist_rate', 'spec_n']],
            on=['nom', 'tranche_distance'], how='left')
    if 'spec_dist_rate' not in df_nc.columns:
        df_nc['spec_dist_rate'] = _fallback
        df_nc['spec_n'] = 0
    df_nc['spec_dist_rate'] = df_nc['spec_dist_rate'].fillna(_fallback)
    df_nc['spec_n']         = df_nc['spec_n'].fillna(0)

    # Spécialisation discipline
    if _spec_disc is not None:
        df_nc = df_nc.merge(
            _spec_disc[['nom', 'discipline', 'spec_disc_rate']],
            on=['nom', 'discipline'], how='left')
    if 'spec_disc_rate' not in df_nc.columns:
        df_nc['spec_disc_rate'] = _fallback
    df_nc['spec_disc_rate'] = df_nc['spec_disc_rate'].fillna(_fallback)

    # ── Prédiction + note ────────────────────────────────────
    probas = _model_pmu.predict_proba(df_nc[_features_pmu])[:, 1]
    df_nc['proba_pmu'] = probas
    df_nc['note_pmu']  = _proba_to_note_api(pd.Series(probas))

    # ── Résultat JSON ────────────────────────────────────────
    result = []
    for _, row in df_nc.sort_values('note_pmu', ascending=False).iterrows():
        result.append({
            "numero":    int(row['numero']),
            "nom":       str(row['nom']),
            "note_pmu":  int(row['note_pmu']),
            "proba_pmu": round(float(row['proba_pmu']) * 100, 1),
            "driver":    str(row['driver']),
            "etranger":  bool(row['cheval_etranger']),
            "cote":      float(row['_cote_app']) if row['_cote_app'] is not None else None,
            "avis":      int(row['avis_entraineur']),
        })

    return jsonify({
        "date":     date_str,
        "reunion":  r_num,
        "course":   c_num,
        "chevaux":  result,
    })


# ============================================================
# DÉMARRAGE
# ============================================================
_charger_modele_pmu()
initialiser()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
