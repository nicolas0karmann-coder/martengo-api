# api.py — Backend Flask pour Martengo Prediction
# Déploiement : Railway / Render / Heroku

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from io import StringIO
import os
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
# DÉMARRAGE
# ============================================================
initialiser()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
