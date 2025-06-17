import os
import sys

# Projektverzeichnis zum Python-Pfad hinzufügen
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

# Django-Settings korrekt setzen
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "anwendungsprojekt.settings")

import django
django.setup()

# Danach erst Django-Importe
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
import joblib
from evaluation import evaluate_and_save


from football_prediction.models_views import (
    MatchWithAvgHomeGoals,
    MatchWithAvgAwayGoals,
    MatchWithHomeWinRate,
    MatchWithAwayWinRate
)
from football_prediction.models import Match



# Alle Matches aus der Datenbank abrufen
qs = Match.objects.all().values()
df = pd.DataFrame.from_records(qs)

# Optional: Datum konvertieren
df['date'] = pd.to_datetime(df['date'])

# Match-ID als Index setzen für Joins
df.set_index('match_id', inplace=True)

# View-Daten aus der Datenbank laden
avg_home = pd.DataFrame.from_records(MatchWithAvgHomeGoals.objects.all().values()).set_index('match_id')
avg_away = pd.DataFrame.from_records(MatchWithAvgAwayGoals.objects.all().values()).set_index('match_id')
win_home = pd.DataFrame.from_records(MatchWithHomeWinRate.objects.all().values()).set_index('match_id')
win_away = pd.DataFrame.from_records(MatchWithAwayWinRate.objects.all().values()).set_index('match_id')

# Nur die berechneten Spalten anhängen
df = df.join(avg_home[['average_home_goals']])
df = df.join(avg_away[['average_away_goals']])
df = df.join(win_home[['home_win_rate']])
df = df.join(win_away[['away_win_rate']])

# Index zurücksetzen auf normale DataFrame-Struktur
df = df.reset_index()
# Konvertiere alle relevanten View-Spalten zu float
df['average_home_goals'] = pd.to_numeric(df['average_home_goals'], errors='coerce')
df['average_away_goals'] = pd.to_numeric(df['average_away_goals'], errors='coerce')
df['home_win_rate'] = pd.to_numeric(df['home_win_rate'], errors='coerce')
df['away_win_rate'] = pd.to_numeric(df['away_win_rate'], errors='coerce')

# Optionales kombiniertes Feature (falls verwendet)
df['goal_avg_diff'] = df['average_home_goals'] - df['average_away_goals']
df['goal_avg_diff'] = pd.to_numeric(df['goal_avg_diff'], errors='coerce')

# Fehlende Werte (NaNs) auffüllen, damit XGBoost nicht crasht
df[['average_home_goals', 'average_away_goals']] = df[['average_home_goals', 'average_away_goals']].fillna(1.5)
df[['home_win_rate', 'away_win_rate']] = df[['home_win_rate', 'away_win_rate']].fillna(0.4)
df['goal_avg_diff'] = df['goal_avg_diff'].fillna(0.0)

print(df[['average_home_goals', 'average_away_goals', 'goal_avg_diff']].dtypes)


print(df.head())
print(f"⚽ Anzahl geladener Spiele: {len(df)}")


# Zielvariable erstellen
def get_result(row):
    if row['home_goals'] > row['away_goals']:
        return 'home_win'
    elif row['home_goals'] < row['away_goals']:
        return 'away_win'
    else:
        return 'draw'

df['result'] = df.apply(get_result, axis=1)

# Saison-Spalte erstellen
def get_season(row):
    jahr = row['date'].year
    monat = row['date'].month
    if monat >= 7:
        return f"{jahr}/{jahr+1}"
    else:
        return f"{jahr-1}/{jahr}"

df['season'] = df.apply(get_season, axis=1)

# Spiele chronologisch sortieren
df = df.sort_values(by='date').reset_index(drop=True)
def expected_score(rating_a, rating_b):
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

def update_elo(rating_a, rating_b, result, k=20):
    expected_a = expected_score(rating_a, rating_b)
    new_rating_a = rating_a + k * (result - expected_a)
    new_rating_b = rating_b + k * ((1 - result) - (1 - expected_a))
    return new_rating_a, new_rating_b

df['elo_home'] = 1500.0
df['elo_away'] = 1500.0
df['elo_diff'] = 0.0

df['home_position'] = 0
df['away_position'] = 0
# df['average_home_goals'] = 0.0
# df['average_away_goals'] = 0.0
# df['home_win_rate'] = 0.0
# df['away_win_rate'] = 0.0

elo_ratings = {}


for idx, match in df.iterrows():
    saison = match['season']
    spiel_datum = pd.to_datetime(match['date'])
    heim = match['home_team']
    auswaerts = match['away_team']
    bisher = df[(df['season'] == saison) & (df['date'] < spiel_datum)]

    if len(bisher) > 0:
        def punkte(hg, ag):
            if hg > ag:
                return (3, 0)
            elif hg < ag:
                return (0, 3)
            else:
                return (1, 1)

        bisher = bisher.copy()
        bisher.loc[:, 'home_pts'], bisher.loc[:, 'away_pts'] = zip(*bisher.apply(
            lambda r: punkte(r['home_goals'], r['away_goals']), axis=1))

        home_stats = bisher.groupby('home_team').agg({
            'home_pts': 'sum', 'home_goals': 'sum', 'away_goals': 'sum'
        }).rename(columns={
            'home_pts': 'points', 'home_goals': 'goals_for', 'away_goals': 'goals_against'
        })

        away_stats = bisher.groupby('away_team').agg({
            'away_pts': 'sum', 'away_goals': 'sum', 'home_goals': 'sum'
        }).rename(columns={
            'away_pts': 'points', 'away_goals': 'goals_for', 'home_goals': 'goals_against'
        })

        tabelle = home_stats.add(away_stats, fill_value=0)
        tabelle['goal_diff'] = tabelle['goals_for'] - tabelle['goals_against']
        tabelle = tabelle.sort_values(by=['points', 'goal_diff'], ascending=False)
        tabelle['position'] = range(1, len(tabelle) + 1)

        df.at[idx, 'home_position'] = tabelle.loc[heim]['position'] if heim in tabelle.index else 0
        df.at[idx, 'away_position'] = tabelle.loc[auswaerts]['position'] if auswaerts in tabelle.index else 0
    else:
        df.at[idx, 'home_position'] = 0
        df.at[idx, 'away_position'] = 0

    # prev_home = df[(df['season'] == saison) & (df['home_team'] == heim) & (df['date'] < spiel_datum)]
    # if len(prev_home) > 0:
    #     df.at[idx, 'average_home_goals'] = prev_home['home_goals'].mean()
    #     df.at[idx, 'home_win_rate'] = (prev_home['result'] == 'home_win').mean()
    # else:
    #     saison_bisher = df[(df['season'] == saison) & (df['date'] < spiel_datum)]
    #     if len(saison_bisher) > 0:
    #         df.at[idx, 'average_home_goals'] = saison_bisher['home_goals'].mean()
    #         df.at[idx, 'home_win_rate'] = (saison_bisher['result'] == 'home_win').mean()
    #     else:
    #         df.at[idx, 'average_home_goals'] = 1.5
    #         df.at[idx, 'home_win_rate'] = 0.4

    # prev_away = df[(df['season'] == saison) & (df['away_team'] == auswaerts) & (df['date'] < spiel_datum)]
    # if len(prev_away) > 0:
    #     df.at[idx, 'average_away_goals'] = prev_away['away_goals'].mean()
    #     df.at[idx, 'away_win_rate'] = (prev_away['result'] == 'away_win').mean()
    # else:
    #     saison_bisher = df[(df['season'] == saison) & (df['date'] < spiel_datum)]
    #     if len(saison_bisher) > 0:
    #         df.at[idx, 'average_away_goals'] = saison_bisher['away_goals'].mean()
    #         df.at[idx, 'away_win_rate'] = (saison_bisher['result'] == 'away_win').mean()
    #     else:
    #         df.at[idx, 'average_away_goals'] = 1.0
    #         df.at[idx, 'away_win_rate'] = 0.3

    elo_home = elo_ratings.get(heim, 1500)
    elo_away = elo_ratings.get(auswaerts, 1500)

    if match['home_goals'] > match['away_goals']:
        result = 1
    elif match['home_goals'] == match['away_goals']:
        result = 0.5
    else:
        result = 0

    new_elo_home, new_elo_away = update_elo(elo_home, elo_away, result)

    df.at[idx, 'elo_home'] = elo_home
    df.at[idx, 'elo_away'] = elo_away
    df.at[idx, 'elo_diff'] = elo_home - elo_away

    Match.objects.filter(match_id=match['match_id']).update(
        elo_home=elo_home,
        elo_away=elo_away
    )

    elo_ratings[heim] = new_elo_home
    elo_ratings[auswaerts] = new_elo_away

def expected_score(rating_a, rating_b):
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

def update_elo(rating_a, rating_b, result, k=20):
    expected_a = expected_score(rating_a, rating_b)
    new_rating_a = rating_a + k * (result - expected_a)
    new_rating_b = rating_b + k * ((1 - result) - (1 - expected_a))
    return new_rating_a, new_rating_b


# Zusätzliche Form-Features

df['home_points'] = df['result'].map({'home_win': 3, 'draw': 1, 'away_win': 0})
df['away_points'] = df['result'].map({'away_win': 3, 'draw': 1, 'home_win': 0})

df['home_form_points'] = df.groupby('home_team')['home_points'].transform(lambda x: x.shift().rolling(3, min_periods=1).mean())
df['away_form_points'] = df.groupby('away_team')['away_points'].transform(lambda x: x.shift().rolling(3, min_periods=1).mean())

df['home_goal_diff'] = df['home_goals'] - df['away_goals']
df['away_goal_diff'] = df['away_goals'] - df['home_goals']

df['home_form_goaldiff'] = df.groupby('home_team')['home_goal_diff'].transform(lambda x: x.shift().rolling(3, min_periods=1).mean())
df['away_form_goaldiff'] = df.groupby('away_team')['away_goal_diff'].transform(lambda x: x.shift().rolling(3, min_periods=1).mean())

df['home_form_curve'] = df.groupby('home_team')['home_points'].transform(lambda x: x.shift().rolling(5, min_periods=1).sum())
df['away_form_curve'] = df.groupby('away_team')['away_points'].transform(lambda x: x.shift().rolling(5, min_periods=1).sum())

df['form_diff'] = df['home_form_goaldiff'] - df['away_form_goaldiff']

df['goal_avg_diff'] = df['average_home_goals'] - df['average_away_goals']

df['form_curve_diff'] = df['home_form_curve'] - df['away_form_curve']


form_features = [
    'home_form_points', 'away_form_points',
    'home_form_goaldiff', 'away_form_goaldiff',
    'home_form_curve', 'away_form_curve'
]

# Sichere Extraktion der Spieltagsnummer als Integer-Spalte (nullable)
df['matchday_number'] = df['matchday'].str.extract(r'(\d+)')
df['matchday_number'] = pd.to_numeric(df['matchday_number'], errors='coerce').astype('Int64')

# Borussia Dortmund Spiele ab Spieltag 5
df_dortmund = df[(df['home_team'] == 'Borussia Dortmund') | (df['away_team'] == 'Borussia Dortmund')]

# print(df_dortmund[df_dortmund['matchday_number'] >= 5][[
#     'matchday', 'date', 'home_team', 'away_team',
#     'home_goals', 'away_goals',
#     'home_form_curve', 'away_form_curve'
# ]].head(10))

# Kodierung der Teams
le_home = LabelEncoder()
le_away = LabelEncoder()
df['home_team_encoded'] = le_home.fit_transform(df['home_team'])
df['away_team_encoded'] = le_away.fit_transform(df['away_team'])

X = df[[ 
    'home_team_encoded', 'away_team_encoded',
    'home_position', 'away_position',
    'average_home_goals', 'average_away_goals',
    'home_win_rate', 'away_win_rate',
    'home_form_points', 'away_form_points',
    'home_form_goaldiff', 'away_form_goaldiff',
    'form_diff', 'goal_avg_diff',  
    'form_curve_diff',
    'elo_home', 'elo_away', 'elo_diff',
            
]]


y = df['result']

le_result = LabelEncoder()
y_encoded = le_result.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# # Modell trainieren
# model = xgb.XGBClassifier(
#     eval_metric='mlogloss',
#     learning_rate=0.05,
#     max_depth=4,
#     n_estimators=100
# )

# model.fit(X_train, y_train)

base_model = xgb.XGBClassifier(
    eval_metric='mlogloss',
    learning_rate=0.05,
    max_depth=4,
    n_estimators=100
)
base_model.fit(X_train, y_train)

calibrated_model = CalibratedClassifierCV(base_model, cv="prefit", method='isotonic')
calibrated_model.fit(X_train, y_train)
# Vorhersage und Bewertung
y_pred = calibrated_model.predict(X_test)

class_names = le_result.classes_
#evaluate_model(y_test, y_pred, class_names=class_names)

accuracy = accuracy_score(y_test, y_pred)
print(f"Genauigkeit (kalibriertes Modell): {accuracy:.2%}")

from evaluation import evaluate_and_save

eval_path = "football_prediction/static/model/xgb_eval.json"
evaluate_and_save(y_test, y_pred, "XGBoost", class_names, eval_path)

# Wahrscheinlichkeiten & Brier-Score
y_proba = calibrated_model.predict_proba(X_test)
import numpy as np
from sklearn.preprocessing import label_binarize

n_classes = len(le_result.classes_)
y_test_binarized = label_binarize(y_test, classes=list(range(n_classes)))

brier = np.mean(np.sum((y_proba - y_test_binarized) ** 2, axis=1))
print(f"Brier Score (multiclass): {brier:.4f}")

print(f"Genauigkeit des verbesserten Modells: {accuracy:.2%}")

from sklearn.model_selection import cross_val_score

scores = cross_val_score(base_model, X, y_encoded, cv=5, scoring='accuracy')
print(f"Durchschnittliche Genauigkeit (Cross-Validation, 5-fach): {scores.mean():.2%}")
print(f"Genauigkeit pro Fold: {scores}")
print(f"Spannweite: {scores.min():.2%} – {scores.max():.2%}")
print(f"Standardabweichung: {scores.std():.2%}")


# from sklearn.model_selection import GridSearchCV
# import xgboost as xgb

# param_grid = {
#     'max_depth': [3, 4],
#     'learning_rate': [0.05, 0.1],
#     'n_estimators': [100, 200]
# }

# grid_search = GridSearchCV(
#     estimator=xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
#     param_grid=param_grid,
#     scoring='accuracy',
#     cv=3,  # Cross-Validation auf 3 Teile reduziert für Geschwindigkeit
#     verbose=1,
#     n_jobs=-1  # nutzt alle verfügbaren CPU-Kerne
# )

# grid_search.fit(X, y_encoded)

# print("\nBeste Parameterkombination:")
# print(grid_search.best_params_)
# print(f"Beste Genauigkeit: {grid_search.best_score_:.2%}")


# import matplotlib.pyplot as plt
# xgb.plot_importance(base_model, max_num_features=10)
# plt.tight_layout()
# plt.show()

import json
import os

# Feature Importance berechnen und sortieren
booster = base_model.get_booster()
importance_dict = booster.get_score(importance_type='weight')
sorted_items = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)

# Export für Plotly
feature_names = [k for k, v in sorted_items]
importances = [v for k, v in sorted_items]

feature_json = {
    'features': feature_names,
    'importances': importances
}



model_dir = os.path.join("football_prediction", "model")
static_dir = os.path.join("football_prediction", "static", "model")
os.makedirs(model_dir, exist_ok=True)
os.makedirs(static_dir, exist_ok=True)

joblib.dump(calibrated_model, os.path.join(model_dir, "xgb_model.joblib"))
joblib.dump(le_home, os.path.join(model_dir, "le_home.joblib"))
joblib.dump(le_away, os.path.join(model_dir, "le_away.joblib"))
joblib.dump(le_result, os.path.join(model_dir, "le_result.joblib"))

base_model = calibrated_model.estimator
importance_dict = {
    feature: float(importance)
    for feature, importance in zip(X.columns, calibrated_model.estimator.feature_importances_)
}

with open(os.path.join(static_dir, "xgb_feature_importance.json"), "w") as f:
    json.dump(importance_dict, f)

