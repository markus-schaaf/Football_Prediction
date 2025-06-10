from django.db.models import OuterRef, Subquery
import pandas as pd
import numpy as np
import os
import django
import joblib
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, brier_score_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
import json
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "anwendungsprojekt.settings")

import django
django.setup()

from football_prediction.models import Match

from django.db.models import F

qs = Match.objects.all()
df = pd.DataFrame.from_records(qs.values(
    "match_id", "date", "matchday", "home_team", "away_team", "home_goals", "away_goals", "scorers", "season", "result",
    "elo_home", "elo_away"
))


# Durchschnittliche Tore berechnen
df["average_home_goals"] = df.groupby("home_team")["home_goals"].transform("mean")
df["average_away_goals"] = df.groupby("away_team")["away_goals"].transform("mean")

# Gewinnraten berechnen
df["home_win_rate"] = df.groupby("home_team")["result"].transform(lambda x: (x == "home_win").mean())
df["away_win_rate"] = df.groupby("away_team")["result"].transform(lambda x: (x == "away_win").mean())

# Zusätzliche Feature (falls nötig)
df["goal_avg_diff"] = df["average_home_goals"] - df["average_away_goals"]

# Datum konvertieren
df['date'] = pd.to_datetime(df['date'])

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

# Elo-Berechnungen
def expected_score(rating_a, rating_b):
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

def update_elo(rating_a, rating_b, result, k=20):
    expected_a = expected_score(rating_a, rating_b)
    new_rating_a = rating_a + k * (result - expected_a)
    new_rating_b = rating_b + k * ((1 - result) - (1 - expected_a))
    return new_rating_a, new_rating_b



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

    prev_home = df[(df['season'] == saison) & (df['home_team'] == heim) & (df['date'] < spiel_datum)]
    if len(prev_home) > 0:
        df.at[idx, 'average_home_goals'] = prev_home['home_goals'].mean()
        df.at[idx, 'home_win_rate'] = (prev_home['result'] == 'home_win').mean()
    else:
        saison_bisher = df[(df['season'] == saison) & (df['date'] < spiel_datum)]
        if len(saison_bisher) > 0:
            df.at[idx, 'average_home_goals'] = saison_bisher['home_goals'].mean()
            df.at[idx, 'home_win_rate'] = (saison_bisher['result'] == 'home_win').mean()
        else:
            df.at[idx, 'average_home_goals'] = 1.5
            df.at[idx, 'home_win_rate'] = 0.4

    prev_away = df[(df['season'] == saison) & (df['away_team'] == auswaerts) & (df['date'] < spiel_datum)]
    if len(prev_away) > 0:
        df.at[idx, 'average_away_goals'] = prev_away['away_goals'].mean()
        df.at[idx, 'away_win_rate'] = (prev_away['result'] == 'away_win').mean()
    else:
        saison_bisher = df[(df['season'] == saison) & (df['date'] < spiel_datum)]
        if len(saison_bisher) > 0:
            df.at[idx, 'average_away_goals'] = saison_bisher['away_goals'].mean()
            df.at[idx, 'away_win_rate'] = (saison_bisher['result'] == 'away_win').mean()
        else:
            df.at[idx, 'average_away_goals'] = 1.0
            df.at[idx, 'away_win_rate'] = 0.3

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

    elo_ratings[heim] = new_elo_home
    elo_ratings[auswaerts] = new_elo_away



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

# Kodierung der Teams
le_home = LabelEncoder()
le_away = LabelEncoder()
df['home_team_encoded'] = le_home.fit_transform(df['home_team'])
df['away_team_encoded'] = le_away.fit_transform(df['away_team'])

le_result = LabelEncoder()
y = le_result.fit_transform(df['result'])

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

X = X.astype(float)

model_dir = os.path.join("football_prediction", "model")
static_dir = os.path.join("football_prediction", "static", "model")
os.makedirs(model_dir, exist_ok=True)
os.makedirs(static_dir, exist_ok=True)

# X.columns speichern
with open(os.path.join(model_dir, "feature_columns.json"), "w") as f:
    json.dump(list(X.columns), f)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train, y_train)

calibrated_model = CalibratedClassifierCV(rf, cv=3, method='isotonic')
calibrated_model.fit(X_train, y_train)



y_pred = calibrated_model.predict(X_test)
print(f"Genauigkeit (kalibriertes Modell): {accuracy_score(y_test, y_pred):.2%}")

y_proba = calibrated_model.predict_proba(X_test)
y_test_binarized = label_binarize(y_test, classes=list(range(len(le_result.classes_))))
brier = np.mean(np.sum((y_proba - y_test_binarized) ** 2, axis=1))
print(f"Brier Score (multiclass): {brier:.4f}")

# Cross-Validation
scores = cross_val_score(rf, X, y, cv=5, scoring='accuracy')
print(f"Durchschnittliche Genauigkeit (Cross-Validation, 5-fach): {scores.mean():.2%}")
print(f"Genauigkeit pro Fold: {scores}")
print(f"Spannweite: {scores.min():.2%} – {scores.max():.2%}")
print(f"Standardabweichung: {scores.std():.2%}")

joblib.dump(calibrated_model, os.path.join(model_dir, "random_forest_model.joblib"))
joblib.dump(le_home, os.path.join(model_dir, "le_home.joblib"))
joblib.dump(le_away, os.path.join(model_dir, "le_away.joblib"))
joblib.dump(le_result, os.path.join(model_dir, "le_result.joblib"))

importance_dict = dict(zip(X.columns, rf.feature_importances_))
with open(os.path.join(static_dir, "rf_feature_importance.json"), "w") as f:
    json.dump(importance_dict, f)