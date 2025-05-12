import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb

# CSV laden
df = pd.read_csv('C:\\Dev\\Anwendungsprojekt\\football_prediction\\data\\bundesliga_gesamt_2020_2024.csv')
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

# Tabellenplatz-Features initialisieren
df['home_position'] = 0
df['away_position'] = 0
df['average_home_goals'] = 0.0
df['average_away_goals'] = 0.0
df['home_win_rate'] = 0.0
df['away_win_rate'] = 0.0

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

# Features und Zielvariable definieren
X = df[[
    'home_team_encoded', 'away_team_encoded',
    'home_position', 'away_position',
    'average_home_goals', 'average_away_goals',
    'home_win_rate', 'away_win_rate',
    'home_form_points', 'away_form_points',
    'home_form_goaldiff', 'away_form_goaldiff',
    'home_form_curve', 'away_form_curve'
]]
y = df['result']

# Zielvariable kodieren
le_result = LabelEncoder()
y_encoded = le_result.fit_transform(y)

# Daten aufteilen
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Modell trainieren
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
model.fit(X_train, y_train)

# Vorhersage und Bewertung
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Genauigkeit des verbesserten Modells: {accuracy:.2%}")

from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y_encoded, cv=5, scoring='accuracy')
print(f"Durchschnittliche Genauigkeit (Cross-Validation, 5-fach): {scores.mean():.2%}")
print(f"Genauigkeit pro Fold: {scores}")
print(f"Spannweite: {scores.min():.2%} – {scores.max():.2%}")
print(f"Standardabweichung: {scores.std():.2%}")
