import pandas as pd

# CSV laden
df = pd.read_csv('C:\\Dev\\Anwendungsprojekt\\football_prediction\\data\\bundesliga_gesamt_2020_2024.csv')

#print(df.head())
#Liest dir richtige Datei aus
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

#print(df[['home_team', 'away_team', 'home_goals', 'away_goals', 'result']].head())
#Test, ob sich die Tabelle richtig rechnet


# Saison-Spalte erstellen
def get_season(row):
    jahr = row['date'].year
    monat = row['date'].month
    if monat >= 7:
        return f"{jahr}/{jahr+1}"
    else:
        return f"{jahr-1}/{jahr}"


df['season'] = df.apply(get_season, axis=1)

#print(df[['date', 'season']].head(20))
#Hier wurde durch das Datum 2020 nicht berücksichtigt

# Spiele chronologisch sortieren
df = df.sort_values(by='date').reset_index(drop=True)

# Spalten für Tabellenplatz erstellen
# Spalten vorbereiten
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

    # Alle bisherigen Spiele der Saison (vor dem aktuellen Spiel)
    bisher = df[(df['season'] == saison) & (df['date'] < spiel_datum)]

    # --- Tabellenplatz berechnen ---
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
            'home_pts': 'sum',
            'home_goals': 'sum',
            'away_goals': 'sum'
        }).rename(columns={
            'home_pts': 'points',
            'home_goals': 'goals_for',
            'away_goals': 'goals_against'
        })

        away_stats = bisher.groupby('away_team').agg({
            'away_pts': 'sum',
            'away_goals': 'sum',
            'home_goals': 'sum'
        }).rename(columns={
            'away_pts': 'points',
            'away_goals': 'goals_for',
            'home_goals': 'goals_against'
        })

        tabelle = home_stats.add(away_stats, fill_value=0)
        tabelle['goal_diff'] = tabelle['goals_for'] - tabelle['goals_against']
        tabelle = tabelle.sort_values(by=['points', 'goal_diff'], ascending=False)
        tabelle['position'] = range(1, len(tabelle) + 1)

        # Tabellenplatz
        df.at[idx, 'home_position'] = tabelle.loc[heim]['position'] if heim in tabelle.index else 0
        df.at[idx, 'away_position'] = tabelle.loc[auswaerts]['position'] if auswaerts in tabelle.index else 0
    else:
        df.at[idx, 'home_position'] = 0
        df.at[idx, 'away_position'] = 0

    # --- Durchschnittstore und Siegquote aus den letzten 5 Spielen ---

    # Alle bisherigen Heimspiele
    prev_home = df[(df['season'] == saison) &
                (df['home_team'] == heim) &
                (df['date'] < spiel_datum)]


    if len(prev_home) > 0:
        df.at[idx, 'average_home_goals'] = prev_home['home_goals'].mean()
        df.at[idx, 'home_win_rate'] = (prev_home['result'] == 'home_win').mean()
    else:
        # Wenn keine vorherigen Heimspiele → Durchschnitt aller bisherigen Spiele der Saison nehmen
        saison_bisher = df[(df['season'] == saison) & (df['date'] < spiel_datum)]
        if len(saison_bisher) > 0:
            df.at[idx, 'average_home_goals'] = saison_bisher['home_goals'].mean()
            df.at[idx, 'home_win_rate'] = (saison_bisher['result'] == 'home_win').mean()
        else:
            df.at[idx, 'average_home_goals'] = 1.5  # typischer Bundesliga-Durchschnitt
            df.at[idx, 'home_win_rate'] = 0.4       # typischer Heimvorteil



    # Alle bisherigen Auswärtsspiele
    prev_away = df[(df['season'] == saison) &
                (df['away_team'] == auswaerts) &
                (df['date'] < spiel_datum)]

    if len(prev_away) > 0:
        df.at[idx, 'average_away_goals'] = prev_away['away_goals'].mean()
        df.at[idx, 'away_win_rate'] = (prev_away['result'] == 'away_win').mean()
    else:
        saison_bisher = df[(df['season'] == saison) & (df['date'] < spiel_datum)]
        if len(saison_bisher) > 0:
            df.at[idx, 'average_away_goals'] = saison_bisher['away_goals'].mean()
            df.at[idx, 'away_win_rate'] = (saison_bisher['result'] == 'away_win').mean()
        else:
            df.at[idx, 'average_away_goals'] = 1.0  # typischer Bundesliga-Auswärtsschnitt
            df.at[idx, 'away_win_rate'] = 0.3       # typischer Auswärtssieganteil


#print(df[['date', 'home_team', 'away_team', 'home_position', 'away_position']].head(50))
print(df[['home_team', 'away_team', 'home_position', 'away_position',
          'average_home_goals', 'average_away_goals',
          'home_win_rate', 'away_win_rate']].head(20))


# Teams numerisch kodieren
from sklearn.preprocessing import LabelEncoder

le_home = LabelEncoder()
le_away = LabelEncoder()

df['home_team_encoded'] = le_home.fit_transform(df['home_team'])
df['away_team_encoded'] = le_away.fit_transform(df['away_team'])

print(df[['home_team', 'away_team',
          'home_team_encoded', 'away_team_encoded',
          'home_position', 'away_position',
          'average_home_goals', 'average_away_goals',
          'home_win_rate', 'away_win_rate']].head(10))


# Features mit dynamischer Tabellenposition
X = df[['home_team_encoded', 'away_team_encoded',
        'home_position', 'away_position',
        'average_home_goals', 'average_away_goals',
        'home_win_rate', 'away_win_rate']]
y = df['result']

# Ergebnis (Zielvariable) numerisch kodieren
le_result = LabelEncoder()
y_encoded = le_result.fit_transform(y)

# Trainings- und Testdaten aufteilen
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

print("Trainingsgröße:", len(X_train))
print("Testgröße:", len(X_test))

# Modell erstellen und trainieren
import xgboost as xgb

model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
model.fit(X_train, y_train)

print("Modell wurde erfolgreich trainiert!")

from sklearn.metrics import accuracy_score

# Vorhersagen machen
y_pred = model.predict(X_test)

# Genauigkeit berechnen
accuracy = accuracy_score(y_test, y_pred)

print(f"Genauigkeit des XGBoost-Modells: {accuracy:.2%}")
