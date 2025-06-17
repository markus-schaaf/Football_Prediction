import pandas as pd
from football_prediction.models import Match

def load_match_data():
    # Rohdaten aus DB holen
    matches = Match.objects.all().values()
    df = pd.DataFrame(matches)

    # Fehlende Spalten ergänzen
    if df.empty:
        raise ValueError("No match data found in database.")

    # Durchschnittstore berechnen
    df['average_home_goals'] = df.groupby('home_team')['home_goals'].transform('mean')
    df['average_away_goals'] = df.groupby('away_team')['away_goals'].transform('mean')

    return df
