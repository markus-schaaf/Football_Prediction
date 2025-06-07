import pandas as pd
import os

# Pfade definieren
data_input = 'football_prediction/data'
output_path = 'football_prediction/data_lake/processed/matches_cleaned.parquet'

# Alle CSV-Dateien im data-Ordner sammeln
csv_files = [f for f in os.listdir(data_input) if f.endswith('.csv')]

all_dfs = []

for file in csv_files:
    file_path = os.path.join(data_input, file)
    df = pd.read_csv(file_path)

    # Datum formatieren
    df['date'] = pd.to_datetime(df['date'])

    # Saison-Spalte erzeugen
    df['season'] = df['date'].apply(
        lambda d: f"{d.year}/{d.year+1}" if d.month >= 7 else f"{d.year-1}/{d.year}"
    )

    # Zielvariable erstellen (home_win, away_win, draw)
    def get_result(row):
        if row['home_goals'] > row['away_goals']:
            return 'home_win'
        elif row['home_goals'] < row['away_goals']:
            return 'away_win'
        else:
            return 'draw'

    df['result'] = df.apply(get_result, axis=1)

    all_dfs.append(df)

# Alle Daten zusammenfügen
final_df = pd.concat(all_dfs, ignore_index=True)

# Speichern als Parquet
final_df.to_parquet(output_path, index=False)
print(f"✅ Daten bereinigt und gespeichert unter: {output_path}")
