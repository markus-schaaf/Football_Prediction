import requests
import pandas as pd
import os

# Saisonjahre (beginnend mit 2020/2021 bis 2023/2024)
seasons = [2020, 2021, 2022, 2023]  # 2020 bedeutet 2020/2021

# Liga-Kürzel für 1. Bundesliga
league_shortcut = "bl1"

# Speicherort
output_folder = 'C:\\Dev\\Anwendungsprojekt\\football_prediction\\data'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def get_match_data(league_shortcut, season):
    url = f"https://www.openligadb.de/api/getmatchdata/{league_shortcut}/{season}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Fehler beim Abrufen der Daten für Saison {season}: {response.status_code}")
        return []

def process_matches(matches):
    data = []
    for match in matches:
        # Skippe, falls Match leer oder kein MatchID vorhanden ist
        if not match or 'matchID' not in match:
            print("Leeres oder ungültiges Match gefunden. Überspringe...")
            continue

        match_id = match['matchID']
        match_date = match['matchDateTime']
        matchday = match['group']['groupName']

        home_team = match['team1']['teamName']
        away_team = match['team2']['teamName']

        home_goals = match['matchResults'][0]['pointsTeam1'] if match['matchResults'] else None
        away_goals = match['matchResults'][0]['pointsTeam2'] if match['matchResults'] else None

        # Torschützen sammeln
        scorers = []
        if 'goalGetter' in match and match['goalGetter']:
            for goal in match['goalGetter']:
                scorer = f"{goal['goalGetterName']} ({goal['matchMinute']}\')"
                scorers.append(scorer)


        data.append({
            'match_id': match_id,
            'date': match_date,
            'matchday': matchday,
            'home_team': home_team,
            'away_team': away_team,
            'home_goals': home_goals,
            'away_goals': away_goals,
            'scorers': scorers
        })
    return data

def load_and_save_data(seasons):
    all_data = []
    for season in seasons:
        print(f"Lade Daten für Saison {season}/{season+1}...")
        matches = get_match_data(league_shortcut, season)
        season_data = process_matches(matches)
        all_data.extend(season_data)

        # Zwischenspeichern pro Saison
        df = pd.DataFrame(season_data)
        filename = os.path.join(output_folder, f"bundesliga_stats_{season}_{season+1}.csv")
        df.to_csv(filename, index=False)
        print(f"Saison {season}/{season+1} gespeichert: {filename}")

    # Gesamtdaten auch speichern
    all_df = pd.DataFrame(all_data)
    all_filename = os.path.join(output_folder, "bundesliga_gesamt_2020_2024.csv")
    all_df.to_csv(all_filename, index=False)
    print(f"Alle Saisons zusammen gespeichert: {all_filename}")

if __name__ == "__main__":
    load_and_save_data(seasons)
