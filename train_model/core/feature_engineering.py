import pandas as pd

def calculate_elo_diff(df):
    df["elo_diff"] = df["elo_home"] - df["elo_away"]
    return df

def add_positions(df):
    team_positions = {}
    for season in df["season"].unique():
        df_season = df[df["season"] == season]
        team_points = {}

        for team in pd.concat([df_season["home_team"], df_season["away_team"]]).unique():
            points = 0
            for _, row in df_season.iterrows():
                if row["home_team"] == team:
                    if row["result"] == "home_win":
                        points += 3
                    elif row["result"] == "draw":
                        points += 1
                if row["away_team"] == team:
                    if row["result"] == "away_win":
                        points += 3
                    elif row["result"] == "draw":
                        points += 1
            team_points[team] = points

        sorted_teams = sorted(team_points.items(), key=lambda x: -x[1])
        team_positions.update({team: i + 1 for i, (team, _) in enumerate(sorted_teams)})

    df["home_position"] = df["home_team"].map(team_positions)
    df["away_position"] = df["away_team"].map(team_positions)
    return df

def add_win_rates(df):
    win_counts = {}
    total_counts = {}

    for team in pd.concat([df["home_team"], df["away_team"]]).unique():
        win_counts[team] = 0
        total_counts[team] = 0

    for _, row in df.iterrows():
        total_counts[row["home_team"]] += 1
        total_counts[row["away_team"]] += 1

        if row["result"] == "home_win":
            win_counts[row["home_team"]] += 1
        elif row["result"] == "away_win":
            win_counts[row["away_team"]] += 1

    home_win_rate = {team: win_counts[team] / total_counts[team] if total_counts[team] else 0 for team in win_counts}
    df["home_win_rate"] = df["home_team"].map(home_win_rate)
    df["away_win_rate"] = df["away_team"].map(home_win_rate)
    return df

def add_form_stats(df):
    df["home_form_points"] = 0
    df["away_form_points"] = 0
    df["home_form_goaldiff"] = 0
    df["away_form_goaldiff"] = 0
    # Optional: historisch letzte 3–5 Spiele analysieren – kann nachgerüstet werden
    return df

def add_form_diffs(df):
    df["form_diff"] = df["home_form_points"] - df["away_form_points"]
    df["form_curve_diff"] = df["home_form_goaldiff"] - df["away_form_goaldiff"]
    return df

def add_goal_avg_diff(df):
    if "average_home_goals" not in df.columns or "average_away_goals" not in df.columns:
        print("Warnung: 'average_home_goals' oder 'average_away_goals' fehlen – 'goal_avg_diff' kann nicht berechnet werden.")
        df["goal_avg_diff"] = 0
    else:
        df["goal_avg_diff"] = df["average_home_goals"] - df["average_away_goals"]
    return df


def prepare_features(df):
    df = calculate_elo_diff(df)
    df = add_positions(df)
    df = add_win_rates(df)
    df = add_form_stats(df)
    df = add_form_diffs(df)
    df = add_goal_avg_diff(df)  # <–– Hier ergänzt
    return df
