from django.shortcuts import render, redirect
from django.urls import reverse
from football_prediction.models import Match
import joblib
import numpy as np
import pandas as pd
import json
from django.db.models import F

def football_prediction(request):
    return render(request, 'homepage.html')

def model_overview(request):
    with open("football_prediction/model/feature_importance.json", "r") as f:
        feature_data = json.load(f)
    return render(request, "overview.html", {
        "feature_names": feature_data["features"],
        "importances": feature_data["importances"]
    })

def get_elo_rating(team_name):
    last_match = Match.objects.filter(home_team=team_name).order_by('-date').first()
    if last_match:
        return last_match.elo_home
    last_match = Match.objects.filter(away_team=team_name).order_by('-date').first()
    if last_match:
        return last_match.elo_away
    return 1500  # Fallback

def get_team_form(team_name):
    matches = Match.objects.filter(home_team=team_name) | Match.objects.filter(away_team=team_name)
    matches = matches.order_by('-date')[:5]

    points = 0
    for m in matches:
        if m.home_team == team_name:
            if m.home_goals > m.away_goals:
                points += 3
            elif m.home_goals == m.away_goals:
                points += 1
        elif m.away_team == team_name:
            if m.away_goals > m.home_goals:
                points += 3
            elif m.away_goals == m.home_goals:
                points += 1
    return points

def get_goal_avg(team_name, home=True):
    if home:
        matches = Match.objects.filter(home_team=team_name)
        goals = matches.values_list('home_goals', flat=True)
    else:
        matches = Match.objects.filter(away_team=team_name)
        goals = matches.values_list('away_goals', flat=True)
    return np.mean(goals) if goals else 1.0

def get_win_rate(team_name, home=True):
    if home:
        matches = Match.objects.filter(home_team=team_name)
        wins = matches.filter(home_goals__gt=F('away_goals')).count()
    else:
        matches = Match.objects.filter(away_team=team_name)
        wins = matches.filter(away_goals__gt=F('home_goals')).count()
    total = matches.count()
    return wins / total if total > 0 else 0.0

def team_selection_view(request):
    if Match.objects.exists():
        home_teams = Match.objects.values_list('home_team', flat=True)
        away_teams = Match.objects.values_list('away_team', flat=True)
        all_teams = list(set(home_teams) | set(away_teams))
        all_teams.sort()
    else:
        all_teams = []

    prediction_result = None

    if request.method == "POST":
        team1 = request.POST.get('team1')
        team2 = request.POST.get('team2')

        if team1 and team2:
            with open('football_prediction/model/feature_columns.json', 'r') as f:
                feature_columns = json.load(f)

            elo_team1 = get_elo_rating(team1)
            elo_team2 = get_elo_rating(team2)
            elo_diff = elo_team1 - elo_team2

            form_team1 = get_team_form(team1)
            form_team2 = get_team_form(team2)
            form_diff = form_team1 - form_team2

            avg_goals_home = get_goal_avg(team1, home=True)
            avg_goals_away = get_goal_avg(team2, home=False)
            goal_avg_diff = avg_goals_home - avg_goals_away

            win_rate_home = get_win_rate(team1, home=True)
            win_rate_away = get_win_rate(team2, home=False)

            input_data = pd.DataFrame([{
                'elo_diff': elo_diff,
                'form_diff': form_diff,
                'goal_avg_diff': goal_avg_diff,
                'form_curve_diff': form_diff,  # Platzhalter
                'home_position': 5,  # Platzhalter
                'away_position': 8,  # Platzhalter
                'average_home_goals': avg_goals_home,
                'average_away_goals': avg_goals_away,
                'home_win_rate': win_rate_home,
                'away_win_rate': win_rate_away
            }])

            for col in feature_columns:
                if col not in input_data.columns:
                    input_data[col] = 0

            input_data = input_data[feature_columns]

            rf_model = joblib.load('football_prediction/model/random_forest_model.joblib')
            xgb_model = joblib.load('football_prediction/model/xgb_model.joblib')

            rf_probs = rf_model.predict_proba(input_data)[0]
            xgb_probs = xgb_model.predict_proba(input_data)[0]

            prediction_result = {
                'team1': team1,
                'team2': team2,
                'rf': {
                    'home_win': f"{rf_probs[0]*100:.2f}%",
                    'draw': f"{rf_probs[1]*100:.2f}%",
                    'away_win': f"{rf_probs[2]*100:.2f}%"
                },
                'xgb': {
                    'home_win': f"{xgb_probs[0]*100:.2f}%",
                    'draw': f"{xgb_probs[1]*100:.2f}%",
                    'away_win': f"{xgb_probs[2]*100:.2f}%"
                }
            }

    return render(request, 'homepage.html', {
        'teams': all_teams,
        'prediction_result': prediction_result
    })
