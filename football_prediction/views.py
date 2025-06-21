from django.shortcuts import render, redirect
from django.urls import reverse
from football_prediction.models import Match, MatchPrediction
import joblib
import numpy as np
import pandas as pd
import json
from django.db.models import F
from collections import Counter
from collections import defaultdict
from django.core.serializers.json import DjangoJSONEncoder
import os


# allgemeine view für dashboard
def dashboard_view(request):
    # Aktuelle Saison bestimmen
    latest_match = Match.objects.order_by('-season').first()
    season = latest_match.season if latest_match else "Keine Daten"

    # Genauigkeiten berechnen
    acc_rf = calculate_model_accuracy("RandomForest", season)
    acc_xgb = calculate_model_accuracy("XGBoost", season)

    # Gesamtzahl Spiele
    total_matches = Match.objects.count()

    # Alle Matches dieser Saison
    season_matches = Match.objects.filter(season=season)

    # Vorhersagen für diese Matches laden
    predictions = MatchPrediction.objects.filter(match__season=season)

    # Index: {(match_id, model_name): prediction}
    pred_dict = {
        (p.match_id, p.model_name): p for p in predictions
    }

    vergleichsdaten = []
    for match in season_matches:
        rf_pred = pred_dict.get((match.match_id, "RandomForest"))
        xgb_pred = pred_dict.get((match.match_id, "XGBoost"))

        vergleichsdaten.append({
            'spiel': f"{match.home_team} vs. {match.away_team}",
            'rf': rf_pred.predicted_result if rf_pred else "k.A.",
            'xgb': xgb_pred.predicted_result if xgb_pred else "k.A.",
            'ergebnis': match.result,
            'abweichung': (
                ('✅' if rf_pred and rf_pred.predicted_result == match.result else '❌') +
                " / " +
                ('✅' if xgb_pred and xgb_pred.predicted_result == match.result else '❌')
            )
        })

    # 📊 Ergebnisverteilung vorbereiten für Chart.js
    result_counts = Counter(match.result for match in season_matches)

    result_mapping = {
        'home_win': 'Heimsieg',
        'draw': 'Unentschieden',
        'away_win': 'Auswärtssieg',
    }

    visual_data = {
        'labels': [result_mapping.get(k, k) for k in ['home_win', 'draw', 'away_win']],
        'counts': [result_counts.get(k, 0) for k in ['home_win', 'draw', 'away_win']]
    }

    # Modellgenauigkeit über Spieltage hinweg
    rf_accuracies = defaultdict(lambda: {'correct': 0, 'total': 0})
    xgb_accuracies = defaultdict(lambda: {'correct': 0, 'total': 0})

    for match in season_matches:
        matchday = match.matchday
        rf_pred = pred_dict.get((match.match_id, "RandomForest"))
        xgb_pred = pred_dict.get((match.match_id, "XGBoost"))

        if rf_pred:
            rf_accuracies[matchday]['total'] += 1
            if rf_pred.predicted_result == match.result:
                rf_accuracies[matchday]['correct'] += 1

        if xgb_pred:
            xgb_accuracies[matchday]['total'] += 1
            if xgb_pred.predicted_result == match.result:
                xgb_accuracies[matchday]['correct'] += 1

    # Sortiert nach Spieltag
    sorted_matchdays = sorted(rf_accuracies.keys(), key=lambda x: int(''.join(filter(str.isdigit, x))))  # z.B. "Spieltag 1"

    rf_accuracy_data = [round((rf_accuracies[md]['correct'] / rf_accuracies[md]['total']) * 100, 2)
                        if rf_accuracies[md]['total'] > 0 else None
                        for md in sorted_matchdays]

    xgb_accuracy_data = [round((xgb_accuracies[md]['correct'] / xgb_accuracies[md]['total']) * 100, 2)
                         if xgb_accuracies[md]['total'] > 0 else None
                         for md in sorted_matchdays]

    model_accuracy_per_matchday = {
        'labels': sorted_matchdays,
        'rf': rf_accuracy_data,
        'xgb': xgb_accuracy_data,
    }

    # Für Visualisierung Vorhersagetypen und Unsicherheiten
    # Vorhersagetypen zählen
    def count_prediction_types(predictions):
        counter = Counter()
        for p in predictions:
            counter[p.predicted_result] += 1
        return dict(counter)

    rf_preds = [p for p in predictions if p.model_name == "RandomForest"]
    xgb_preds = [p for p in predictions if p.model_name == "XGBoost"]

    prediction_type_data = {
        "RandomForest": count_prediction_types(rf_preds),
        "XGBoost": count_prediction_types(xgb_preds),
    }


    # Tatsächliche Ergebnisverteilung
    def count_results(matches):
        counter = Counter()
        for m in matches:
            counter[m.result] += 1
        return dict(counter)

    actual_counts = count_results(season_matches)
    actual_type_data = {
        "home_win": actual_counts.get('home_win', 0),
        "draw": actual_counts.get('draw', 0),
        "away_win": actual_counts.get('away_win', 0),
    }


    # Unsicherheit = höchste Wahrscheinlichkeit der Vorhersage
    def extract_confidences(preds):
        return [
            max(p.prob_home_win, p.prob_draw, p.prob_away_win)
            for p in preds
        ]

    confidence_data = {
        "RandomForest": extract_confidences(rf_preds),
        "XGBoost": extract_confidences(xgb_preds),
    }



    context = {
        'total_matches': total_matches,
        'season': season,
        'acc_rf': acc_rf,
        'acc_xgb': acc_xgb,
        'vergleichsdaten': vergleichsdaten,
        'visual_data_json': json.dumps(visual_data),
        'accuracy_chart_data': json.dumps(model_accuracy_per_matchday),
        'prediction_type_data': json.dumps(prediction_type_data),
        'actual_type_data': json.dumps(actual_type_data),
        'confidence_data': json.dumps(confidence_data),
    }

    # Inhalt für Erklärbakeit-Tab
    rf_imp, xgb_imp = get_feature_importances()
    context['rf_feature_importances'] = json.dumps(rf_imp)
    context['xgb_feature_importances'] = json.dumps(xgb_imp)



    return render(request, 'dashboard.html', context)





# berechnet model accuracy für ausgewählte Saison
def calculate_model_accuracy(model_name, season):
    correct = 0
    total = 0

    predictions = MatchPrediction.objects.select_related('match').filter(
        model_name=model_name,
        match__season=season
    )

    for prediction in predictions:
        if prediction.match.result == prediction.predicted_result:
            correct += 1
        total += 1

    return round((correct / total) * 100, 2) if total > 0 else None


# berechnet die Importances für Erklärbakeitstab
def get_feature_importances():
    model_dir = r'C:\Users\gillo\Anwendungsprojekt\Football_Prediction\train_model\football_prediction\model'
    # Feature-Namen laden
    with open(os.path.join(model_dir, 'feature_columns.json'), 'r') as f:
        feature_names = json.load(f)

    # Modelle laden
    rf_model = joblib.load(os.path.join(model_dir, 'rf_model.joblib'))
    xgb_model = joblib.load(os.path.join(model_dir, 'xgb_model.joblib'))

    # Importance-Werte extrahieren
    rf_importances = rf_model.feature_importances_
    xgb_importances = xgb_model.feature_importances_

    # Liste aus Namen und Werten
    rf_data = sorted([(name, float(importance)) for name, importance in zip(feature_names, rf_importances)], key=lambda x: x[1], reverse=True)
    xgb_data = sorted([(name, float(importance)) for name, importance in zip(feature_names, xgb_importances)], key=lambda x: x[1], reverse=True)

    return rf_data, xgb_data







### Schon bestehender Stuff - für Backend wichtig? Oder löschen?
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


def get_win_rate(team_name, home=True):
    if home:
        matches = Match.objects.filter(home_team=team_name)
        wins = matches.filter(home_goals__gt=F('away_goals')).count()
    else:
        matches = Match.objects.filter(away_team=team_name)
        wins = matches.filter(away_goals__gt=F('home_goals')).count()
    total = matches.count()
    return wins / total if total > 0 else 0.0
