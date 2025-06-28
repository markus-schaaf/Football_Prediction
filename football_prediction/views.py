from django.shortcuts import render
from football_prediction.models import Match, MatchPrediction
from collections import Counter, defaultdict
import json
import os
import joblib


def dashboard_view(request):
    latest_match = Match.objects.order_by('-season').first()
    season = latest_match.season if latest_match else "Keine Daten"

    acc_rf = calculate_model_accuracy("RandomForest", season)
    acc_xgb = calculate_model_accuracy("XGBoost", season)
    total_matches = Match.objects.count()
    season_matches = Match.objects.filter(season=season)
    predictions = MatchPrediction.objects.filter(match__season=season)

    pred_dict = {(p.match_id, p.model_name): p for p in predictions}

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

    result_counts = Counter(match.result for match in season_matches)
    result_mapping = {'home_win': 'Heimsieg', 'draw': 'Unentschieden', 'away_win': 'Auswärtssieg'}
    visual_data = {
        'labels': [result_mapping.get(k, k) for k in ['home_win', 'draw', 'away_win']],
        'counts': [result_counts.get(k, 0) for k in ['home_win', 'draw', 'away_win']]
    }

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

    sorted_matchdays = sorted(rf_accuracies.keys(), key=lambda x: int(''.join(filter(str.isdigit, x))))
    rf_accuracy_data = [round((rf_accuracies[md]['correct'] / rf_accuracies[md]['total']) * 100, 2)
                        if rf_accuracies[md]['total'] > 0 else None for md in sorted_matchdays]
    xgb_accuracy_data = [round((xgb_accuracies[md]['correct'] / xgb_accuracies[md]['total']) * 100, 2)
                         if xgb_accuracies[md]['total'] > 0 else None for md in sorted_matchdays]
    model_accuracy_per_matchday = {
        'labels': sorted_matchdays,
        'rf': rf_accuracy_data,
        'xgb': xgb_accuracy_data,
    }

    def count_prediction_types(preds):
        counter = Counter()
        for p in preds:
            counter[p.predicted_result] += 1
        return dict(counter)

    rf_preds = [p for p in predictions if p.model_name == "RandomForest"]
    xgb_preds = [p for p in predictions if p.model_name == "XGBoost"]
    prediction_type_data = {
        "RandomForest": count_prediction_types(rf_preds),
        "XGBoost": count_prediction_types(xgb_preds),
    }

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

    def extract_confidences(preds):
        return [max(p.prob_home_win, p.prob_draw, p.prob_away_win) for p in preds]

    confidence_data = {
        "RandomForest": extract_confidences(rf_preds),
        "XGBoost": extract_confidences(xgb_preds),
    }

    teams = sorted(set(Match.objects.values_list('home_team', flat=True).distinct()))

    rf_imp, xgb_imp = get_feature_importances()

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
        'teams': teams,
        'rf_feature_importances': json.dumps(rf_imp),
        'xgb_feature_importances': json.dumps(xgb_imp),
    }

    return render(request, 'dashboard.html', context)


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


def get_feature_importances():
    model_dir = r'.\train_model\football_prediction\model'
    with open(os.path.join(model_dir, 'feature_columns.json'), 'r') as f:
        feature_names = json.load(f)

    rf_model = joblib.load(os.path.join(model_dir, 'rf_model.joblib'))
    xgb_model = joblib.load(os.path.join(model_dir, 'xgb_model.joblib'))

    rf_data = sorted([(name, float(imp)) for name, imp in zip(feature_names, rf_model.feature_importances_)], key=lambda x: x[1], reverse=True)
    xgb_data = sorted([(name, float(imp)) for name, imp in zip(feature_names, xgb_model.feature_importances_)], key=lambda x: x[1], reverse=True)

    return rf_data, xgb_data
