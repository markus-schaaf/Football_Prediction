from django.shortcuts import render

def football_prediction(request):
    return render (request, 'homepage.html')

import json

def model_overview(request):
    with open("football_prediction/model/feature_importance.json", "r") as f:
        feature_data = json.load(f)
    return render(request, "overview.html", {
        "feature_names": feature_data["features"],
        "importances": feature_data["importances"]
    })