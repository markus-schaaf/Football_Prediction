import os
import sys
import django
import pandas as pd
import joblib

# Django Setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'anwendungsprojekt.settings')
django.setup()

from football_prediction.models import Match, MatchPrediction
from train_model.config.paths import MODEL_DIR
from train_model.config.feature_config import FEATURE_COLUMNS
from train_model.train.train_xgb import load_model as load_xgb_model
from train_model.train.train_rf import load_model as load_rf_model
from train_model.core.feature_engineering import prepare_features
from train_model.core.preprocessing import encode_features
from train_model.core.data_loading import load_match_data

# Modelle & Encoder laden
xgb = load_xgb_model()
rf = load_rf_model()
le_home = joblib.load(os.path.join(MODEL_DIR, "le_home.joblib"))
le_away = joblib.load(os.path.join(MODEL_DIR, "le_away.joblib"))
le_result = joblib.load(os.path.join(MODEL_DIR, "le_result.joblib"))

# Echte Matches laden
df_matches = load_match_data()

# Feature-Engineering
try:
    df_prepared = prepare_features(df_matches)
except Exception as e:
    print(f"❌ Fehler bei Feature-Berechnung: {type(e).__name__} – {e}")
    sys.exit(1)

# Vorhersage-Schleife über alle echten Spiele
for idx, row in df_prepared.iterrows():
    match_id = row["match_id"]

    try:
        match = Match.objects.get(match_id=match_id)
    except Match.DoesNotExist:
        print(f"⚠️ Match-ID {match_id} nicht in DB gefunden – übersprungen.")
        continue

    try:
        row_df = pd.DataFrame([row])
        df_encoded, _, _ = encode_features(row_df, FEATURE_COLUMNS, target_column=None)
        X_input = df_encoded[FEATURE_COLUMNS]
    except Exception as e:
        print(f"❌ Fehler beim Encoding für Match {match_id}: {type(e).__name__} – {e}")
        continue

    for model, model_name in [(rf, 'RandomForest'), (xgb, 'XGBoost')]:
        try:
            probs = model.predict_proba(X_input)[0]
            predicted_class = model.classes_[probs.argmax()]
            predicted_label = le_result.inverse_transform([predicted_class])[0]

            MatchPrediction.objects.update_or_create(
                match=match,
                model_name=model_name,
                defaults={
                    'prob_home_win': probs[0],
                    'prob_draw': probs[1],
                    'prob_away_win': probs[2],
                    'predicted_result': predicted_label
                }
            )

        except Exception as e:
            print(f"❌ Fehler bei Vorhersage für Match {match_id} mit {model_name}: {type(e).__name__} – {e}")
            continue

print("✅ Alle Vorhersagen wurden erfolgreich berechnet und gespeichert.")
