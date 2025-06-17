import os
import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'anwendungsprojekt.settings')
django.setup()

from train_model.config.feature_config import FEATURE_COLUMNS, TARGET_COLUMN
from train_model.config.constants import CLASS_NAMES
from train_model.config.paths import MODEL_DIR
from train_model.config.rf_config import RF_PARAMS

from train_model.core.data_loading import load_match_data
from train_model.core.feature_engineering import prepare_features
from train_model.core.preprocessing import encode_features
from train_model.core.rf_training import train_rf
from train_model.core.evaluation import evaluate_model
from train_model.core.utils import save_model

from sklearn.model_selection import train_test_split
import json

def main():
    df = load_match_data()
    df = prepare_features(df)

    X, y, encoders = encode_features(df, FEATURE_COLUMNS, TARGET_COLUMN)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = train_rf(X_train, y_train, RF_PARAMS)

    acc, report = evaluate_model(model, X_test, y_test)
    print(f"Random Forest Accuracy: {acc:.2%}")
    print(report)

    save_model(model, os.path.join(MODEL_DIR, "rf_model.joblib"))
    save_model(encoders['home_team'], os.path.join(MODEL_DIR, "le_home.joblib"))
    save_model(encoders['away_team'], os.path.join(MODEL_DIR, "le_away.joblib"))
    save_model(encoders['result'], os.path.join(MODEL_DIR, "le_result.joblib"))

    with open(os.path.join(MODEL_DIR, "feature_columns.json"), "w") as f:
        json.dump(FEATURE_COLUMNS, f)

if __name__ == "__main__":
    main()
