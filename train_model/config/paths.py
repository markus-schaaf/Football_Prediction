import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "football_prediction", "model")
STATIC_DIR = os.path.join(BASE_DIR, "football_prediction", "static", "model")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)
