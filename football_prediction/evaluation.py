import json
import os
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

def evaluate_and_save(y_true, y_pred, model_name, class_names, save_path):
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    cm = confusion_matrix(y_true, y_pred).tolist()

    metrics = {
        "model": model_name,
        "accuracy": acc,
        "classification_report": report,
        "confusion_matrix": cm,
        "classes": list(class_names)
    }

    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"📊 Evaluation gespeichert unter: {save_path}")
