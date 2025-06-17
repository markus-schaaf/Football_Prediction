from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

def train_xgb(X_train, y_train):
    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    model.fit(X_train, y_train)
    return model

def train_rf(X_train, y_train):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model
