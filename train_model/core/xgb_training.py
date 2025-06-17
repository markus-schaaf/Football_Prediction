from xgboost import XGBClassifier

def train_xgb(X, y, params):
    model = XGBClassifier(**params)
    model.fit(X, y)
    return model
