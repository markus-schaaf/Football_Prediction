from sklearn.preprocessing import LabelEncoder

def encode_features(df, feature_columns, target_column=None):
    le_home = LabelEncoder()
    le_away = LabelEncoder()
    le_result = LabelEncoder()

    df["home_team_encoded"] = le_home.fit_transform(df["home_team"])
    df["away_team_encoded"] = le_away.fit_transform(df["away_team"])

    encoders = {
        "home_team": le_home,
        "away_team": le_away
    }

    if target_column is not None and target_column in df.columns:
        df["result_encoded"] = le_result.fit_transform(df[target_column])
        encoders["result"] = le_result
        y = df["result_encoded"]
    else:
        y = None

    X = df[feature_columns]
    return X, y, encoders
