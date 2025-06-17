from sklearn.preprocessing import LabelEncoder

def encode_features(df, feature_columns, target_column):
    le_home = LabelEncoder()
    le_away = LabelEncoder()
    le_result = LabelEncoder()

    df["home_team_encoded"] = le_home.fit_transform(df["home_team"])
    df["away_team_encoded"] = le_away.fit_transform(df["away_team"])
    df["result_encoded"] = le_result.fit_transform(df[target_column])

    X = df[feature_columns]
    y = df["result_encoded"]

    encoders = {
        "home_team": le_home,
        "away_team": le_away,
        "result": le_result
    }

    return X, y, encoders
