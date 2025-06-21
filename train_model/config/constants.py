FEATURE_COLUMNS = [
    'home_team_encoded', 'away_team_encoded',
    'home_position', 'away_position',
    'average_home_goals', 'average_away_goals',
    'home_win_rate', 'away_win_rate',
    'home_form_points', 'away_form_points',
    'home_form_goaldiff', 'away_form_goaldiff',
    'form_diff', 'goal_avg_diff',
    'form_curve_diff',
    'elo_home', 'elo_away', 'elo_diff',
]

TARGET_COLUMN = 'result'

CLASS_NAMES = ['away_win', 'draw', 'home_win']
