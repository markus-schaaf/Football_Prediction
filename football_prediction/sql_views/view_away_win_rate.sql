CREATE OR REPLACE VIEW football_prediction_match_away_win_rate AS
SELECT
    match_id,
    date,
    away_team,
    away_goals,
    home_goals,
    AVG(CASE WHEN away_goals > home_goals THEN 1 ELSE 0 END) OVER (PARTITION BY away_team ORDER BY date ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING) AS away_win_rate
FROM football_prediction_match;
