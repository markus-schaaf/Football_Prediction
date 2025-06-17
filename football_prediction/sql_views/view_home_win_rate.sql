CREATE OR REPLACE VIEW football_prediction_match_home_win_rate AS
SELECT
    match_id,
    date,
    home_team,
    home_goals,
    away_goals,
    AVG(CASE WHEN home_goals > away_goals THEN 1 ELSE 0 END) OVER (PARTITION BY home_team ORDER BY date ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING) AS home_win_rate
FROM football_prediction_match;
