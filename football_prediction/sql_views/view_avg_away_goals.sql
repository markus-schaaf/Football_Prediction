CREATE OR REPLACE VIEW football_prediction_match_avg_away_goals AS
SELECT
    match_id,
    date,
    away_team,
    away_goals,
    AVG(away_goals) OVER (PARTITION BY away_team ORDER BY date ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING) AS average_away_goals
FROM football_prediction_match;
