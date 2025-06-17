CREATE OR REPLACE VIEW football_prediction_match_avg_home_goals AS
SELECT
    match_id,
    date,
    home_team,
    home_goals,
    AVG(home_goals) OVER (PARTITION BY home_team ORDER BY date ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING) AS average_home_goals
FROM football_prediction_match;
