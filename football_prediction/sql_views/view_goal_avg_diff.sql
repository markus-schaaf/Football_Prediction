CREATE OR REPLACE VIEW football_prediction_matchwithgoalavgdiff AS
SELECT
    m.match_id,
    h.average_home_goals,
    a.average_away_goals,
    (h.average_home_goals - a.average_away_goals) AS goal_avg_diff
FROM football_prediction_match m
JOIN football_prediction_matchwithavghomegoals h ON m.match_id = h.match_id
JOIN football_prediction_match_avg_away_goals a ON m.match_id = a.match_id;
