CREATE OR REPLACE VIEW football_prediction_matchwithgoalavgdiff AS
SELECT
    h.match_id,
    h.average_home_goals,
    a.average_away_goals,
    h.average_home_goals - a.average_away_goals AS goal_avg_diff
FROM football_prediction_match_avg_home_goals h
JOIN football_prediction_match_avg_away_goals a ON h.match_id = a.match_id;
