CREATE OR REPLACE VIEW football_prediction_match_away_win_rate AS
SELECT
  match_id,
  date,
  away_team,
  away_goals,
  home_goals,
  SUM(CASE WHEN away_goals > home_goals THEN 1 ELSE 0 END) OVER (
    PARTITION BY away_team
    ORDER BY date
    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
  ) /
  NULLIF(
    COUNT(*) OVER (
      PARTITION BY away_team
      ORDER BY date
      ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ), 0
  ) AS away_win_rate
FROM football_prediction_match;
