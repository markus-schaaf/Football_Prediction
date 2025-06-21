from django.db import models

class MatchWithAvgHomeGoals(models.Model):
    match_id = models.BigIntegerField(primary_key=True)
    date = models.DateTimeField()
    home_team = models.CharField(max_length=100)
    home_goals = models.IntegerField()
    average_home_goals = models.FloatField()

    class Meta:
        managed = False
        db_table = 'football_prediction_match_avg_home_goals'


class MatchWithHomeWinRate(models.Model):
    match_id = models.BigIntegerField(primary_key=True)
    date = models.DateTimeField()
    home_team = models.CharField(max_length=100)
    home_goals = models.IntegerField()
    away_goals = models.IntegerField()
    home_win_rate = models.FloatField()

    class Meta:
        managed = False
        db_table = 'football_prediction_match_home_win_rate'

class MatchWithAvgAwayGoals(models.Model):
    match_id = models.BigIntegerField(primary_key=True)
    date = models.DateTimeField()
    away_team = models.CharField(max_length=100)
    away_goals = models.IntegerField()
    average_away_goals = models.FloatField()

    class Meta:
        managed = False
        db_table = 'football_prediction_match_avg_away_goals'

class MatchWithAwayWinRate(models.Model):
    match_id = models.BigIntegerField(primary_key=True)
    date = models.DateTimeField()
    away_team = models.CharField(max_length=100)
    away_goals = models.IntegerField()
    home_goals = models.IntegerField()
    away_win_rate = models.FloatField()

    class Meta:
        managed = False
        db_table = 'football_prediction_match_away_win_rate'

class MatchWithGoalAvgDiff(models.Model):
    match_id = models.IntegerField(primary_key=True)
    average_home_goals = models.FloatField()
    average_away_goals = models.FloatField()
    goal_avg_diff = models.FloatField()

    class Meta:
        managed = False
        db_table = 'football_prediction_matchwithgoalavgdiff'
