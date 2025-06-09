from django.db import models

class Match(models.Model):
    match_id = models.BigIntegerField(primary_key=True)
    date = models.DateTimeField()
    matchday = models.CharField(max_length=50)
    home_team = models.CharField(max_length=100)
    away_team = models.CharField(max_length=100)
    home_goals = models.IntegerField()
    away_goals = models.IntegerField()
    scorers = models.TextField()
    season = models.CharField(max_length=20)
    result = models.CharField(max_length=20)
    elo_home = models.FloatField(null=True, blank=True)
    elo_away = models.FloatField(null=True, blank=True)

    def __str__(self):
        return f"{self.date.date()} {self.home_team} vs {self.away_team}"
