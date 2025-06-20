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

from django.db.models import UniqueConstraint

class MatchPrediction(models.Model):
    match = models.ForeignKey('Match', on_delete=models.CASCADE, null=True, blank=True)
    model_name = models.CharField(max_length=50)

    prob_home_win = models.FloatField()
    prob_draw = models.FloatField()
    prob_away_win = models.FloatField()

    predicted_result = models.CharField(max_length=10)
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        constraints = [
            UniqueConstraint(fields=['match', 'model_name'], name='unique_prediction_per_model')
        ]

    def __str__(self):
        return f"{self.model_name} prediction – {self.predicted_result}"