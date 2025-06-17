# football_prediction/tests/test_models.py

from django.test import TestCase
from football_prediction.models import Match

class MatchModelTest(TestCase):
    def test_create_match(self):
        match = Match.objects.create(
            match_id=1,
            date="2023-08-01 18:30",
            matchday="1. Spieltag",
            home_team="1. FC Köln",
            away_team="FC Bayern",
            home_goals=1,
            away_goals=2,
            scorers="[]",
            season="2023/2024",
            result="away_win",
            elo_home=1500.0,
            elo_away=1600.0,
        )
        self.assertEqual(match.home_team, "1. FC Köln")
        self.assertEqual(match.away_goals, 2)
