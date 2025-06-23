from django.core.management.base import BaseCommand
import os
import pandas as pd
from django.utils import timezone
from football_prediction.models import Match

class Command(BaseCommand):
    help = 'Importiert Match-Daten aus Parquet-Datei'

    def handle(self, *args, **kwargs):
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        file_path = os.path.join(base_dir, 'data_lake', 'processed', 'matches_cleaned.parquet')

        df = pd.read_parquet(file_path)

        tz = timezone.get_current_timezone()

        count = 0
        for _, row in df.iterrows():
            aware_date = row['date'].replace(tzinfo=tz)

            match, created = Match.objects.update_or_create(
            match_id=row['match_id'],
            defaults={
                'date': aware_date,
                'matchday': row['matchday'],
                'home_team': row['home_team'],
                'away_team': row['away_team'],
                'home_goals': row['home_goals'],
                'away_goals': row['away_goals'],
                'scorers': row['scorers'],
                'season': row['season'],
                'result': row['result'],
                'elo_home': row.get('elo_home', None),  
                'elo_away': row.get('elo_away', None), 
            }
        )
            if created:
                count += 1

        self.stdout.write(self.style.SUCCESS(f' {count} Matches erfolgreich importiert.'))