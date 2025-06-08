from django.db import migrations

class Migration(migrations.Migration):

    dependencies = [
        ('football_prediction', '0002_match_elo_away_match_elo_home'),  # exakt übernehmen
    ]

    operations = [
        migrations.RunSQL(
            sql=open('football_prediction/sql_views/view_avg_home_goals.sql').read(),
            reverse_sql="DROP VIEW IF EXISTS football_prediction_match_avg_home_goals;"
        ),
    ]
