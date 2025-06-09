from django.db import migrations

class Migration(migrations.Migration):

    dependencies = [
        ('football_prediction', '0005_create_avg_away_goals_view'),
    ]

    operations = [
        migrations.RunSQL(
            sql=open('football_prediction/sql_views/view_away_win_rate.sql').read(),
            reverse_sql="DROP VIEW IF EXISTS football_prediction_match_away_win_rate;"
        ),
    ]
