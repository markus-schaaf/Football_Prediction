from django.db import migrations

class Migration(migrations.Migration):

    dependencies = [
        ('football_prediction', '0004_create_home_win_rate_view'),
    ]

    operations = [
        migrations.RunSQL(
            sql=open('football_prediction/sql_views/view_avg_away_goals.sql').read(),
            reverse_sql="DROP VIEW IF EXISTS football_prediction_match_avg_away_goals;"
        ),
    ]
