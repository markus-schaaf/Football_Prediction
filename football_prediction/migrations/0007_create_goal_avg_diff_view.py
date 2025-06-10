from django.db import migrations

class Migration(migrations.Migration):

    dependencies = [
        ('football_prediction', '0006_create_away_win_rate_view'),
        ('football_prediction', '0005_create_avg_away_goals_view'),
        ('football_prediction', '0003_create_avg_home_goals_view'),
    ]


    operations = [
        migrations.RunSQL(
            sql=open('football_prediction/sql_views/view_goal_avg_diff.sql').read(),
            reverse_sql="DROP VIEW IF EXISTS football_prediction_matchwithgoalavgdiff;"
        ),
    ]
