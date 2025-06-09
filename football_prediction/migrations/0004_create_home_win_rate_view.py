from django.db import migrations

class Migration(migrations.Migration):

    dependencies = [
        ('football_prediction', '0003_create_avg_home_goals_view'),
    ]

    operations = [
        migrations.RunSQL(
            sql=open('football_prediction/sql_views/view_home_win_rate.sql').read(),
            reverse_sql="DROP VIEW IF EXISTS football_prediction_match_home_win_rate;"
        ),
    ]
