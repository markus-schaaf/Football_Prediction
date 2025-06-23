import os
from django.db import migrations

def read_sql(filename):
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    sql_path = os.path.join(base_dir, 'football_prediction', 'sql_views', filename)
    with open(sql_path, 'r', encoding='utf-8') as f:
        return f.read()

class Migration(migrations.Migration):

    dependencies = [
        ('football_prediction', '0005_create_avg_away_goals_view'),
    ]

    operations = [
        migrations.RunSQL(
            sql=read_sql('view_away_win_rate.sql'),
            reverse_sql="DROP VIEW IF EXISTS football_prediction_match_away_win_rate;"
        ),
    ]
