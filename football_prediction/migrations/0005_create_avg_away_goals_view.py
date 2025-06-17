import os
from django.db import migrations

# Funktion zum robusten Einlesen der SQL-Datei
def read_sql(filename):
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    sql_path = os.path.join(base_dir, 'football_prediction', 'sql_views', filename)
    with open(sql_path, 'r', encoding='utf-8') as f:
        return f.read()

class Migration(migrations.Migration):

    dependencies = [
        ('football_prediction', '0004_create_home_win_rate_view'),
    ]

    operations = [
        migrations.RunSQL(
            sql=read_sql('view_avg_away_goals.sql'),
            reverse_sql="DROP VIEW IF EXISTS football_prediction_match_avg_away_goals;"
        ),
    ]
