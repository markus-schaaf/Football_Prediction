import os
from django.db import migrations

# Funktion zum robusten Laden der SQL-Datei
def read_sql(filename):
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    sql_path = os.path.join(base_dir, 'football_prediction', 'sql_views', filename)
    with open(sql_path, 'r', encoding='utf-8') as f:
        return f.read()

class Migration(migrations.Migration):

    dependencies = [
        ('football_prediction', '0003_create_avg_home_goals_view'),
    ]

    operations = [
        migrations.RunSQL(
            sql=read_sql('view_home_win_rate.sql'),
            reverse_sql="DROP VIEW IF EXISTS football_prediction_match_home_win_rate;"
        ),
    ]
