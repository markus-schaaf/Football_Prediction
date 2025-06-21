from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('football_prediction', '0007_create_goal_avg_diff_view'),
    ]

    operations = [
        migrations.CreateModel(
            name='MatchWithAvgAwayGoals',
            fields=[
                ('match_id', models.BigIntegerField(primary_key=True, serialize=False)),
                ('date', models.DateTimeField()),
                ('away_team', models.CharField(max_length=100)),
                ('away_goals', models.IntegerField()),
                ('average_away_goals', models.FloatField()),
            ],
            options={
                'db_table': 'football_prediction_match_avg_away_goals',
                'managed': False,
            },
        ),
        migrations.CreateModel(
            name='MatchWithAvgHomeGoals',
            fields=[
                ('match_id', models.BigIntegerField(primary_key=True, serialize=False)),
                ('date', models.DateTimeField()),
                ('home_team', models.CharField(max_length=100)),
                ('home_goals', models.IntegerField()),
                ('average_home_goals', models.FloatField()),
            ],
            options={
                'db_table': 'football_prediction_match_avg_home_goals',
                'managed': False,
            },
        ),
        migrations.CreateModel(
            name='MatchWithAwayWinRate',
            fields=[
                ('match_id', models.BigIntegerField(primary_key=True, serialize=False)),
                ('date', models.DateTimeField()),
                ('away_team', models.CharField(max_length=100)),
                ('away_goals', models.IntegerField()),
                ('home_goals', models.IntegerField()),
                ('away_win_rate', models.FloatField()),
            ],
            options={
                'db_table': 'football_prediction_match_away_win_rate',
                'managed': False,
            },
        ),
        migrations.CreateModel(
            name='MatchWithGoalAvgDiff',
            fields=[
                ('match_id', models.IntegerField(primary_key=True, serialize=False)),
                ('average_home_goals', models.FloatField()),
                ('average_away_goals', models.FloatField()),
                ('goal_avg_diff', models.FloatField()),
            ],
            options={
                'db_table': 'football_prediction_matchwithgoalavgdiff',
                'managed': False,
            },
        ),
        migrations.CreateModel(
            name='MatchWithHomeWinRate',
            fields=[
                ('match_id', models.BigIntegerField(primary_key=True, serialize=False)),
                ('date', models.DateTimeField()),
                ('home_team', models.CharField(max_length=100)),
                ('home_goals', models.IntegerField()),
                ('away_goals', models.IntegerField()),
                ('home_win_rate', models.FloatField()),
            ],
            options={
                'db_table': 'football_prediction_match_home_win_rate',
                'managed': False,
            },
        ),
    ]
