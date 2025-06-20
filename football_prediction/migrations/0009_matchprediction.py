import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('football_prediction', '0008_matchwithavgawaygoals_matchwithavghomegoals_and_more'),
    ]

    operations = [
        migrations.CreateModel(
            name='MatchPrediction',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('model_name', models.CharField(max_length=50)),
                ('prob_home_win', models.FloatField()),
                ('prob_draw', models.FloatField()),
                ('prob_away_win', models.FloatField()),
                ('predicted_result', models.CharField(max_length=10)),
                ('timestamp', models.DateTimeField(auto_now_add=True)),
                ('match', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='football_prediction.match')),
            ],
            options={
                'unique_together': {('match', 'model_name')},
            },
        ),
    ]
