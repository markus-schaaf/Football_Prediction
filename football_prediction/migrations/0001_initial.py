from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Match',
            fields=[
                ('match_id', models.BigIntegerField(primary_key=True, serialize=False)),
                ('date', models.DateTimeField()),
                ('matchday', models.CharField(max_length=50)),
                ('home_team', models.CharField(max_length=100)),
                ('away_team', models.CharField(max_length=100)),
                ('home_goals', models.IntegerField()),
                ('away_goals', models.IntegerField()),
                ('scorers', models.TextField()),
                ('season', models.CharField(max_length=20)),
                ('result', models.CharField(max_length=20)),
            ],
        ),
    ]
