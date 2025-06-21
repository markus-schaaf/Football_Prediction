from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('football_prediction', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='match',
            name='elo_away',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='match',
            name='elo_home',
            field=models.FloatField(blank=True, null=True),
        ),
    ]
