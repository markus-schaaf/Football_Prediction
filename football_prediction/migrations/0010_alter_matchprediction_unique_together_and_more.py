import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('football_prediction', '0009_matchprediction'),
    ]

    operations = [
        migrations.AlterUniqueTogether(
            name='matchprediction',
            unique_together=set(),
        ),
        migrations.AlterField(
            model_name='matchprediction',
            name='match',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, to='football_prediction.match'),
        ),
        migrations.AddConstraint(
            model_name='matchprediction',
            constraint=models.UniqueConstraint(fields=('match', 'model_name'), name='unique_prediction_per_model'),
        ),
    ]
