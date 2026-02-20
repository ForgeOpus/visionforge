import uuid
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('block_manager', '0004_add_user_to_project'),
    ]

    operations = [
        migrations.AddField(
            model_name='project',
            name='share_token',
            field=models.UUIDField(
                blank=True,
                db_index=True,
                default=None,
                help_text='Unique token for public sharing; generated on first share',
                null=True,
                unique=True,
            ),
        ),
        migrations.AddField(
            model_name='project',
            name='is_shared',
            field=models.BooleanField(
                default=False,
                help_text='Whether this project is publicly accessible via share link',
            ),
        ),
    ]
