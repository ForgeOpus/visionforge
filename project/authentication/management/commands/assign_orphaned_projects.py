"""
Django management command to assign orphaned projects to a specific user.

Usage:
    # Preview changes (dry run)
    python manage.py assign_orphaned_projects user@example.com --dry-run

    # Execute assignment
    python manage.py assign_orphaned_projects user@example.com
"""
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction
from authentication.models import User
from block_manager.models import Project


class Command(BaseCommand):
    help = 'Assign projects with no user to a specified user by email'

    def add_arguments(self, parser):
        parser.add_argument(
            'email',
            type=str,
            help='Email address of the user to assign orphaned projects to'
        )
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Preview changes without actually modifying the database'
        )

    def handle(self, *args, **options):
        email = options['email']
        dry_run = options['dry_run']

        # Find the target user
        try:
            user = User.objects.get(email=email)
        except User.DoesNotExist:
            raise CommandError(f'User with email "{email}" not found')

        # Find all orphaned projects (projects with no user assigned)
        orphaned_projects = Project.objects.filter(user__isnull=True)
        count = orphaned_projects.count()

        if count == 0:
            self.stdout.write(self.style.SUCCESS('No orphaned projects found.'))
            return

        # Display information
        self.stdout.write(self.style.WARNING(f'\nFound {count} orphaned project(s):'))
        self.stdout.write('-' * 80)

        for project in orphaned_projects:
            self.stdout.write(f'  • {project.name}')
            self.stdout.write(f'    ID: {project.id}')
            self.stdout.write(f'    Framework: {project.framework}')
            self.stdout.write(f'    Created: {project.created_at}')
            self.stdout.write('')

        self.stdout.write('-' * 80)
        self.stdout.write(f'Target User: {user.display_name or user.email}')
        self.stdout.write(f'  Email: {user.email}')
        self.stdout.write(f'  Current project count: {user.project_count}')
        self.stdout.write(f'  New project count: {user.project_count + count}')
        self.stdout.write('-' * 80)

        if dry_run:
            self.stdout.write(self.style.WARNING('\n[DRY RUN] No changes made.'))
            self.stdout.write(f'Run without --dry-run flag to assign {count} project(s) to {user.email}')
            return

        # Execute assignment
        try:
            with transaction.atomic():
                # Assign all orphaned projects to the user
                orphaned_projects.update(user=user)

                # Update user's project count
                user.project_count += count
                user.save(update_fields=['project_count'])

            self.stdout.write(self.style.SUCCESS(f'\n✓ Successfully assigned {count} project(s) to {user.email}'))
            self.stdout.write(self.style.SUCCESS(f'✓ Updated {user.email}\'s project count to {user.project_count}'))
        except Exception as e:
            raise CommandError(f'Failed to assign projects: {str(e)}')
