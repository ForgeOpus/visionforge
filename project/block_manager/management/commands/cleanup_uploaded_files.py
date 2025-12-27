"""
Django management command to clean up old uploaded files.
Usage: python manage.py cleanup_uploaded_files
"""
import os
import time
import logging
from pathlib import Path
from django.core.management.base import BaseCommand
from django.conf import settings

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Clean up uploaded files older than the retention period'

    def add_arguments(self, parser):
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Show what would be deleted without actually deleting',
        )
        parser.add_argument(
            '--retention-hours',
            type=int,
            default=getattr(settings, 'UPLOAD_RETENTION_HOURS', 2),
            help='Number of hours to retain files (default: 2)',
        )

    def handle(self, *args, **options):
        dry_run = options['dry_run']
        retention_hours = options['retention_hours']
        retention_seconds = retention_hours * 3600

        upload_dir = Path(getattr(settings, 'TEMP_UPLOAD_DIR', '/tmp/visionforge_uploads'))

        if not upload_dir.exists():
            self.stdout.write(self.style.WARNING(f'Upload directory does not exist: {upload_dir}'))
            return

        current_time = time.time()
        deleted_count = 0
        deleted_size = 0
        error_count = 0

        self.stdout.write(f'Scanning directory: {upload_dir}')
        self.stdout.write(f'Retention period: {retention_hours} hours')

        for file_path in upload_dir.rglob('*'):
            if not file_path.is_file():
                continue

            try:
                file_age = current_time - file_path.stat().st_mtime

                if file_age > retention_seconds:
                    file_size = file_path.stat().st_size

                    if dry_run:
                        self.stdout.write(
                            self.style.WARNING(
                                f'Would delete: {file_path.name} '
                                f'(age: {file_age/3600:.1f}h, size: {file_size/1024:.1f}KB)'
                            )
                        )
                    else:
                        file_path.unlink()
                        logger.info(f'Deleted old upload: {file_path.name} (age: {file_age/3600:.1f}h)')
                        self.stdout.write(
                            self.style.SUCCESS(
                                f'Deleted: {file_path.name} '
                                f'(age: {file_age/3600:.1f}h, size: {file_size/1024:.1f}KB)'
                            )
                        )

                    deleted_count += 1
                    deleted_size += file_size

            except Exception as e:
                error_count += 1
                logger.error(f'Error processing file {file_path}: {str(e)}')
                self.stdout.write(self.style.ERROR(f'Error processing {file_path.name}: {str(e)}'))

        # Clean up empty directories
        if not dry_run:
            for dir_path in sorted(upload_dir.rglob('*'), reverse=True):
                if dir_path.is_dir() and not any(dir_path.iterdir()):
                    try:
                        dir_path.rmdir()
                        self.stdout.write(self.style.SUCCESS(f'Removed empty directory: {dir_path}'))
                    except Exception as e:
                        logger.error(f'Error removing directory {dir_path}: {str(e)}')

        # Summary
        self.stdout.write(self.style.SUCCESS('\n' + '='*50))
        if dry_run:
            self.stdout.write(self.style.WARNING('DRY RUN - No files were actually deleted'))
        self.stdout.write(self.style.SUCCESS(f'Files processed: {deleted_count}'))
        self.stdout.write(self.style.SUCCESS(f'Total size: {deleted_size/1024/1024:.2f} MB'))
        if error_count > 0:
            self.stdout.write(self.style.ERROR(f'Errors encountered: {error_count}'))
        self.stdout.write(self.style.SUCCESS('='*50))
