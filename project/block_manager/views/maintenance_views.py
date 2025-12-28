"""
Maintenance endpoints for system administration tasks.
"""
import logging
import time
from pathlib import Path
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.conf import settings

logger = logging.getLogger(__name__)


def _cleanup_old_files(retention_hours=None):
    """
    Internal function to clean up old uploaded files.

    Args:
        retention_hours: Number of hours to retain files (default from settings)

    Returns:
        dict: Cleanup statistics
    """
    if retention_hours is None:
        retention_hours = getattr(settings, 'UPLOAD_RETENTION_HOURS', 2)

    retention_seconds = retention_hours * 3600
    upload_dir = Path(getattr(settings, 'TEMP_UPLOAD_DIR', '/tmp/visionforge_uploads'))

    if not upload_dir.exists():
        return {
            'deleted_count': 0,
            'deleted_size_mb': 0,
            'error_count': 0,
            'message': 'Upload directory does not exist'
        }

    current_time = time.time()
    deleted_count = 0
    deleted_size = 0
    error_count = 0

    for file_path in upload_dir.rglob('*'):
        if not file_path.is_file():
            continue

        try:
            file_age = current_time - file_path.stat().st_mtime

            if file_age > retention_seconds:
                file_size = file_path.stat().st_size
                file_path.unlink()
                logger.info(f'Deleted old upload: {file_path.name} (age: {file_age/3600:.1f}h)')
                deleted_count += 1
                deleted_size += file_size

        except Exception as e:
            error_count += 1
            logger.error(f'Error processing file {file_path}: {str(e)}')

    # Clean up empty directories
    for dir_path in sorted(upload_dir.rglob('*'), reverse=True):
        if dir_path.is_dir() and not any(dir_path.iterdir()):
            try:
                dir_path.rmdir()
                logger.info(f'Removed empty directory: {dir_path}')
            except Exception as e:
                logger.error(f'Error removing directory {dir_path}: {str(e)}')

    return {
        'deleted_count': deleted_count,
        'deleted_size_mb': round(deleted_size / 1024 / 1024, 2),
        'error_count': error_count,
        'retention_hours': retention_hours
    }


@csrf_exempt
@require_http_methods(["POST"])
def trigger_file_cleanup(request):
    """
    Endpoint to trigger file cleanup remotely.
    Protected by a secret token to prevent unauthorized access.

    POST /api/v1/maintenance/cleanup-files
    Body: {"secret": "your-secret-token"}

    Returns:
        200: Cleanup completed successfully
        401: Unauthorized
    """
    import json

    try:
        data = json.loads(request.body)
        provided_secret = data.get('secret', '')

        # Verify secret token
        expected_secret = settings.CLEANUP_SECRET_TOKEN
        if not expected_secret or provided_secret != expected_secret:
            logger.warning(f"Unauthorized cleanup attempt from IP {request.META.get('REMOTE_ADDR')}")
            return JsonResponse({
                'error': 'Unauthorized',
                'message': 'Invalid secret token'
            }, status=401)

        # Run cleanup
        stats = _cleanup_old_files()
        logger.info(f"File cleanup completed: {stats['deleted_count']} files, {stats['deleted_size_mb']}MB")

        return JsonResponse({
            'success': True,
            'message': 'File cleanup completed',
            'stats': stats
        })

    except json.JSONDecodeError:
        return JsonResponse({
            'error': 'Invalid JSON',
            'message': 'Request body must be valid JSON'
        }, status=400)
    except Exception as e:
        logger.error(f"Error in trigger_file_cleanup: {str(e)}", exc_info=True)
        return JsonResponse({
            'error': 'Server error',
            'message': 'An error occurred during cleanup'
        }, status=500)


@csrf_exempt
@require_http_methods(["GET"])
def get_upload_stats(request):
    """
    Get statistics about uploaded files.

    GET /api/v1/maintenance/upload-stats?secret=your-secret-token

    Returns:
        200: Upload statistics
        401: Unauthorized
    """
    try:
        provided_secret = request.GET.get('secret', '')

        # Verify secret token
        expected_secret = settings.CLEANUP_SECRET_TOKEN
        if not expected_secret or provided_secret != expected_secret:
            logger.warning(f"Unauthorized stats access attempt from IP {request.META.get('REMOTE_ADDR')}")
            return JsonResponse({
                'error': 'Unauthorized',
                'message': 'Invalid secret token'
            }, status=401)

        from block_manager.utils.file_cleanup import get_upload_directory_size
        upload_dir = Path(settings.TEMP_UPLOAD_DIR)
        total_size = get_upload_directory_size()
        file_count = sum(1 for _ in upload_dir.rglob('*') if _.is_file()) if upload_dir.exists() else 0

        # Get oldest file age
        oldest_age = None
        if upload_dir.exists():
            files = [f for f in upload_dir.rglob('*') if f.is_file()]
            if files:
                oldest_file = min(files, key=lambda f: f.stat().st_mtime)
                oldest_age = (time.time() - oldest_file.stat().st_mtime) / 3600  # hours

        return JsonResponse({
            'success': True,
            'stats': {
                'total_size_mb': round(total_size / 1024 / 1024, 2),
                'file_count': file_count,
                'oldest_file_age_hours': round(oldest_age, 2) if oldest_age else None,
                'retention_hours': settings.UPLOAD_RETENTION_HOURS,
                'upload_directory': str(upload_dir)
            }
        })

    except Exception as e:
        logger.error(f"Error in get_upload_stats: {str(e)}", exc_info=True)
        return JsonResponse({
            'error': 'Server error',
            'message': 'An error occurred while fetching stats'
        }, status=500)
