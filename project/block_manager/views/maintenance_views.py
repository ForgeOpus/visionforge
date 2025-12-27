"""
Maintenance endpoints for system administration tasks.
"""
import logging
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.conf import settings
from django.core.management import call_command
from io import StringIO

logger = logging.getLogger(__name__)


@csrf_exempt
@require_http_methods(["POST"])
def trigger_file_cleanup(request):
    """
    Endpoint to trigger file cleanup remotely.
    Protected by a secret token to prevent unauthorized access.

    POST /api/maintenance/cleanup-files
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

        # Run cleanup command
        output = StringIO()
        call_command('cleanup_uploaded_files', stdout=output)
        result = output.getvalue()

        logger.info(f"File cleanup completed successfully from remote trigger")

        return JsonResponse({
            'success': True,
            'message': 'File cleanup completed',
            'output': result
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

    GET /api/maintenance/upload-stats?secret=your-secret-token

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
        from pathlib import Path
        import time

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
