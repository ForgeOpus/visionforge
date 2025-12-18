"""
Health check endpoint for monitoring
"""
from django.http import JsonResponse
from django.utils import timezone
from django.db import connection
import firebase_admin


def health_check(request):
    """
    Health check endpoint for monitoring and load balancers

    Returns:
        JSON response with health status of all system components
    """
    health_status = {
        'status': 'healthy',
        'timestamp': timezone.now().isoformat(),
        'components': {}
    }

    # Check database connection
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
            cursor.fetchone()
        health_status['components']['database'] = {
            'status': 'healthy',
            'type': 'oracle'
        }
    except Exception as e:
        health_status['status'] = 'degraded'
        health_status['components']['database'] = {
            'status': 'unhealthy',
            'error': str(e)
        }

    # Check Firebase connection
    try:
        firebase_app = firebase_admin.get_app()
        health_status['components']['firebase'] = {
            'status': 'healthy',
            'project_id': firebase_app.project_id
        }
    except Exception as e:
        health_status['status'] = 'degraded'
        health_status['components']['firebase'] = {
            'status': 'unhealthy',
            'error': str(e)
        }

    # Determine HTTP status code
    status_code = 200 if health_status['status'] == 'healthy' else 503

    return JsonResponse(health_status, status=status_code)
