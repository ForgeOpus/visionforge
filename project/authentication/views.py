"""
Authentication API views for Firebase integration.
"""
from typing import Dict, Any
from django.http import JsonResponse, HttpRequest
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.utils import timezone
from django_ratelimit.decorators import ratelimit
import json
import traceback

from .firebase_auth import verify_firebase_token, get_user_info_from_token
from .models import User
from .middleware import require_authentication


@csrf_exempt
@require_http_methods(["POST"])
@ratelimit(key='ip', rate='30/m', method='POST', block=True)
def verify_token(request: HttpRequest) -> JsonResponse:
    """
    Verify Firebase ID token and create/update user in database.

    POST /api/auth/verify-token
    Body: {
        "token": "firebase_id_token"
    }

    Returns:
        200: User data if successful
        400: Invalid request
        401: Invalid token
    """
    try:
        data = json.loads(request.body)
        token = data.get('token')

        if not token:
            return JsonResponse({
                'error': 'Token required',
                'message': 'Firebase ID token is required'
            }, status=400)

        # Verify token
        decoded_token = verify_firebase_token(token)
        if not decoded_token:
            return JsonResponse({
                'error': 'Invalid token',
                'message': 'Failed to verify Firebase token'
            }, status=401)

        # Extract user info
        user_info = get_user_info_from_token(decoded_token)
        if not user_info or not user_info.get('firebase_uid'):
            return JsonResponse({
                'error': 'Invalid token data',
                'message': 'Could not extract user information from token'
            }, status=401)

        # Map Firebase provider to our format
        provider_map = {
            'google.com': 'google',
            'github.com': 'github',
            'password': 'email',
        }
        auth_provider = provider_map.get(
            user_info.get('auth_provider'),
            user_info.get('auth_provider', 'unknown')
        )

        # Create or update user (using try/except to avoid Oracle ORA-12839 error)
        try:
            user = User.objects.get(firebase_uid=user_info['firebase_uid'])
            # Update existing user
            user.email = user_info.get('email', '')
            user.display_name = user_info.get('display_name', '')
            user.avatar_url = user_info.get('avatar_url', '')
            user.auth_provider = auth_provider
            user.last_login_at = timezone.now()
            user.save()
            user.increment_session()
            created = False
        except User.DoesNotExist:
            # Create new user
            user = User.objects.create(
                firebase_uid=user_info['firebase_uid'],
                email=user_info.get('email', ''),
                display_name=user_info.get('display_name', ''),
                avatar_url=user_info.get('avatar_url', ''),
                auth_provider=auth_provider,
                last_login_at=timezone.now(),
            )
            created = True

        return JsonResponse({
            'success': True,
            'user': {
                'firebase_uid': user.firebase_uid,
                'email': user.email,
                'display_name': user.display_name,
                'avatar_url': user.avatar_url,
                'auth_provider': user.auth_provider,
                'tier': user.tier,
                'project_count': user.project_count,
                'created_at': user.created_at.isoformat() if user.created_at else None,
                'last_login_at': user.last_login_at.isoformat() if user.last_login_at else None,
            },
            'is_new_user': created,
        }, status=200)

    except json.JSONDecodeError:
        return JsonResponse({
            'error': 'Invalid JSON',
            'message': 'Request body must be valid JSON'
        }, status=400)
    except Exception as e:
        traceback.print_exc()  # Print full traceback to console for debugging
        return JsonResponse({
            'error': 'Server error',
            'message': 'An unexpected error occurred. Please try again later.'
        }, status=500)


@csrf_exempt
@require_http_methods(["GET"])
@require_authentication
def get_current_user(request: HttpRequest) -> JsonResponse:
    """
    Get current authenticated user information.

    GET /api/auth/me
    Headers: Authorization: Bearer <firebase_token>

    Returns:
        200: User data
        401: Not authenticated
    """
    user = request.firebase_user

    return JsonResponse({
        'success': True,
        'user': {
            'firebase_uid': user.firebase_uid,
            'email': user.email,
            'display_name': user.display_name,
            'avatar_url': user.avatar_url,
            'auth_provider': user.auth_provider,
            'tier': user.tier,
            'project_count': user.project_count,
            'total_sessions': user.total_sessions,
            'total_time_spent_minutes': user.total_time_spent_minutes,
            'created_at': user.created_at.isoformat() if user.created_at else None,
            'last_login_at': user.last_login_at.isoformat() if user.last_login_at else None,
            'first_model_created_at': user.first_model_created_at.isoformat() if user.first_model_created_at else None,
            'first_export_at': user.first_export_at.isoformat() if user.first_export_at else None,
            'days_to_first_export': user.days_to_first_export,
        }
    }, status=200)


@csrf_exempt
@require_http_methods(["POST"])
@require_authentication
@ratelimit(key='user_or_ip', rate='120/h', method='POST', block=True)
def update_session(request):
    """
    Update session information (increment session count, add time spent).

    POST /api/auth/update-session
    Body: {
        "minutes": 5  // optional, minutes to add to total time spent
    }

    Returns:
        200: Updated session data
        401: Not authenticated
    """
    try:
        user = request.firebase_user

        # Parse request body if present
        minutes = 0
        if request.body:
            data = json.loads(request.body)
            minutes = data.get('minutes', 0)

        # Add time spent if provided
        if minutes > 0:
            user.add_time_spent(minutes)

        # Update last login
        user.update_last_login()

        return JsonResponse({
            'success': True,
            'total_sessions': user.total_sessions,
            'total_time_spent_minutes': user.total_time_spent_minutes,
            'last_login_at': user.last_login_at.isoformat(),
        }, status=200)

    except json.JSONDecodeError:
        return JsonResponse({
            'error': 'Invalid JSON',
            'message': 'Request body must be valid JSON'
        }, status=400)
    except Exception as e:
        return JsonResponse({
            'error': 'Server error',
            'message': str(e)
        }, status=500)


@csrf_exempt
@require_http_methods(["POST"])
@require_authentication
@ratelimit(key='user_or_ip', rate='10/m', method='POST', block=True)
def logout(request):
    """
    Logout user (update last login timestamp).

    POST /api/auth/logout

    Returns:
        200: Success
        401: Not authenticated
    """
    try:
        user = request.firebase_user
        user.update_last_login()

        return JsonResponse({
            'success': True,
            'message': 'Logged out successfully'
        }, status=200)

    except Exception as e:
        return JsonResponse({
            'error': 'Server error',
            'message': str(e)
        }, status=500)


@csrf_exempt
@require_http_methods(["POST"])
@require_authentication
@ratelimit(key='user_or_ip', rate='60/h', method='POST', block=True)
def mark_milestone(request):
    """
    Mark user milestones (first model created, first export).

    POST /api/auth/milestone
    Body: {
        "type": "first_model" | "first_export"
    }

    Returns:
        200: Success
        400: Invalid request
        401: Not authenticated
    """
    try:
        data = json.loads(request.body)
        milestone_type = data.get('type')

        if not milestone_type:
            return JsonResponse({
                'error': 'Milestone type required',
                'message': 'type field is required'
            }, status=400)

        user = request.firebase_user

        if milestone_type == 'first_model':
            user.mark_first_model_created()
        elif milestone_type == 'first_export':
            user.mark_first_export()
        else:
            return JsonResponse({
                'error': 'Invalid milestone type',
                'message': 'type must be "first_model" or "first_export"'
            }, status=400)

        return JsonResponse({
            'success': True,
            'message': f'Milestone {milestone_type} marked'
        }, status=200)

    except json.JSONDecodeError:
        return JsonResponse({
            'error': 'Invalid JSON',
            'message': 'Request body must be valid JSON'
        }, status=400)
    except Exception as e:
        return JsonResponse({
            'error': 'Server error',
            'message': str(e)
        }, status=500)
