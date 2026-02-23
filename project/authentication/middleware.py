"""
Firebase authentication middleware for Django.
"""
import logging
from django.utils.deprecation import MiddlewareMixin
from django.http import JsonResponse
from .firebase_auth import verify_firebase_token, get_user_info_from_token
from .models import User

logger = logging.getLogger('authentication')


class FirebaseAuthenticationMiddleware(MiddlewareMixin):
    """
    Middleware to verify Firebase ID tokens and attach user to request.

    Looks for Firebase token in:
    1. Authorization header (Bearer token)
    2. X-Firebase-Token header

    If token is valid, attaches the user object to request.firebase_user
    """

    def process_request(self, request):
        """
        Process incoming request and verify Firebase token if present.
        """
        # Skip authentication for certain paths
        exempt_paths = [
            '/admin/',
            '/api/auth/verify-token',
            '/api/auth/csrf',
            '/api/v1/shared/',
        ]

        if any(request.path.startswith(path) for path in exempt_paths):
            return None

        # Extract token from headers
        token = None
        auth_header = request.META.get('HTTP_AUTHORIZATION', '')
        firebase_token_header = request.META.get('HTTP_X_FIREBASE_TOKEN', '')

        if auth_header.startswith('Bearer '):
            token = auth_header.split('Bearer ')[1]
        elif firebase_token_header:
            token = firebase_token_header

        # If no token, continue without authentication (guest access)
        if not token:
            request.firebase_user = None
            return None

        # Verify token
        try:
            decoded_token = verify_firebase_token(token)
            if decoded_token:
                user_info = get_user_info_from_token(decoded_token)
                if user_info and user_info.get('firebase_uid'):
                    # Try to get user from database
                    try:
                        user = User.objects.get(firebase_uid=user_info['firebase_uid'])
                        request.firebase_user = user
                    except User.DoesNotExist:
                        # User not in database yet, attach user info for creation
                        request.firebase_user = None
                        request.firebase_user_info = user_info
            else:
                request.firebase_user = None
        except Exception as e:
            logger.error(f"Firebase authentication error: {str(e)}", exc_info=True)
            request.firebase_user = None

        return None


def require_authentication(view_func):
    """
    Decorator to require Firebase authentication for a view.

    Usage:
        @require_authentication
        def my_view(request):
            user = request.firebase_user
            ...
    """
    def wrapper(request, *args, **kwargs):
        if not hasattr(request, 'firebase_user') or request.firebase_user is None:
            logger.warning(f"Unauthorized access attempt to {request.path} from IP {request.META.get('REMOTE_ADDR')}")
            return JsonResponse({
                'error': 'Authentication required',
                'message': 'You must be logged in to access this endpoint'
            }, status=401)
        return view_func(request, *args, **kwargs)
    return wrapper
