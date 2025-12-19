"""
Standardized API response utilities
"""
from typing import Dict, Any, Optional, List
from django.http import JsonResponse
from rest_framework.response import Response
from rest_framework import status


def success_response(
    data: Any = None,
    message: str = "Success",
    status_code: int = 200
) -> JsonResponse:
    """
    Create a standardized success response

    Args:
        data: Response data
        message: Success message
        status_code: HTTP status code

    Returns:
        JsonResponse with standardized format
    """
    return JsonResponse({
        'success': True,
        'message': message,
        'data': data,
    }, status=status_code)


def error_response(
    error: str,
    message: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    status_code: int = 400
) -> JsonResponse:
    """
    Create a standardized error response

    Args:
        error: Error type/code
        message: Human-readable error message
        details: Additional error details
        status_code: HTTP status code

    Returns:
        JsonResponse with standardized format
    """
    response_data = {
        'success': False,
        'error': error,
    }

    if message:
        response_data['message'] = message

    if details:
        response_data['details'] = details

    return JsonResponse(response_data, status=status_code)


def validation_error_response(
    errors: List[Dict[str, Any]],
    message: str = "Validation failed"
) -> JsonResponse:
    """
    Create a standardized validation error response

    Args:
        errors: List of validation errors
        message: Error message

    Returns:
        JsonResponse with validation errors
    """
    return JsonResponse({
        'success': False,
        'error': 'validation_error',
        'message': message,
        'validationErrors': errors,
        'errorCount': len(errors)
    }, status=400)


def not_found_response(
    resource: str = "Resource",
    message: Optional[str] = None
) -> JsonResponse:
    """
    Create a standardized 404 response

    Args:
        resource: Name of the resource that wasn't found
        message: Custom error message

    Returns:
        JsonResponse with 404 status
    """
    return error_response(
        error='not_found',
        message=message or f"{resource} not found",
        status_code=404
    )


def unauthorized_response(
    message: str = "Authentication required"
) -> JsonResponse:
    """
    Create a standardized 401 response

    Args:
        message: Error message

    Returns:
        JsonResponse with 401 status
    """
    return error_response(
        error='unauthorized',
        message=message,
        status_code=401
    )


def forbidden_response(
    message: str = "Access forbidden"
) -> JsonResponse:
    """
    Create a standardized 403 response

    Args:
        message: Error message

    Returns:
        JsonResponse with 403 status
    """
    return error_response(
        error='forbidden',
        message=message,
        status_code=403
    )


def server_error_response(
    error_details: Optional[str] = None,
    include_traceback: bool = False
) -> JsonResponse:
    """
    Create a standardized 500 response

    Args:
        error_details: Error details to include
        include_traceback: Whether to include traceback (dev only)

    Returns:
        JsonResponse with 500 status
    """
    response_data = {
        'error': 'server_error',
        'message': 'An internal server error occurred',
    }

    if error_details:
        response_data['details'] = error_details

    if include_traceback:
        import traceback
        response_data['traceback'] = traceback.format_exc()

    return error_response(
        error='server_error',
        message='An internal server error occurred',
        details={'error': error_details} if error_details else None,
        status_code=500
    )
