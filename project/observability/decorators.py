"""
Decorators for instrumenting functions with OpenTelemetry metrics.
"""

import time
import logging
from functools import wraps
from typing import Callable, Any

from .metrics import get_metrics


logger = logging.getLogger(__name__)


def track_export(func: Callable) -> Callable:
    """
    Decorator to track export operations.

    Records:
    - export.request (counter)
    - export.success (counter)
    - export.failure (counter with error_type label)
    - export.duration (histogram)
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        metrics = get_metrics()
        start_time = time.time()

        # Extract format from request if available
        request = args[0] if args else None
        export_format = "unknown"
        if request and hasattr(request, 'data'):
            export_format = request.data.get('format', 'unknown')

        # Track request
        metrics.export_request_count.add(1, {"format": export_format})

        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time

            # Track success
            metrics.export_success_count.add(1, {"format": export_format})
            metrics.export_duration.record(duration, {"format": export_format, "status": "success"})

            return result

        except Exception as e:
            duration = time.time() - start_time

            # Track failure with error type
            error_type = type(e).__name__
            metrics.export_failure_count.add(1, {
                "format": export_format,
                "error_type": error_type,
            })
            metrics.export_duration.record(duration, {"format": export_format, "status": "failure"})

            # Re-raise the exception
            raise

    return wrapper


def track_validation(func: Callable) -> Callable:
    """
    Decorator to track validation operations.

    Records:
    - validation.request (counter)
    - validation.error (counter with error_code label)
    - validation.duration (histogram)
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        metrics = get_metrics()
        start_time = time.time()

        # Track request
        metrics.validation_request_count.add(1, {})

        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time

            # Track validation errors from result
            # Handle both dict results and Response objects
            result_data = result
            if hasattr(result, 'data'):
                result_data = result.data

            if isinstance(result_data, dict):
                errors = result_data.get('errors', [])
                for error in errors:
                    error_code = error.get('type', 'unknown')
                    metrics.validation_error_count.add(1, {"error_code": error_code})

                has_errors = len(errors) > 0
                metrics.validation_duration.record(duration, {"has_errors": str(has_errors).lower()})
            else:
                metrics.validation_duration.record(duration, {"has_errors": "unknown"})

            return result

        except Exception as e:
            duration = time.time() - start_time
            metrics.validation_duration.record(duration, {"has_errors": "true"})
            raise

    return wrapper


def track_ai_request(provider: str, operation: str):
    """
    Decorator factory to track AI service requests.

    Args:
        provider: AI provider name (gemini, claude)
        operation: Operation type (chat, file_upload, suggestions)

    Records:
    - ai.request (counter)
    - ai.request.duration (histogram)
    - ai.error (counter with error_class label)
    - ai.tokens.used (counter, if available)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            metrics = get_metrics()
            start_time = time.time()

            labels = {
                "provider": provider,
                "operation": operation,
            }

            # Track request
            metrics.ai_request_count.add(1, labels)

            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time

                # Track duration
                metrics.ai_request_duration.record(duration, {**labels, "status": "success"})

                # Track token usage if available in result
                if isinstance(result, dict) and 'usage_metadata' in result:
                    usage = result['usage_metadata']
                    total_tokens = usage.get('total_token_count', 0)
                    if total_tokens > 0:
                        metrics.ai_tokens_used.add(total_tokens, labels)

                return result

            except Exception as e:
                duration = time.time() - start_time

                # Classify error
                error_class = _classify_ai_error(e)

                # Track error
                metrics.ai_error_count.add(1, {**labels, "error_class": error_class})
                metrics.ai_request_duration.record(duration, {**labels, "status": "error"})

                raise

        return wrapper
    return decorator


def _classify_ai_error(error: Exception) -> str:
    """
    Classify AI errors into stable categories to avoid cardinality explosion.

    Returns one of: rate_limit, auth, timeout, network, api_error, unknown
    """
    error_str = str(error).lower()
    error_type = type(error).__name__.lower()

    # Rate limiting
    if 'rate' in error_str or 'quota' in error_str or '429' in error_str:
        return 'rate_limit'

    # Authentication
    if 'auth' in error_str or 'api key' in error_str or '401' in error_str or '403' in error_str:
        return 'auth'

    # Timeout
    if 'timeout' in error_str or 'timed out' in error_str:
        return 'timeout'

    # Network
    if 'connection' in error_str or 'network' in error_str:
        return 'network'

    # API errors
    if '400' in error_str or '500' in error_str or 'api' in error_type:
        return 'api_error'

    return 'unknown'
