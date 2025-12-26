"""
Django middleware for OpenTelemetry metrics collection.

Automatically tracks HTTP request metrics for all endpoints.
"""

import time
import logging
from typing import Callable

from django.http import HttpRequest, HttpResponse
from django.urls import resolve, Resolver404

from .metrics import get_metrics


logger = logging.getLogger(__name__)


class MetricsMiddleware:
    """
    Middleware to collect HTTP request metrics.

    Tracks:
    - Request duration by route and status code
    - Request count by method, route, and status
    """

    def __init__(self, get_response: Callable):
        self.get_response = get_response

    def __call__(self, request: HttpRequest) -> HttpResponse:
        start_time = time.time()

        # Get the response
        response = self.get_response(request)

        # Calculate duration
        duration = time.time() - start_time

        # Extract route pattern
        route = self._get_route_pattern(request)

        # Record metrics
        try:
            metrics = get_metrics()

            # Labels for metrics (low cardinality)
            labels = {
                "method": request.method,
                "route": route,
                "status": str(response.status_code),
            }

            # Record duration histogram
            metrics.http_request_duration.record(duration, labels)

            # Increment request counter
            metrics.http_request_count.add(1, labels)

        except Exception as e:
            logger.error(f"Failed to record HTTP metrics: {e}")

        return response

    def _get_route_pattern(self, request: HttpRequest) -> str:
        """
        Extract route pattern for low-cardinality labeling.

        Returns parameterized route like '/api/v1/project/<id>'
        instead of '/api/v1/project/123' to prevent cardinality explosion.
        """
        try:
            match = resolve(request.path)
            # Return the route pattern with placeholders
            if match.route:
                return f"/{match.route}"
            # Fallback to view name if route not available
            return match.view_name or request.path
        except Resolver404:
            # For unmatched routes, use path with query params removed
            return request.path.split('?')[0] if '?' in request.path else request.path
        except Exception:
            return "unknown"
