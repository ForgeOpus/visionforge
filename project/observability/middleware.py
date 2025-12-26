"""
Django middleware for OpenTelemetry metrics collection.

Automatically tracks HTTP request metrics for all endpoints.
Telemetry failures are silent and never block requests.
"""

import time
from typing import Callable

from django.http import HttpRequest, HttpResponse
from django.urls import resolve, Resolver404

from .metrics import get_metrics


class MetricsMiddleware:
    """
    Middleware to collect HTTP request metrics.

    Tracks:
    - Request duration by route and status code
    - Request count by method, route, and status

    Silently degrades if telemetry is unavailable - never blocks requests.
    """

    def __init__(self, get_response: Callable):
        self.get_response = get_response

    def __call__(self, request: HttpRequest) -> HttpResponse:
        start_time = time.time()

        # Get the response
        response = self.get_response(request)

        # Record metrics (silent failure - never blocks)
        try:
            duration = time.time() - start_time
            route = self._get_route_pattern(request)
            metrics = get_metrics()

            labels = {
                "method": request.method,
                "route": route,
                "status": str(response.status_code),
            }

            metrics.http_request_duration.record(duration, labels)
            metrics.http_request_count.add(1, labels)
        except Exception:
            pass  # Silent failure - telemetry unavailable

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
