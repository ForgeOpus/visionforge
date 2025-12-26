"""
Django app configuration for observability.

Initializes OpenTelemetry on application startup.
"""

from django.apps import AppConfig
from django.conf import settings


class ObservabilityConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'observability'

    def ready(self):
        """Initialize OpenTelemetry when Django starts."""
        from .telemetry import initialize_telemetry, get_meter
        from .metrics import initialize_metrics

        # Initialize telemetry with settings from Django config
        initialize_telemetry(
            service_name=settings.OTEL_SERVICE_NAME,
            service_version=settings.OTEL_SERVICE_VERSION,
            enable_prometheus=settings.OTEL_ENABLE_PROMETHEUS,
            enable_otlp=settings.OTEL_ENABLE_OTLP,
            otlp_endpoint=settings.OTEL_EXPORTER_OTLP_ENDPOINT,
        )

        # Initialize metrics
        meter = get_meter()
        initialize_metrics(meter)
