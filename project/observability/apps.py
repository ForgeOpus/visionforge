"""
Django app configuration for observability.

Initializes OpenTelemetry on application startup.
Failures are silent and never prevent Django from starting.
"""

import logging
from django.apps import AppConfig
from django.conf import settings


logger = logging.getLogger(__name__)


class ObservabilityConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'observability'

    def ready(self):
        """
        Initialize OpenTelemetry when Django starts.

        Silently fails if initialization errors occur - never blocks Django startup.
        """
        try:
            from .telemetry import initialize_telemetry, get_meter
            from .metrics import initialize_metrics

            # Initialize telemetry with settings from Django config
            initialize_telemetry(
                service_name=getattr(settings, 'OTEL_SERVICE_NAME', 'visionforge-backend'),
                service_version=getattr(settings, 'OTEL_SERVICE_VERSION', '1.0.0'),
                enable_prometheus=getattr(settings, 'OTEL_ENABLE_PROMETHEUS', True),
                enable_otlp=getattr(settings, 'OTEL_ENABLE_OTLP', False),
                otlp_endpoint=getattr(settings, 'OTEL_EXPORTER_OTLP_ENDPOINT', None),
            )

            # Initialize metrics
            meter = get_meter()
            initialize_metrics(meter)
        except Exception:
            # Silent failure - telemetry unavailable but Django continues
            pass
