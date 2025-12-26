"""
OpenTelemetry observability initialization for VisionForge.
"""

default_app_config = 'observability.apps.ObservabilityConfig'

from .telemetry import initialize_telemetry, get_meter, get_tracer

__all__ = ['initialize_telemetry', 'get_meter', 'get_tracer']
