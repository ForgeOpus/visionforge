"""
OpenTelemetry initialization and configuration.

This module provides centralized telemetry setup for metrics and traces.
Supports both Prometheus and OTLP exporters.
"""

import os
import logging
from typing import Optional

from opentelemetry import metrics, trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter


logger = logging.getLogger(__name__)

# Global telemetry instances
_meter_provider: Optional[MeterProvider] = None
_tracer_provider: Optional[TracerProvider] = None
_meter: Optional[metrics.Meter] = None
_tracer: Optional[trace.Tracer] = None


def initialize_telemetry(
    service_name: str = "visionforge-backend",
    service_version: str = "1.0.0",
    enable_prometheus: bool = True,
    enable_otlp: bool = False,
    otlp_endpoint: Optional[str] = None,
) -> None:
    """
    Initialize OpenTelemetry with configured exporters.

    Args:
        service_name: Name of the service for resource attribution
        service_version: Version of the service
        enable_prometheus: Enable Prometheus exporter (default: True)
        enable_otlp: Enable OTLP exporter (default: False)
        otlp_endpoint: OTLP collector endpoint (e.g., 'localhost:4317')
    """
    global _meter_provider, _tracer_provider, _meter, _tracer

    # Create resource attributes
    resource = Resource.create({
        SERVICE_NAME: service_name,
        SERVICE_VERSION: service_version,
        "deployment.environment": os.getenv("ENVIRONMENT", "dev"),
    })

    # Initialize metrics
    metric_readers = []

    if enable_prometheus:
        try:
            prometheus_reader = PrometheusMetricReader()
            metric_readers.append(prometheus_reader)
            logger.info("Prometheus metrics exporter initialized on port 9090")
        except Exception as e:
            logger.error(f"Failed to initialize Prometheus exporter: {e}")

    if enable_otlp:
        endpoint = otlp_endpoint or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "localhost:4317")
        try:
            otlp_exporter = OTLPMetricExporter(endpoint=endpoint, insecure=True)
            otlp_reader = PeriodicExportingMetricReader(otlp_exporter, export_interval_millis=60000)
            metric_readers.append(otlp_reader)
            logger.info(f"OTLP metrics exporter initialized: {endpoint}")
        except Exception as e:
            logger.error(f"Failed to initialize OTLP metrics exporter: {e}")

    if metric_readers:
        _meter_provider = MeterProvider(
            resource=resource,
            metric_readers=metric_readers,
        )
        metrics.set_meter_provider(_meter_provider)
        _meter = _meter_provider.get_meter(service_name, service_version)
        logger.info(f"Metrics initialized for {service_name}")
    else:
        logger.warning("No metric exporters configured")

    # Initialize tracing
    if enable_otlp:
        endpoint = otlp_endpoint or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "localhost:4317")
        try:
            otlp_trace_exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
            _tracer_provider = TracerProvider(resource=resource)
            _tracer_provider.add_span_processor(BatchSpanProcessor(otlp_trace_exporter))
            trace.set_tracer_provider(_tracer_provider)
            _tracer = _tracer_provider.get_tracer(service_name, service_version)
            logger.info(f"Tracing initialized for {service_name}")
        except Exception as e:
            logger.error(f"Failed to initialize OTLP trace exporter: {e}")


def get_meter() -> metrics.Meter:
    """Get the global meter instance."""
    if _meter is None:
        raise RuntimeError("Telemetry not initialized. Call initialize_telemetry() first.")
    return _meter


def get_tracer() -> trace.Tracer:
    """Get the global tracer instance."""
    if _tracer is None:
        raise RuntimeError("Telemetry not initialized. Call initialize_telemetry() first.")
    return _tracer
