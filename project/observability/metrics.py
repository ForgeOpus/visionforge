"""
Centralized metrics definitions for VisionForge.

All metrics are defined here to ensure consistency and prevent duplication.
Follows OpenTelemetry semantic conventions where applicable.
"""

from opentelemetry import metrics
from opentelemetry.metrics import Counter, Histogram


class VisionForgeMetrics:
    """
    Singleton class containing all application metrics.

    Naming convention: <domain>.<action>[.<outcome>]
    Label cardinality: Keep labels low (< 100 unique combinations per metric)
    """

    def __init__(self, meter: metrics.Meter):
        # HTTP Request Metrics
        self.http_request_duration: Histogram = meter.create_histogram(
            name="http.request.duration",
            description="HTTP request duration in seconds",
            unit="s",
        )

        self.http_request_count: Counter = meter.create_counter(
            name="http.request.count",
            description="Total HTTP requests",
            unit="1",
        )

        # Export Metrics
        self.export_request_count: Counter = meter.create_counter(
            name="export.request",
            description="Export requests initiated",
            unit="1",
        )

        self.export_success_count: Counter = meter.create_counter(
            name="export.success",
            description="Successful export completions",
            unit="1",
        )

        self.export_failure_count: Counter = meter.create_counter(
            name="export.failure",
            description="Failed export attempts",
            unit="1",
        )

        self.export_duration: Histogram = meter.create_histogram(
            name="export.duration",
            description="Export processing duration in seconds",
            unit="s",
        )

        # AI Service Metrics (Gemini/Claude)
        self.ai_request_count: Counter = meter.create_counter(
            name="ai.request",
            description="AI service requests",
            unit="1",
        )

        self.ai_request_duration: Histogram = meter.create_histogram(
            name="ai.request.duration",
            description="AI service request duration in seconds",
            unit="s",
        )

        self.ai_error_count: Counter = meter.create_counter(
            name="ai.error",
            description="AI service errors",
            unit="1",
        )

        self.ai_tokens_used: Counter = meter.create_counter(
            name="ai.tokens.used",
            description="AI tokens consumed",
            unit="1",
        )

        # Validation Metrics
        self.validation_request_count: Counter = meter.create_counter(
            name="validation.request",
            description="Architecture validation requests",
            unit="1",
        )

        self.validation_error_count: Counter = meter.create_counter(
            name="validation.error",
            description="Validation errors by type",
            unit="1",
        )

        self.validation_duration: Histogram = meter.create_histogram(
            name="validation.duration",
            description="Validation processing duration in seconds",
            unit="s",
        )

        # Authentication Metrics
        self.auth_attempt_count: Counter = meter.create_counter(
            name="auth.attempt",
            description="Authentication attempts",
            unit="1",
        )

        # Rate Limiting Metrics
        self.rate_limit_hit_count: Counter = meter.create_counter(
            name="rate_limit.hit",
            description="Rate limit violations",
            unit="1",
        )


# Global metrics instance (initialized by middleware)
_metrics: VisionForgeMetrics = None


def initialize_metrics(meter: metrics.Meter) -> VisionForgeMetrics:
    """Initialize the global metrics instance."""
    global _metrics
    _metrics = VisionForgeMetrics(meter)
    return _metrics


def get_metrics() -> VisionForgeMetrics:
    """Get the global metrics instance."""
    if _metrics is None:
        raise RuntimeError("Metrics not initialized. Call initialize_metrics() first.")
    return _metrics
