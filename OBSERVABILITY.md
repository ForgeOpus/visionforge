# VisionForge Observability Guide

## Overview

VisionForge uses **OpenTelemetry** for comprehensive observability, tracking metrics and traces across both backend (Django) and frontend (React). All metrics are designed to be **low-cardinality**, **privacy-preserving**, and **actionable**.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     VisionForge App                         │
├─────────────────────┬───────────────────────────────────────┤
│  Backend (Django)   │      Frontend (React)                 │
│  - HTTP Metrics     │      - User Interaction Metrics       │
│  - Export Metrics   │      - Canvas Activity                │
│  - AI API Metrics   │      - API Call Metrics               │
│  - Validation       │      - Session Tracking               │
└──────────┬──────────┴──────────────┬────────────────────────┘
           │                         │
           │ Prometheus Exporter     │ OTLP HTTP Exporter
           │ (port 9090)             │
           │                         │
           ▼                         ▼
    ┌────────────┐          ┌────────────────┐
    │ Prometheus │          │ OTLP Collector │
    │  Server    │          │ (optional)     │
    └────────────┘          └────────────────┘
```

## Backend Metrics

### 1. HTTP Request Metrics

**Purpose:** Track all HTTP requests for performance and usage patterns.

**Metrics:**
- `http.request.duration` (histogram) - Request duration in seconds
  - Labels: `method`, `route`, `status`
- `http.request.count` (counter) - Total requests
  - Labels: `method`, `route`, `status`

**Example:**
```
http.request.duration{method="POST",route="/api/v1/export",status="200"} = 2.5s
http.request.count{method="POST",route="/api/v1/export",status="200"} = 42
```

### 2. Export Metrics

**Purpose:** Track model export operations, success rates, and performance.

**Metrics:**
- `export.request` (counter) - Export requests initiated
  - Labels: `format` (pytorch|tensorflow)
- `export.success` (counter) - Successful exports
  - Labels: `format`
- `export.failure` (counter) - Failed exports
  - Labels: `format`, `error_type`
- `export.duration` (histogram) - Export processing time
  - Labels: `format`, `status` (success|failure)

**Example:**
```
export.success{format="pytorch"} = 128
export.duration{format="pytorch",status="success"} = 1.8s
export.failure{format="tensorflow",error_type="validation_error"} = 3
```

### 3. AI Service Metrics (Gemini/Claude)

**Purpose:** Monitor AI service performance, costs, and errors.

**Metrics:**
- `ai.request` (counter) - AI service requests
  - Labels: `provider` (gemini|claude), `operation` (chat|file_upload|suggestions)
- `ai.request.duration` (histogram) - Request latency
  - Labels: `provider`, `operation`, `status`
- `ai.error` (counter) - AI service errors
  - Labels: `provider`, `operation`, `error_class`
- `ai.tokens.used` (counter) - Token consumption
  - Labels: `provider`, `operation`

**Error Classification:**
- `rate_limit` - Rate limiting errors
- `auth` - Authentication failures
- `timeout` - Request timeouts
- `network` - Network errors
- `api_error` - API errors (4xx, 5xx)
- `unknown` - Unclassified errors

**Example:**
```
ai.request{provider="gemini",operation="chat"} = 456
ai.request.duration{provider="gemini",operation="chat",status="success"} = 0.8s
ai.tokens.used{provider="gemini",operation="chat"} = 12345
ai.error{provider="gemini",operation="chat",error_class="rate_limit"} = 2
```

### 4. Validation Metrics

**Purpose:** Track validation usage and error patterns.

**Metrics:**
- `validation.request` (counter) - Validation requests
- `validation.error` (counter) - Validation errors by type
  - Labels: `error_code`
- `validation.duration` (histogram) - Validation time
  - Labels: `has_errors` (true|false)

## Frontend Metrics

### 1. Session Metrics

**Purpose:** Understand user engagement and session duration.

**Metrics:**
- `ui.session.start` (counter) - Sessions initiated
- `ui.session.duration` (histogram) - Session duration in seconds

### 2. Canvas Interaction Metrics

**Purpose:** Track how users build architectures.

**Metrics:**
- `ui.layer.added` (counter) - Layers added to canvas
  - Labels: `layer_type` (conv2d, linear, relu, etc.)
- `ui.layer.removed` (counter) - Layers removed
  - Labels: `layer_type`
- `ui.node.connected` (counter) - Node connections made
  - Labels: `from_type`, `to_type`
- `ui.parameter.edited` (counter) - Parameter edits (sampled at 10%)
  - Labels: `node_type`, `parameter`
- `ui.canvas.time_spent` (histogram) - Time spent on canvas per session

**Example:**
```
ui.layer.added{layer_type="conv2d"} = 234
ui.node.connected{from_type="conv2d",to_type="relu"} = 189
```

### 3. Export Metrics

**Purpose:** Track export button clicks and outcomes.

**Metrics:**
- `ui.export.click` (counter) - Export button clicked
  - Labels: `format`
- `ui.export.success` (counter) - Export succeeded
  - Labels: `format`
- `ui.export.failure` (counter) - Export failed
  - Labels: `format`, `error_type`
- `ui.export.duration` (histogram) - Export operation duration

### 4. AI Assistant Metrics

**Purpose:** Track AI assistant usage and performance.

**Metrics:**
- `ui.ai.query` (counter) - AI queries sent
- `ui.ai.query.duration` (histogram) - Query latency
  - Labels: `status`
- `ui.ai.query.success` (counter) - Successful queries
- `ui.ai.query.failure` (counter) - Failed queries
  - Labels: `error_type`

### 5. API Call Metrics

**Purpose:** Monitor frontend API requests.

**Metrics:**
- `ui.api.request` (counter) - API requests initiated
  - Labels: `endpoint`, `method`
- `ui.api.request.duration` (histogram) - Request duration
  - Labels: `endpoint`, `method`
- `ui.api.error` (counter) - API errors
  - Labels: `endpoint`, `method`, `error_code`

## Configuration

### Backend Configuration

Set environment variables:

```bash
# Service identification
export OTEL_SERVICE_NAME="visionforge-backend"
export OTEL_SERVICE_VERSION="1.0.0"

# Enable Prometheus exporter (default: True)
export OTEL_ENABLE_PROMETHEUS="True"

# Enable OTLP exporter (default: False)
export OTEL_ENABLE_OTLP="False"

# OTLP collector endpoint (if OTLP enabled)
export OTEL_EXPORTER_OTLP_ENDPOINT="localhost:4317"
```

Metrics are automatically initialized on Django startup via the `ObservabilityConfig` app.

### Frontend Configuration

Set environment variable:

```bash
# OTLP HTTP endpoint (default: http://localhost:4318/v1/metrics)
export VITE_OTEL_ENDPOINT="http://localhost:4318/v1/metrics"
```

Metrics are automatically initialized in `main.tsx` on app load.

## Viewing Metrics

### Option 1: Prometheus (Backend Only)

The backend exposes a Prometheus endpoint on port 9090 by default.

1. **Add to Prometheus config** (`prometheus.yml`):
```yaml
scrape_configs:
  - job_name: 'visionforge-backend'
    static_configs:
      - targets: ['localhost:9090']
    scrape_interval: 15s
```

2. **View in Prometheus UI**: `http://localhost:9090`

3. **Example queries**:
```promql
# Request rate
rate(http_request_count[5m])

# Export success rate
rate(export_success[5m]) / rate(export_request[5m])

# AI request latency (p95)
histogram_quantile(0.95, ai_request_duration_bucket)
```

### Option 2: OTLP Collector (Both Frontend and Backend)

Use an OpenTelemetry Collector to aggregate metrics from both services.

**Collector config** (`otel-collector-config.yaml`):
```yaml
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317
      http:
        endpoint: 0.0.0.0:4318

processors:
  batch:
    timeout: 10s

exporters:
  prometheus:
    endpoint: "0.0.0.0:8889"
  logging:
    loglevel: debug

service:
  pipelines:
    metrics:
      receivers: [otlp]
      processors: [batch]
      exporters: [prometheus, logging]
```

**Run collector**:
```bash
docker run -p 4317:4317 -p 4318:4318 -p 8889:8889 \
  -v $(pwd)/otel-collector-config.yaml:/etc/otel-collector-config.yaml \
  otel/opentelemetry-collector:latest \
  --config=/etc/otel-collector-config.yaml
```

**Enable OTLP in backend**:
```bash
export OTEL_ENABLE_OTLP="True"
export OTEL_EXPORTER_OTLP_ENDPOINT="localhost:4317"
```

**Configure frontend**:
```bash
export VITE_OTEL_ENDPOINT="http://localhost:4318/v1/metrics"
```

## Privacy and Cardinality

### Privacy Measures

1. **No PII Collection**: No user identifiers, email addresses, or personally identifiable information is collected in metrics.
2. **Sampled Events**: Parameter edits are sampled at 10% to reduce cardinality and data volume.
3. **Stable Labels**: Only pre-defined, low-cardinality labels are used (no user input).

### Cardinality Management

**Low-cardinality labels** (safe):
- `layer_type` - Fixed set of node types (~30 values)
- `format` - pytorch|tensorflow (2 values)
- `error_type` - Classified into ~10 stable categories
- `provider` - gemini|claude (2 values)
- `operation` - chat|file_upload|suggestions (~3-5 values)

**Avoided high-cardinality labels**:
- ❌ User IDs
- ❌ Project names
- ❌ Timestamps
- ❌ Dynamic error messages
- ❌ Freeform text

## Querying Examples

### Backend Queries

**Export success rate**:
```promql
rate(export_success{format="pytorch"}[5m]) / rate(export_request{format="pytorch"}[5m])
```

**AI request latency by provider**:
```promql
histogram_quantile(0.95, rate(ai_request_duration_bucket{provider="gemini"}[5m]))
```

**Validation error breakdown**:
```promql
sum by (error_code) (validation_error)
```

### Frontend Queries

**Most popular layers**:
```promql
topk(10, sum by (layer_type) (ui_layer_added))
```

**Session duration distribution**:
```promql
histogram_quantile(0.5, ui_session_duration_bucket)
```

**API error rate**:
```promql
rate(ui_api_error[5m]) / rate(ui_api_request[5m])
```

## Alerts (Recommended)

Set up alerts for critical issues:

```yaml
groups:
  - name: visionforge_alerts
    rules:
      # Export failure rate > 10%
      - alert: HighExportFailureRate
        expr: |
          rate(export_failure[5m]) / rate(export_request[5m]) > 0.1
        for: 5m
        annotations:
          summary: "High export failure rate"

      # AI service errors
      - alert: AIServiceErrors
        expr: |
          rate(ai_error{error_class="rate_limit"}[5m]) > 0.1
        for: 2m
        annotations:
          summary: "AI service rate limiting detected"

      # High API latency
      - alert: SlowAPIRequests
        expr: |
          histogram_quantile(0.95, rate(http_request_duration_bucket[5m])) > 5
        for: 5m
        annotations:
          summary: "95th percentile request latency > 5s"
```

## Troubleshooting

### Backend metrics not appearing

1. Check Django logs for initialization errors
2. Verify Prometheus endpoint: `curl http://localhost:9090/metrics`
3. Ensure `observability` app is in `INSTALLED_APPS`
4. Check middleware is loaded: `observability.middleware.MetricsMiddleware`

### Frontend metrics not appearing

1. Check browser console for telemetry errors
2. Verify OTLP endpoint is accessible
3. Check collector logs for incoming data
4. Ensure metrics are initialized before first user action

### High cardinality warnings

If you see warnings about high cardinality:
1. Review label values - they should be from a fixed set
2. Check for dynamic labels (error messages, IDs)
3. Increase sampling rate or remove problematic labels

## Production Checklist

- [ ] Environment variables configured
- [ ] Prometheus scraping backend endpoint
- [ ] OTLP collector running (if using collector)
- [ ] Frontend OTLP endpoint accessible
- [ ] Alerts configured
- [ ] Grafana dashboards created
- [ ] Retention policies set (Prometheus)
- [ ] High-cardinality labels reviewed

## Cost Optimization

1. **Sampling**: Parameter edits are sampled at 10%. Adjust in `storeInstrumentation.ts` if needed.
2. **Export Interval**: Frontend exports every 60s. Adjust in `telemetry.ts`.
3. **Retention**: Configure Prometheus retention based on data volume:
   ```bash
   --storage.tsdb.retention.time=30d
   ```

## Next Steps

1. **Create Grafana dashboards** for visualization
2. **Set up alerts** for critical metrics
3. **Analyze metrics** to identify friction points
4. **Optimize based on data** (e.g., slow endpoints, high-error operations)

## Support

For issues or questions about observability:
- Check logs: `docker logs <container>` or Django console
- Review metric definitions in code
- Verify label cardinality: `curl -s http://localhost:9090/api/v1/label/__name__/values`
