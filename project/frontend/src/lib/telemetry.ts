/**
 * OpenTelemetry initialization for VisionForge frontend.
 *
 * Provides metrics collection for user interactions and application behavior.
 * Exports metrics to OTLP HTTP endpoint (backend or collector).
 * Failures are silent and never visible to users.
 */

import { MeterProvider, PeriodicExportingMetricReader } from '@opentelemetry/sdk-metrics';
import { OTLPMetricExporter } from '@opentelemetry/exporter-metrics-otlp-http';
import { Resource } from '@opentelemetry/resources';
import { ATTR_SERVICE_NAME, ATTR_SERVICE_VERSION } from '@opentelemetry/semantic-conventions';
import { metrics, Meter } from '@opentelemetry/api';

let meterProvider: MeterProvider | null = null;
let meter: Meter | null = null;

/**
 * Initialize OpenTelemetry metrics for the frontend.
 *
 * @param endpoint - OTLP HTTP endpoint (e.g., 'http://localhost:4318/v1/metrics')
 *
 * Silent failure - never throws or logs to console.
 */
export function initializeTelemetry(
  endpoint: string = import.meta.env.VITE_OTEL_ENDPOINT || 'http://localhost:4318/v1/metrics'
): void {
  if (meterProvider) {
    return; // Already initialized
  }

  try {
    // Create resource with service information
    const resource = Resource.default().merge(
      new Resource({
        [ATTR_SERVICE_NAME]: 'visionforge-frontend',
        [ATTR_SERVICE_VERSION]: '1.0.0',
        'deployment.environment': import.meta.env.MODE || 'development',
      })
    );

    // Create OTLP exporter
    const exporter = new OTLPMetricExporter({
      url: endpoint,
      headers: {},
      concurrencyLimit: 1,
    });

    // Create metric reader with 60s export interval
    const metricReader = new PeriodicExportingMetricReader({
      exporter,
      exportIntervalMillis: 60000, // Export every 60 seconds
    });

    // Create and set meter provider
    meterProvider = new MeterProvider({
      resource,
      readers: [metricReader],
    });

    metrics.setGlobalMeterProvider(meterProvider);
    meter = metrics.getMeter('visionforge-frontend', '1.0.0');
  } catch (error) {
    // Silent failure - telemetry unavailable
  }
}

/**
 * Get the global meter instance.
 *
 * @throws Error if telemetry not initialized (caught by callers)
 */
export function getMeter(): Meter {
  if (!meter) {
    throw new Error('Telemetry not initialized');
  }
  return meter;
}

/**
 * Shutdown telemetry (call on app unmount/cleanup).
 */
export async function shutdownTelemetry(): Promise<void> {
  if (meterProvider) {
    await meterProvider.shutdown();
    meterProvider = null;
    meter = null;
  }
}
