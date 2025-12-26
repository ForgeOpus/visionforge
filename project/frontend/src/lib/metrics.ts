/**
 * Frontend metrics definitions.
 *
 * All frontend metrics are defined here for consistency.
 * Follows low-cardinality labeling to prevent metric explosion.
 */

import { Counter, Histogram, Meter } from '@opentelemetry/api';

export class FrontendMetrics {
  // Session metrics
  public readonly sessionStart: Counter;
  public readonly sessionDuration: Histogram;

  // Canvas interaction metrics
  public readonly layerAdded: Counter;
  public readonly layerRemoved: Counter;
  public readonly nodeConnected: Counter;
  public readonly parameterEdited: Counter;

  // Export metrics
  public readonly exportButtonClick: Counter;
  public readonly exportSuccess: Counter;
  public readonly exportFailure: Counter;
  public readonly exportDuration: Histogram;

  // AI assistant metrics
  public readonly aiQuerySent: Counter;
  public readonly aiQueryDuration: Histogram;
  public readonly aiQuerySuccess: Counter;
  public readonly aiQueryFailure: Counter;

  // Canvas time tracking
  public readonly canvasTimeSpent: Histogram;

  // API call metrics
  public readonly apiRequest: Counter;
  public readonly apiRequestDuration: Histogram;
  public readonly apiError: Counter;

  // Validation metrics
  public readonly validationTriggered: Counter;
  public readonly validationErrors: Counter;

  constructor(meter: Meter) {
    // Session metrics
    this.sessionStart = meter.createCounter('ui.session.start', {
      description: 'User session started',
      unit: '1',
    });

    this.sessionDuration = meter.createHistogram('ui.session.duration', {
      description: 'Session duration in seconds',
      unit: 's',
    });

    // Canvas interaction metrics
    this.layerAdded = meter.createCounter('ui.layer.added', {
      description: 'Layer added to canvas',
      unit: '1',
    });

    this.layerRemoved = meter.createCounter('ui.layer.removed', {
      description: 'Layer removed from canvas',
      unit: '1',
    });

    this.nodeConnected = meter.createCounter('ui.node.connected', {
      description: 'Nodes connected on canvas',
      unit: '1',
    });

    this.parameterEdited = meter.createCounter('ui.parameter.edited', {
      description: 'Parameter edited (sampled)',
      unit: '1',
    });

    // Export metrics
    this.exportButtonClick = meter.createCounter('ui.export.click', {
      description: 'Export button clicked',
      unit: '1',
    });

    this.exportSuccess = meter.createCounter('ui.export.success', {
      description: 'Export succeeded',
      unit: '1',
    });

    this.exportFailure = meter.createCounter('ui.export.failure', {
      description: 'Export failed',
      unit: '1',
    });

    this.exportDuration = meter.createHistogram('ui.export.duration', {
      description: 'Export operation duration in seconds',
      unit: 's',
    });

    // AI assistant metrics
    this.aiQuerySent = meter.createCounter('ui.ai.query', {
      description: 'AI assistant query sent',
      unit: '1',
    });

    this.aiQueryDuration = meter.createHistogram('ui.ai.query.duration', {
      description: 'AI query duration in seconds',
      unit: 's',
    });

    this.aiQuerySuccess = meter.createCounter('ui.ai.query.success', {
      description: 'AI query succeeded',
      unit: '1',
    });

    this.aiQueryFailure = meter.createCounter('ui.ai.query.failure', {
      description: 'AI query failed',
      unit: '1',
    });

    // Canvas time tracking
    this.canvasTimeSpent = meter.createHistogram('ui.canvas.time_spent', {
      description: 'Time spent on canvas per session in seconds',
      unit: 's',
    });

    // API call metrics
    this.apiRequest = meter.createCounter('ui.api.request', {
      description: 'API request initiated',
      unit: '1',
    });

    this.apiRequestDuration = meter.createHistogram('ui.api.request.duration', {
      description: 'API request duration in seconds',
      unit: 's',
    });

    this.apiError = meter.createCounter('ui.api.error', {
      description: 'API request error',
      unit: '1',
    });

    // Validation metrics
    this.validationTriggered = meter.createCounter('ui.validation.triggered', {
      description: 'Validation triggered by user',
      unit: '1',
    });

    this.validationErrors = meter.createCounter('ui.validation.errors', {
      description: 'Validation errors encountered',
      unit: '1',
    });
  }
}

// Global metrics instance
let _metrics: FrontendMetrics | null = null;

export function initializeMetrics(meter: Meter): FrontendMetrics {
  _metrics = new FrontendMetrics(meter);
  return _metrics;
}

export function getMetrics(): FrontendMetrics {
  if (!_metrics) {
    throw new Error('Metrics not initialized. Call initializeMetrics() first.');
  }
  return _metrics;
}
