/**
 * API instrumentation helpers for tracking API calls.
 */

import { getMetrics } from './metrics';

/**
 * Track export button click.
 */
export function trackExportClick(format: string): void {
  try {
    const metrics = getMetrics();
    metrics.exportButtonClick.add(1, { format });
  } catch (error) {
    console.debug('Failed to track export click:', error);
  }
}

/**
 * Track export success.
 */
export function trackExportSuccess(format: string, durationSeconds: number): void {
  try {
    const metrics = getMetrics();
    metrics.exportSuccess.add(1, { format });
    metrics.exportDuration.record(durationSeconds, { format, status: 'success' });
  } catch (error) {
    console.debug('Failed to track export success:', error);
  }
}

/**
 * Track export failure.
 */
export function trackExportFailure(format: string, durationSeconds: number, errorType: string): void {
  try {
    const metrics = getMetrics();
    metrics.exportFailure.add(1, { format, error_type: errorType });
    metrics.exportDuration.record(durationSeconds, { format, status: 'failure' });
  } catch (error) {
    console.debug('Failed to track export failure:', error);
  }
}

/**
 * Track AI query sent.
 */
export function trackAIQuerySent(): void {
  try {
    const metrics = getMetrics();
    metrics.aiQuerySent.add(1, {});
  } catch (error) {
    console.debug('Failed to track AI query:', error);
  }
}

/**
 * Track AI query success.
 */
export function trackAIQuerySuccess(durationSeconds: number): void {
  try {
    const metrics = getMetrics();
    metrics.aiQuerySuccess.add(1, {});
    metrics.aiQueryDuration.record(durationSeconds, { status: 'success' });
  } catch (error) {
    console.debug('Failed to track AI query success:', error);
  }
}

/**
 * Track AI query failure.
 */
export function trackAIQueryFailure(durationSeconds: number, errorType: string): void {
  try {
    const metrics = getMetrics();
    metrics.aiQueryFailure.add(1, { error_type: errorType });
    metrics.aiQueryDuration.record(durationSeconds, { status: 'failure' });
  } catch (error) {
    console.debug('Failed to track AI query failure:', error);
  }
}

/**
 * Track generic API request.
 */
export function trackAPIRequest(endpoint: string, method: string): void {
  try {
    const metrics = getMetrics();
    metrics.apiRequest.add(1, { endpoint, method });
  } catch (error) {
    console.debug('Failed to track API request:', error);
  }
}

/**
 * Track API request duration.
 */
export function trackAPIRequestDuration(endpoint: string, method: string, durationSeconds: number): void {
  try {
    const metrics = getMetrics();
    metrics.apiRequestDuration.record(durationSeconds, { endpoint, method });
  } catch (error) {
    console.debug('Failed to track API request duration:', error);
  }
}

/**
 * Track API error.
 */
export function trackAPIError(endpoint: string, method: string, errorCode: string): void {
  try {
    const metrics = getMetrics();
    metrics.apiError.add(1, { endpoint, method, error_code: errorCode });
  } catch (error) {
    console.debug('Failed to track API error:', error);
  }
}

/**
 * Classify error into stable categories.
 */
export function classifyError(error: any): string {
  const errorStr = String(error).toLowerCase();

  if (errorStr.includes('network') || errorStr.includes('failed to fetch')) {
    return 'network';
  }
  if (errorStr.includes('timeout')) {
    return 'timeout';
  }
  if (errorStr.includes('401') || errorStr.includes('unauthorized')) {
    return 'auth';
  }
  if (errorStr.includes('400') || errorStr.includes('bad request')) {
    return 'bad_request';
  }
  if (errorStr.includes('404')) {
    return 'not_found';
  }
  if (errorStr.includes('500') || errorStr.includes('server error')) {
    return 'server_error';
  }

  return 'unknown';
}
