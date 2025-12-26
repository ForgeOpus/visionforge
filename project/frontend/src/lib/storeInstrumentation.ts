/**
 * Store instrumentation helpers for tracking user interactions.
 *
 * Provides functions to record metrics for Zustand store actions.
 */

import { getMetrics } from './metrics';
import { BlockType } from './types';

/**
 * Track when a layer/node is added to the canvas.
 */
export function trackLayerAdded(layerType: BlockType): void {
  try {
    const metrics = getMetrics();
    metrics.layerAdded.add(1, { layer_type: layerType });
  } catch (error) {
    // Silently fail - don't break app functionality
    console.debug('Failed to track layer added:', error);
  }
}

/**
 * Track when a layer/node is removed from the canvas.
 */
export function trackLayerRemoved(layerType: BlockType): void {
  try {
    const metrics = getMetrics();
    metrics.layerRemoved.add(1, { layer_type: layerType });
  } catch (error) {
    console.debug('Failed to track layer removed:', error);
  }
}

/**
 * Track when nodes are connected.
 */
export function trackNodeConnected(fromType: BlockType, toType: BlockType): void {
  try {
    const metrics = getMetrics();
    metrics.nodeConnected.add(1, {
      from_type: fromType,
      to_type: toType,
    });
  } catch (error) {
    console.debug('Failed to track node connection:', error);
  }
}

/**
 * Track parameter edits (sampled at 10% to reduce cardinality).
 */
export function trackParameterEdit(nodeType: BlockType, parameterName: string): void {
  try {
    // Sample at 10% to reduce cardinality
    if (Math.random() > 0.1) return;

    const metrics = getMetrics();
    metrics.parameterEdited.add(1, {
      node_type: nodeType,
      parameter: parameterName,
    });
  } catch (error) {
    console.debug('Failed to track parameter edit:', error);
  }
}

/**
 * Track validation trigger.
 */
export function trackValidationTriggered(): void {
  try {
    const metrics = getMetrics();
    metrics.validationTriggered.add(1, {});
  } catch (error) {
    console.debug('Failed to track validation:', error);
  }
}

/**
 * Track validation errors.
 */
export function trackValidationErrors(errorCount: number, errorType: string): void {
  try {
    const metrics = getMetrics();
    metrics.validationErrors.add(errorCount, {
      error_type: errorType,
    });
  } catch (error) {
    console.debug('Failed to track validation errors:', error);
  }
}

/**
 * Track session start.
 */
export function trackSessionStart(): void {
  try {
    const metrics = getMetrics();
    metrics.sessionStart.add(1, {});
  } catch (error) {
    console.debug('Failed to track session start:', error);
  }
}

/**
 * Track session duration on unmount.
 */
export function trackSessionDuration(durationSeconds: number): void {
  try {
    const metrics = getMetrics();
    metrics.sessionDuration.record(durationSeconds, {});
  } catch (error) {
    console.debug('Failed to track session duration:', error);
  }
}

/**
 * Track canvas time spent.
 */
export function trackCanvasTimeSpent(durationSeconds: number): void {
  try {
    const metrics = getMetrics();
    metrics.canvasTimeSpent.record(durationSeconds, {});
  } catch (error) {
    console.debug('Failed to track canvas time:', error);
  }
}
