/**
 * Hook for tracking user session metrics.
 *
 * Automatically tracks session start and duration.
 */

import { useEffect, useRef } from 'react';
import { trackSessionStart, trackSessionDuration, trackCanvasTimeSpent } from '../lib/storeInstrumentation';

/**
 * Track session metrics on mount/unmount.
 */
export function useSessionTracking() {
  const sessionStartTime = useRef<number>(Date.now());

  useEffect(() => {
    // Track session start
    trackSessionStart();

    // Track session duration on unmount
    return () => {
      const durationSeconds = (Date.now() - sessionStartTime.current) / 1000;
      trackSessionDuration(durationSeconds);
    };
  }, []);
}

/**
 * Track time spent on canvas.
 */
export function useCanvasTimeTracking() {
  const canvasStartTime = useRef<number>(Date.now());
  const isTracking = useRef<boolean>(true);

  useEffect(() => {
    // Track canvas time on unmount
    return () => {
      if (isTracking.current) {
        const durationSeconds = (Date.now() - canvasStartTime.current) / 1000;
        trackCanvasTimeSpent(durationSeconds);
        isTracking.current = false;
      }
    };
  }, []);
}
