/**
 * Hook to detect user inactivity and trigger timeout
 */
import { useEffect, useRef, useState } from 'react';

interface UseIdleTimeoutOptions {
  onIdle: () => void;
  onWarning?: () => void;
  idleTime?: number; // milliseconds
  warningTime?: number; // milliseconds before idle
}

export const useIdleTimeout = ({
  onIdle,
  onWarning,
  idleTime = 10 * 60 * 1000, // 10 minutes default
  warningTime = 60 * 1000, // 1 minute warning default
}: UseIdleTimeoutOptions) => {
  const [isIdle, setIsIdle] = useState(false);
  const [showWarning, setShowWarning] = useState(false);
  const idleTimerRef = useRef<NodeJS.Timeout | null>(null);
  const warningTimerRef = useRef<NodeJS.Timeout | null>(null);

  const resetTimer = () => {
    // Clear existing timers
    if (idleTimerRef.current) {
      clearTimeout(idleTimerRef.current);
    }
    if (warningTimerRef.current) {
      clearTimeout(warningTimerRef.current);
    }

    // Reset idle state
    setIsIdle(false);
    setShowWarning(false);

    // Set warning timer
    if (onWarning) {
      warningTimerRef.current = setTimeout(() => {
        setShowWarning(true);
        onWarning();
      }, idleTime - warningTime);
    }

    // Set idle timer
    idleTimerRef.current = setTimeout(() => {
      setIsIdle(true);
      onIdle();
    }, idleTime);
  };

  useEffect(() => {
    // Events that reset the idle timer
    const events = [
      'mousedown',
      'mousemove',
      'keypress',
      'scroll',
      'touchstart',
      'click',
    ];

    // Reset timer on user activity
    const handleActivity = () => {
      if (!isIdle) {
        resetTimer();
      }
    };

    // Set up event listeners
    events.forEach((event) => {
      window.addEventListener(event, handleActivity);
    });

    // Start initial timer
    resetTimer();

    // Cleanup
    return () => {
      events.forEach((event) => {
        window.removeEventListener(event, handleActivity);
      });
      if (idleTimerRef.current) {
        clearTimeout(idleTimerRef.current);
      }
      if (warningTimerRef.current) {
        clearTimeout(warningTimerRef.current);
      }
    };
  }, [isIdle, idleTime, warningTime, onIdle, onWarning]);

  return { isIdle, showWarning, resetTimer };
};
