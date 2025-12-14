/**
 * Guest state management for localStorage-based canvas persistence
 * Allows guests to try the canvas without signup, with auto-save to localStorage
 */
import { Node, Edge } from '@xyflow/react';
import { BlockData } from './types';

const GUEST_CANVAS_KEY = 'visionforge_guest_canvas';
const GUEST_TIMESTAMP_KEY = 'visionforge_guest_timestamp';

export interface GuestCanvasState {
  nodes: Node<BlockData>[];
  edges: Edge[];
  groupDefinitions?: any[];
  timestamp: number;
}

/**
 * Save guest canvas to localStorage
 * Auto-saves the current canvas state for guest users
 */
export function saveGuestCanvas(
  nodes: Node<BlockData>[],
  edges: Edge[],
  groupDefinitions?: Map<string, any>
): void {
  try {
    const canvasState: GuestCanvasState = {
      nodes,
      edges,
      groupDefinitions: groupDefinitions ? Array.from(groupDefinitions.values()) : [],
      timestamp: Date.now(),
    };

    localStorage.setItem(GUEST_CANVAS_KEY, JSON.stringify(canvasState));
    localStorage.setItem(GUEST_TIMESTAMP_KEY, canvasState.timestamp.toString());
  } catch (error) {
    console.error('Failed to save guest canvas to localStorage:', error);
  }
}

/**
 * Load guest canvas from localStorage
 * Returns null if no saved canvas exists
 */
export function loadGuestCanvas(): GuestCanvasState | null {
  try {
    const savedCanvas = localStorage.getItem(GUEST_CANVAS_KEY);

    if (!savedCanvas) {
      return null;
    }

    const canvasState: GuestCanvasState = JSON.parse(savedCanvas);
    return canvasState;
  } catch (error) {
    console.error('Failed to load guest canvas from localStorage:', error);
    return null;
  }
}

/**
 * Check if guest has unsaved canvas work
 * Returns true if there's canvas data in localStorage
 */
export function hasGuestCanvas(): boolean {
  return localStorage.getItem(GUEST_CANVAS_KEY) !== null;
}

/**
 * Get guest canvas data for transfer to user account
 * Used during guest-to-user conversion flow
 */
export function getGuestCanvasForTransfer(): {
  nodes: Node<BlockData>[];
  edges: Edge[];
  groupDefinitions?: any[];
} | null {
  const canvasState = loadGuestCanvas();

  if (!canvasState) {
    return null;
  }

  return {
    nodes: canvasState.nodes,
    edges: canvasState.edges,
    groupDefinitions: canvasState.groupDefinitions,
  };
}

/**
 * Clear guest canvas from localStorage
 * Called after successful transfer to user account or on explicit reset
 */
export function clearGuestCanvas(): void {
  try {
    localStorage.removeItem(GUEST_CANVAS_KEY);
    localStorage.removeItem(GUEST_TIMESTAMP_KEY);
  } catch (error) {
    console.error('Failed to clear guest canvas from localStorage:', error);
  }
}

/**
 * Get timestamp of last guest canvas save
 * Returns null if no saved canvas exists
 */
export function getGuestCanvasTimestamp(): number | null {
  try {
    const timestamp = localStorage.getItem(GUEST_TIMESTAMP_KEY);
    return timestamp ? parseInt(timestamp, 10) : null;
  } catch (error) {
    console.error('Failed to get guest canvas timestamp:', error);
    return null;
  }
}

/**
 * Check if guest canvas is recent (within last 7 days)
 * Used to determine if we should prompt user about saving their work
 */
export function isGuestCanvasRecent(): boolean {
  const timestamp = getGuestCanvasTimestamp();

  if (!timestamp) {
    return false;
  }

  const sevenDaysAgo = Date.now() - (7 * 24 * 60 * 60 * 1000);
  return timestamp > sevenDaysAgo;
}
