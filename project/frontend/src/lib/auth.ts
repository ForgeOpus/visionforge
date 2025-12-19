/**
 * Authentication utilities for Firebase token management
 */
import { auth } from './firebase';

/**
 * Get the current Firebase ID token for authenticated requests
 * @returns Promise<string | null> - The ID token or null if not authenticated
 */
export async function getAuthToken(): Promise<string | null> {
  const user = auth.currentUser;
  if (!user) {
    return null;
  }

  try {
    const token = await user.getIdToken();
    return token;
  } catch (error) {
    console.error('Failed to get auth token:', error);
    return null;
  }
}

/**
 * Get authentication headers for API requests
 * @returns Promise<HeadersInit> - Headers object with Authorization header if authenticated
 */
export async function getAuthHeaders(): Promise<HeadersInit> {
  const token = await getAuthToken();

  const headers: HeadersInit = {
    'Content-Type': 'application/json',
  };

  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }

  return headers;
}
