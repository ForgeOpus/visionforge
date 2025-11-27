/**
 * Shared API utilities for all API files
 * Centralizes CSRF token handling and fetch configuration
 */

export const API_BASE_URL = import.meta.env.VITE_API_URL || '/api'

/**
 * Get CSRF token from cookie
 */
export function getCsrfToken(): string | null {
  const name = 'csrftoken'
  const cookies = document.cookie.split(';')
  for (let cookie of cookies) {
    cookie = cookie.trim()
    if (cookie.startsWith(name + '=')) {
      return cookie.substring(name.length + 1)
    }
  }
  return null
}

/**
 * Create headers with CSRF token for unsafe HTTP methods
 */
export function createHeaders(method?: string, additionalHeaders?: HeadersInit): HeadersInit {
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    ...(additionalHeaders as Record<string, string> || {}),
  }

  // Add CSRF token for unsafe methods (POST, PUT, DELETE, PATCH)
  const isUnsafeMethod = method && 
    !['GET', 'HEAD', 'OPTIONS', 'TRACE'].includes(method.toUpperCase())

  if (isUnsafeMethod) {
    const csrfToken = getCsrfToken()
    if (csrfToken) {
      headers['X-CSRFToken'] = csrfToken
    } else {
      console.warn('CSRF token not found. Cookie:', document.cookie)
    }
  }

  return headers
}

/**
 * Create fetch options with credentials and headers
 */
export function createFetchOptions(
  method: string,
  body?: any,
  additionalHeaders?: HeadersInit
): RequestInit {
  const options: RequestInit = {
    method,
    headers: createHeaders(method, additionalHeaders),
    credentials: 'include',
  }

  if (body) {
    options.body = JSON.stringify(body)
  }

  return options
}
