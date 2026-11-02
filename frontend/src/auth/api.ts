export const API_BASE = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '')

export type AuthUser = {
  user_id: string
  username: string
  email: string
  csrf_token: string
}

export class ApiError extends Error {
  status: number

  constructor(message: string, status: number) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

function detailMessage(detail: unknown): string | undefined {
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail)) {
    const descriptions = detail
      .filter((item): item is { msg: string } => !!item && typeof item.msg === 'string')
      .map((item) => item.msg.replace(/^Value error, /, ''))
    if (descriptions.length) return descriptions.join('; ')
  }
  return undefined
}

export async function apiRequest<T>(
  path: string,
  options: RequestInit = {},
  csrfToken?: string,
  onUnauthorized?: () => void,
): Promise<T> {
  const headers = new Headers(options.headers)
  if (csrfToken) headers.set('X-CSRF-Token', csrfToken)

  let response: Response
  try {
    response = await fetch(`${API_BASE}${path}`, {
      ...options,
      credentials: 'include',
      headers,
    })
  } catch {
    throw new Error('Unable to connect to DoubtDesk. Check that the API is running.')
  }

  if (response.status === 401 && onUnauthorized) onUnauthorized()

  if (!response.ok) {
    const body: unknown = await response.json().catch(() => null)
    const detail = body && typeof body === 'object' && 'detail' in body
      ? detailMessage(body.detail)
      : undefined
    throw new ApiError(detail || `Request failed (${response.status}).`, response.status)
  }

  if (response.status === 204) return undefined as T
  const type = response.headers.get('content-type') || ''
  if (type.includes('image/') || type.includes('application/octet-stream')) {
    return (await response.blob()) as T
  }
  return response.json() as Promise<T>
}

export function jsonPost<T>(path: string, body: unknown, csrfToken?: string, onUnauthorized?: () => void) {
  return apiRequest<T>(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  }, csrfToken, onUnauthorized)
}

export const authApi = {
  me: () => apiRequest<AuthUser>('/auth/me'),
  login: (identifier: string, password: string) =>
    jsonPost<AuthUser>('/auth/login', { identifier, password }),
  signup: (username: string, email: string, password: string) =>
    jsonPost<{ user_id: string; username: string; email: string; email_verified: boolean }>(
      '/auth/signup', { username, email, password },
    ),
  verify: (token: string) => jsonPost<{ message: string }>('/auth/verify-email', { token }),
  resend: (email: string) => jsonPost<{ message: string }>('/auth/resend-verification', { email }),
  forgot: (email: string) => jsonPost<{ message: string }>('/auth/forgot-password', { email }),
  reset: (token: string, newPassword: string) =>
    jsonPost<{ message: string }>('/auth/reset-password', { token, new_password: newPassword }),
  logout: (csrfToken: string) => jsonPost<{ message: string }>('/auth/logout', {}, csrfToken),
}
