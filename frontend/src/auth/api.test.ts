import { beforeEach, describe, expect, it, vi } from 'vitest'
import { apiRequest, jsonPost, ApiError } from './api'

const response = (body: unknown, status = 200, contentType = 'application/json'): Response => ({
  ok: status >= 200 && status < 300,
  status,
  headers: new Headers({ 'content-type': contentType }),
  json: async () => body,
  blob: async () => new Blob(['image-data'], { type: contentType }),
}) as Response

beforeEach(() => {
  vi.mocked(fetch).mockResolvedValue(response({ ok: true }))
})

describe('authenticated HTTP client', () => {
  it('includes session cookies and an optional CSRF token', async () => {
    const result = await apiRequest<{ ok: boolean }>('/documents', { method: 'POST' }, 'csrf-123')
    expect(result).toEqual({ ok: true })
    const [url, options] = vi.mocked(fetch).mock.calls[0]
    expect(String(url)).toMatch(/\/documents$/)
    expect(options?.credentials).toBe('include')
    expect(new Headers(options?.headers).get('X-CSRF-Token')).toBe('csrf-123')
  })

  it('does not send a CSRF header for read-only requests', async () => {
    await apiRequest('/documents')
    const options = vi.mocked(fetch).mock.calls[0][1]
    expect(new Headers(options?.headers).has('X-CSRF-Token')).toBe(false)
  })

  it('sends JSON POST requests with the expected content type', async () => {
    await jsonPost('/ask', { question: 'Hello' }, 'csrf-123')
    const options = vi.mocked(fetch).mock.calls[0][1]
    expect(options?.method).toBe('POST')
    expect(new Headers(options?.headers).get('Content-Type')).toBe('application/json')
    expect(new Headers(options?.headers).get('X-CSRF-Token')).toBe('csrf-123')
    expect(JSON.parse(String(options?.body))).toEqual({ question: 'Hello' })
  })

  it('calls the unauthorized handler and throws a typed error on 401', async () => {
    vi.mocked(fetch).mockResolvedValue(response({ detail: 'Session expired.' }, 401))
    const unauthorized = vi.fn()
    await expect(apiRequest('/documents', {}, undefined, unauthorized))
      .rejects.toMatchObject({ status: 401, message: 'Session expired.' })
    expect(unauthorized).toHaveBeenCalledOnce()
  })

  it('extracts multiple FastAPI validation errors', async () => {
    vi.mocked(fetch).mockResolvedValue(response({ detail: [
      { msg: 'Value error, Invalid email' },
      { msg: 'Password too short' },
      { other: 'ignored' },
    ] }, 422))
    await expect(apiRequest('/auth/signup'))
      .rejects.toThrow('Invalid email; Password too short')
  })

  it('uses the HTTP status if the server does not return a recognized detail', async () => {
    vi.mocked(fetch).mockResolvedValue(response({ error: 'unknown' }, 503))
    await expect(apiRequest('/documents')).rejects.toMatchObject({
      name: 'ApiError', status: 503, message: 'Request failed (503).',
    })
  })

  it('handles malformed error JSON without crashing', async () => {
    vi.mocked(fetch).mockResolvedValue({
      ...response({}, 400),
      json: async () => { throw new SyntaxError('Bad JSON') },
    } as Response)
    await expect(apiRequest('/ask')).rejects.toThrow('Request failed (400).')
  })

  it('produces a helpful message if the API is offline', async () => {
    vi.mocked(fetch).mockRejectedValue(new TypeError('Failed to fetch'))
    await expect(apiRequest('/auth/me')).rejects.toThrow(
      'Unable to connect to DoubtDesk. Check that the API is running.',
    )
  })

  it('handles empty responses', async () => {
    vi.mocked(fetch).mockResolvedValue(response(null, 204))
    await expect(apiRequest('/auth/logout')).resolves.toBeUndefined()
  })

  it('returns image responses as blobs', async () => {
    vi.mocked(fetch).mockResolvedValue(response(null, 200, 'image/png'))
    const result = await apiRequest<Blob>('/documents/id/pages/1/image')
    expect(result).toBeInstanceOf(Blob)
    expect(result.type).toBe('image/png')
  })

  it('returns octet-stream responses as blobs', async () => {
    vi.mocked(fetch).mockResolvedValue(response(null, 200, 'application/octet-stream'))
    expect(await apiRequest('/export')).toBeInstanceOf(Blob)
  })

  it('retains status and message on ApiError', () => {
    const error = new ApiError('Forbidden', 403)
    expect(error.name).toBe('ApiError')
    expect(error.status).toBe(403)
    expect(error.message).toBe('Forbidden')
  })
})
