import { beforeEach, describe, expect, it, vi } from 'vitest'
import { conversationApi } from './conversations'

const unauthorized = vi.fn()
const data = { conversation_id: 'id', title: 'New chat', created_at: 'date', last_accessed_ns: 3, message_count: 0 }

beforeEach(() => {
  vi.stubGlobal('fetch', vi.fn(async (url: RequestInfo | URL, options?: RequestInit) => ({
    ok: true, status: options?.method === 'DELETE' ? 204 : 200,
    headers: new Headers({ 'content-type': 'application/json' }),
    json: async () => String(url) === '/conversations' && !options?.method ? [data] : data,
  })))
  unauthorized.mockClear()
})

describe('Conversation API client', () => {
  it('lists current-user conversations with cookie authentication', async () => {
    await expect(conversationApi.list(unauthorized)).resolves.toEqual([data])
    expect(fetch).toHaveBeenCalledWith('/conversations', expect.objectContaining({ credentials: 'include' }))
  })


  it('rejects malformed conversation lists instead of crashing the sidebar', async () => {
    vi.mocked(fetch).mockImplementationOnce(async () => ({
      ok: true, status: 200, headers: new Headers({ 'content-type': 'application/json' }),
      json: async () => ({ message: 'Not a conversation list' }),
    } as Response))
    await expect(conversationApi.list(unauthorized)).rejects.toThrow('Unexpected conversation list response')
  })

  it('rejects malformed conversation entries', async () => {
    vi.mocked(fetch).mockImplementationOnce(async () => ({
      ok: true, status: 200, headers: new Headers({ 'content-type': 'application/json' }),
      json: async () => [{ conversation_id: null, title: 123 }],
    } as Response))
    await expect(conversationApi.list(unauthorized)).rejects.toThrow('Unexpected conversation list response')
  })
  it('creates a chat with CSRF protection', async () => {
    await conversationApi.create('csrf', unauthorized)
    expect(fetch).toHaveBeenCalledWith('/conversations', expect.objectContaining({
      method: 'POST', headers: expect.any(Headers),
    }))
    const options = vi.mocked(fetch).mock.calls[0][1]!
    expect(new Headers(options.headers).get('X-CSRF-Token')).toBe('csrf')
  })

  it('loads history, renames safely, and deletes the right conversation', async () => {
    const id = 'one chat/with slash'
    await conversationApi.get(id, unauthorized)
    await conversationApi.rename(id, 'New title', 'csrf', unauthorized)
    await conversationApi.remove(id, 'csrf', unauthorized)
    const calls = vi.mocked(fetch).mock.calls
    expect(calls.map(([url]) => url)).toEqual(Array(3).fill('/conversations/one%20chat%2Fwith%20slash'))
    expect(calls[1][1]?.method).toBe('PATCH')
    expect(JSON.parse(String(calls[1][1]?.body))).toEqual({ title: 'New title' })
    expect(calls[2][1]?.method).toBe('DELETE')
  })
})
