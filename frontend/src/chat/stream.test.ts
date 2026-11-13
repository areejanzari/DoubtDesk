import { beforeEach, describe, expect, it, vi } from 'vitest'
import { streamAsk } from './stream'
import { streamResponse } from '../test/mockStream'

const payload = { question: 'Explain atoms', doc_ids: ['doc-1'] }
const answer = { status: 'grounded', answer: 'Atoms are matter.', sources: [], formatted_answer: '', conversation_id: 'chat-1' }
const unauthorized = vi.fn()
const events = vi.fn()

beforeEach(() => {
  vi.stubGlobal('fetch', vi.fn())
  events.mockClear()
  unauthorized.mockClear()
})

describe('SSE answer transport', () => {
  it('handles arbitrarily fragmented events, retains cookies and CSRF', async () => {
    vi.mocked(fetch).mockResolvedValue(streamResponse(answer, 1))
    await streamAsk(payload, 'csrf-token', unauthorized, events)
    const received = events.mock.calls.map(([event]) => event.event)
    expect(received).toEqual(['status', 'status', 'token', 'complete'])
    expect(events.mock.calls.at(-1)?.[0].data).toEqual(answer)
    expect(fetch).toHaveBeenCalledWith('/ask/stream', expect.objectContaining({
      method: 'POST', credentials: 'include', body: JSON.stringify(payload),
      headers: expect.objectContaining({ 'X-CSRF-Token': 'csrf-token' }),
    }))
  })

  it('rejects HTTP errors with the backend message', async () => {
    vi.mocked(fetch).mockResolvedValue({
      ok: false, status: 409, json: async () => ({ detail: '30-chat limit reached.' }),
    } as Response)
    await expect(streamAsk(payload, 'csrf', unauthorized, events)).rejects.toThrow('30-chat limit reached.')
  })

  it('notifies the app about an expired session', async () => {
    vi.mocked(fetch).mockResolvedValue({ ok: false, status: 401, json: async () => null } as Response)
    await expect(streamAsk(payload, 'csrf', unauthorized, events)).rejects.toThrow('Request failed (401).')
    expect(unauthorized).toHaveBeenCalledOnce()
  })

  it('surfaces fetch failures with a helpful message', async () => {
    vi.mocked(fetch).mockRejectedValue(new TypeError('fetch failed'))
    await expect(streamAsk(payload, 'csrf', unauthorized, events)).rejects.toThrow('Unable to connect to DoubtDesk')
  })

  it('requires a readable event-stream body', async () => {
    vi.mocked(fetch).mockResolvedValue({ ok: true, status: 200, body: null } as Response)
    await expect(streamAsk(payload, 'csrf', unauthorized, events)).rejects.toThrow('Streaming response has no body.')
  })

  it('does not treat a truncated response as a successful answer', async () => {
    const body = new ReadableStream({ start(controller) {
      controller.enqueue(new TextEncoder().encode('event: token\ndata: {"text":"partial"}\n\n'))
      controller.close()
    } })
    vi.mocked(fetch).mockResolvedValue({ ok: true, body, status: 200 } as Response)
    await expect(streamAsk(payload, 'csrf', unauthorized, events)).rejects.toThrow('without completing')
  })

  it('surfaces SSE error messages and rejects malformed JSON', async () => {
    for (const text of ['event: error\ndata: {"message":"Provider offline"}\n\n', 'event: token\ndata: {invalid}\n\n']) {
      const body = new ReadableStream({ start(controller) {
        controller.enqueue(new TextEncoder().encode(text))
        controller.close()
      } })
      vi.mocked(fetch).mockResolvedValue({ ok: true, body, status: 200 } as Response)
      await expect(streamAsk(payload, 'csrf', unauthorized, events)).rejects.toThrow(
        text.includes('Provider') ? 'Provider offline' : 'Invalid event',
      )
    }
  })
})
