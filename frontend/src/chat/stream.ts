import { API_BASE, ApiError } from '../auth/api'

export type StreamEvent = {
  event: string
  data: Record<string, unknown>
}

function describeError(detail: unknown, status: number): string {
  if (typeof detail === 'string' && detail.trim()) return detail
  return `Request failed (${status}).`
}

export async function streamAsk(
  payload: { question: string; doc_ids: string[]; conversation_id?: string },
  csrfToken: string,
  onUnauthorized: () => void,
  onEvent: (event: StreamEvent) => void,
): Promise<void> {
  let response: Response
  try {
    response = await fetch(`${API_BASE}/ask/stream`, {
      method: 'POST',
      credentials: 'include',
      headers: {
        'Content-Type': 'application/json',
        'X-CSRF-Token': csrfToken,
      },
      body: JSON.stringify(payload),
    })
  } catch {
    throw new Error('Unable to connect to DoubtDesk. Check that the API is running.')
  }
  if (response.status === 401) onUnauthorized()
  if (!response.ok) {
    const body = await response.json().catch(() => null)
    throw new ApiError(describeError(body?.detail, response.status), response.status)
  }
  if (!response.body) throw new Error('Streaming response has no body.')
  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let completed = false

  function consume(final = false) {
    buffer = buffer.replace(/\r\n/g, '\n')
    let index = buffer.indexOf('\n\n')
    while (index >= 0) {
      const block = buffer.slice(0, index)
      buffer = buffer.slice(index + 2)
      const eventName = block.split('\n').find(line => line.startsWith('event:'))?.slice(6).trim() || 'message'
      const raw = block.split('\n').filter(line => line.startsWith('data:')).map(line => line.slice(5).trimStart()).join('\n')
      if (raw) {
        let data: Record<string, unknown>
        try { data = JSON.parse(raw) as Record<string, unknown> }
        catch { throw new Error('Invalid event from the answer stream.') }
        onEvent({ event: eventName, data })
        if (eventName === 'complete') completed = true
        if (eventName === 'error') throw new Error(typeof data.message === 'string' ? data.message : 'Answer stream failed.')
      }
      index = buffer.indexOf('\n\n')
    }
    if (final && buffer.trim()) {
      buffer += '\n\n'
      consume()
    }
  }
  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      consume()
    }
    buffer += decoder.decode()
    consume(true)
    if (!completed) throw new Error('Answer stream ended without completing.')
  } finally {
    reader.releaseLock()
  }
}
