type Answer = Record<string, unknown>
export type MockEvent = [string, Record<string, unknown>]

function eventStream(events: MockEvent[], chunkSize = 0, gate?: Promise<void>, gateAfter = events.length): ReadableStream<Uint8Array> {
  const encode = (items: MockEvent[]) => new TextEncoder().encode(items.map(([event, data]) => `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`).join(''))
  const parts = [encode(events.slice(0, gateAfter)), encode(events.slice(gateAfter))]
  let part = 0
  let offset = 0
  return new ReadableStream<Uint8Array>({
    async pull(controller) {
      while (part < parts.length && offset >= parts[part].length) {
        part += 1
        offset = 0
        if (part === 1 && gate) await gate
      }
      if (part >= parts.length) { controller.close(); return }
      const bytes = parts[part]
      const end = chunkSize ? Math.min(offset + chunkSize, bytes.length) : bytes.length
      controller.enqueue(bytes.slice(offset, end))
      offset = end
    },
  })
}

export function eventsResponse(events: MockEvent[], options: { chunkSize?: number; gate?: Promise<void>; gateAfter?: number } = {}): Response {
  return {
    ok: true, status: 200,
    headers: new Headers({ 'content-type': 'text/event-stream' }),
    body: eventStream(events, options.chunkSize, options.gate, options.gateAfter),
    json: async () => events.find(([event]) => event === 'complete')?.[1],
  } as Response
}

export function streamResponse(answer: Answer, chunkSize = 0): Response {
  return eventsResponse([
    ['status', { stage: 'preparing' }],
    ['status', { stage: 'checking' }],
    ['token', { text: answer.answer || '' }],
    ['complete', answer],
  ], { chunkSize })
}

export function revisedStreamEvents(rejected: string, answer: Answer): MockEvent[] {
  return [
    ['status', { stage: 'preparing' }],
    ['status', { stage: 'generating' }],
    ['status', { stage: 'verifying' }],
    ['token', { text: rejected }],
    ['status', { stage: 'checking' }],
    ['reset', { reason: 'judge_rejected' }],
    ['status', { stage: 'revising' }],
    ['token', { text: String(answer.answer || '') }],
    ['complete', answer],
  ]
}
