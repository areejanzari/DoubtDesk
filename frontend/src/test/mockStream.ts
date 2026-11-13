type Answer = Record<string, unknown>

export function streamResponse(answer: Answer, chunkSize = 0): Response {
  const output = [
    'event: status\ndata: {"stage":"preparing"}\n\n',
    'event: status\ndata: {"stage":"checking"}\n\n',
    `event: token\ndata: ${JSON.stringify({ text: answer.answer || '' })}\n\n`,
    `event: complete\ndata: ${JSON.stringify(answer)}\n\n`,
  ].join('')
  const bytes = new TextEncoder().encode(output)
  let offset = 0
  const body = new ReadableStream<Uint8Array>({
    pull(controller) {
      if (offset >= bytes.length) { controller.close(); return }
      const end = chunkSize ? Math.min(offset + chunkSize, bytes.length) : bytes.length
      controller.enqueue(bytes.slice(offset, end))
      offset = end
    },
  })
  return {
    ok: true, status: 200,
    headers: new Headers({ 'content-type': 'text/event-stream' }),
    body,
    json: async () => answer,
  } as Response
}
