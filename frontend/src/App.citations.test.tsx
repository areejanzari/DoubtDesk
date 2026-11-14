import { beforeEach, describe, expect, it, vi } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import App from './App'
import { streamResponse } from './test/mockStream'

const id = 'e8be2d4d-9094-43e8-8cec-7fe4e7853cce'
const other = '1c2f3e40-1111-2222-3333-444455556666'
const chatId = '00000000-0000-4000-8000-000000000001'
const chat = { conversation_id: chatId, title: 'Ureotelic animals?', message_count: 1, created_at: '2026-11-14', last_accessed_ns: 1 }
const answer = { status: 'grounded', answer: 'Ureotelic animals mainly excrete urea. (PDF p. 1)', formatted_answer: 'biology.pdf: PDF page(s) 1', sources: [{ doc_id: other, page_numbers: [1] }], conversation_id: chatId }
const response = (body: unknown) => ({ ok: true, status: 200, headers: new Headers({ 'content-type': 'application/json' }), json: async () => body }) as Response

beforeEach(() => {
  window.history.replaceState({}, '', '/')
  vi.mocked(fetch).mockImplementation(async input => {
    const url = String(input)
    if (url.endsWith('/auth/me')) return response({ user_id: 'owner', username: 'student', email: 'student@example.com', csrf_token: 'csrf' })
    if (url.endsWith('/documents')) return response([{ doc_id: id, filename: 'biology.pdf', page_count: 5 }, { doc_id: other, filename: 'biology.pdf', page_count: 5 }])
    if (url.endsWith('/conversations')) return response([chat])
    if (url.endsWith(`/conversations/${chatId}`)) return response({ ...chat, messages: [{ id: 1, question: chat.title, ...answer, doc_ids: [other], created_at: '2026-11-14' }] })
    if (url.endsWith('/ask/stream')) return streamResponse(answer, 7)
    if (url.includes('/pages/1/image')) return { ok: true, status: 200, headers: new Headers({ 'content-type': 'image/png' }), blob: async () => new Blob(['image'], { type: 'image/png' }) } as Response
    throw new Error(`Unexpected request ${url}`)
  })
})

describe('clickable PDF citations in the real App', () => {
  it.each(['stream', 'history'])('opens the exact source from %s', async mode => {
    render(<App />)
    await screen.findByRole('heading', { name: 'AI Tutor' })
    await screen.findAllByText('biology.pdf', { exact: true })
    const user = userEvent.setup()
    if (mode === 'stream') {
      await user.click(screen.getAllByRole('checkbox')[1])
      await user.type(screen.getByPlaceholderText('Ask a follow-up question…'), chat.title)
      await user.click(screen.getByRole('button', { name: 'Send message' }))
    } else {
      await user.click(await screen.findByRole('button', {name: chat.title}))
    }
    const link = await screen.findByRole('button', { name: 'Open source 1' })
    expect(link).toHaveTextContent('[1]')
    expect(screen.queryByText(/\(PDF p\. 1\)/)).not.toBeInTheDocument()
    await user.click(link)
    await waitFor(() => expect(vi.mocked(fetch).mock.calls.some(([url]) => String(url).endsWith(`/documents/${other}/pages/1/image`))).toBe(true))
    expect(vi.mocked(fetch).mock.calls.some(([url]) => String(url).endsWith(`/documents/${id}/pages/1/image`))).toBe(false)
    expect(screen.getByText('Page 1 of 5')).toBeInTheDocument()
  })
})
