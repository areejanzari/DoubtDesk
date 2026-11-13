import { beforeEach, describe, expect, it, vi } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import App from './App'
import { streamResponse } from './test/mockStream'

const account = { user_id: 'owner-1', username: 'student', email: 'student@example.com', csrf_token: 'csrf' }
const id = '00000000-0000-4000-8000-000000000001'
const chat = { conversation_id: id, title: 'Nephron structure', message_count: 1, created_at: '2026-11-13', last_accessed_ns: 1 }
const detail = { ...chat, messages: [
  { id: 1, question: 'Describe a nephron', answer: 'A nephron filters blood.',
    formatted_answer: 'biology.pdf: PDF page(s) 2', status: 'grounded',
    sources: [{ doc_id: 'doc-1', page_numbers: [2] }], doc_ids: ['doc-1'], created_at: '2026-11-13' },
] }
const document = { doc_id: 'doc-1', filename: 'biology.pdf', page_count: 5 }

const response = (body: unknown, status = 200) => ({
  ok: status >= 200 && status < 300, status,
  headers: new Headers({ 'content-type': 'application/json' }),
  json: async () => body,
}) as Response

let chats = [chat]
let details = detail
let fails = ''

beforeEach(() => {
  chats = [chat]
  details = detail
  fails = ''
  vi.stubGlobal('confirm', vi.fn(() => true))
  vi.mocked(fetch).mockImplementation(async (input, options) => {
    const url = String(input)
    if (url === '/auth/me') return response(account)
    if (url === '/documents') return response([document])
    if (url === '/conversations' && (!options?.method || options.method === 'GET')) {
      return fails === 'list' ? response({ detail: 'History offline' }, 503) : response(chats)
    }
    if (url === '/conversations' && options?.method === 'POST') {
      if (fails === 'create') return response({ detail: 'Limit reached' }, 409)
      const created = { ...chat, conversation_id: '00000000-0000-4000-8000-000000000002', title: 'New chat', message_count: 0 }
      chats = [created, ...chats]
      return response(created, 201)
    }
    if (url === `/conversations/${id}` && (!options?.method || options.method === 'GET')) {
      return fails === 'get' ? response({ detail: 'Chat unavailable' }, 404) : response(details)
    }
    if (url === `/conversations/${id}` && options?.method === 'PATCH') {
      if (fails === 'rename') return response({ detail: 'Rename denied' }, 422)
      const updated = { ...chat, title: JSON.parse(String(options.body)).title }
      chats = chats.map(item => item.conversation_id === id ? updated : item)
      return response(updated)
    }
    if (url === `/conversations/${id}` && options?.method === 'DELETE') {
      if (fails === 'delete') return response({ detail: 'Delete denied' }, 503)
      chats = chats.filter(item => item.conversation_id !== id)
      return { ok: true, status: 204, headers: new Headers() } as Response
    }
    if (url === '/ask/stream') {
      chats = [chat]
      return streamResponse({ status: 'grounded', answer: 'A nephron filters blood.',
      formatted_answer: 'biology.pdf: PDF page(s) 2', sources: [{ doc_id: 'doc-1', page_numbers: [2] }], conversation_id: id })
    }
    return response({ detail: 'Unmocked endpoint' }, 404)
  })
})

async function openApp() {
  render(<App />)
  await screen.findByRole('heading', { name: 'AI Tutor' })
  await screen.findByRole('button', { name: 'Nephron structure' })
  await screen.findByText('biology.pdf')
}

describe('Persistent conversation history integration', () => {
  it('loads saved conversation and associated PDF citations', async () => {
    await openApp()
    await userEvent.setup().click(screen.getByRole('button', { name: 'Nephron structure' }))
    expect(await screen.findByText('Describe a nephron')).toBeInTheDocument()
    expect(screen.getByText('A nephron filters blood.')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /biology\.pdf.*Page 2/ })).toBeInTheDocument()
    expect(screen.getByRole('checkbox')).toBeChecked()
  })

  it('creates a new chat and clears the previous messages', async () => {
    await openApp()
    await userEvent.setup().click(screen.getByRole('button', { name: 'Nephron structure' }))
    await screen.findByText('Describe a nephron')
    await userEvent.setup().click(screen.getByRole('button', { name: 'New Chat' }))
    expect(await screen.findByRole('button', { name: 'New chat' })).toBeInTheDocument()
    expect(screen.queryByText('Describe a nephron')).not.toBeInTheDocument()
    const request = vi.mocked(fetch).mock.calls.find(([url, opts]) => url === '/conversations' && opts?.method === 'POST')!
    expect(new Headers(request[1]?.headers).get('X-CSRF-Token')).toBe('csrf')
  })

  it('renames conversation without losing the current chat', async () => {
    await openApp()
    const user = userEvent.setup()
    await user.click(screen.getByRole('button', { name: 'Rename Nephron structure' }))
    await user.clear(screen.getByRole('textbox', { name: 'Conversation title' }))
    await user.type(screen.getByRole('textbox', { name: 'Conversation title' }), 'Kidney notes')
    await user.click(screen.getByRole('button', { name: 'Save title' }))
    expect(await screen.findByRole('button', { name: 'Kidney notes' })).toBeInTheDocument()
  })

  it('deletes a chat and clears currently loaded messages', async () => {
    await openApp()
    const user = userEvent.setup()
    await user.click(screen.getByRole('button', { name: 'Nephron structure' }))
    await screen.findByText('Describe a nephron')
    await user.click(screen.getByRole('button', { name: 'Delete Nephron structure' }))
    await waitFor(() => expect(screen.queryByText('Describe a nephron')).not.toBeInTheDocument())
    expect(screen.getByText('No saved chats yet.')).toBeInTheDocument()
  })

  it('disables New Chat at 30 and blocks unassigned requests', async () => {
    chats = Array.from({ length: 30 }, (_, n) => ({ ...chat, conversation_id: `chat-${n}`, title: `Topic ${n}` }))
    render(<App />)
    await screen.findByText('30/30')
    expect(screen.getByRole('button', { name: 'New Chat' })).toBeDisabled()
    await userEvent.setup().type(screen.getByPlaceholderText('Ask a follow-up question…'), 'Can you teach?')
    await userEvent.setup().click(screen.getByRole('button', { name: 'Send message' }))
    expect(screen.getByRole('alert')).toHaveTextContent('30-chat limit reached')
    expect(vi.mocked(fetch).mock.calls.some(([url]) => url === '/ask/stream')).toBe(false)
  })

  it('shows history-load errors and does not lose the tutor', async () => {
    fails = 'list'
    render(<App />)
    expect(await screen.findByText('History offline')).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'AI Tutor' })).toBeInTheDocument()
  })

  it('shows errors when a saved chat cannot be restored', async () => {
    await openApp()
    fails = 'get'
    await userEvent.setup().click(screen.getByRole('button', { name: 'Nephron structure' }))
    expect(await screen.findByText('Chat unavailable')).toBeInTheDocument()
  })

  it('does not silently erase a chat when deletion fails', async () => {
    await openApp()
    fails = 'delete'
    await userEvent.setup().click(screen.getByRole('button', { name: 'Delete Nephron structure' }))
    expect(await screen.findByText('Delete denied')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Nephron structure' })).toBeInTheDocument()
  })

  it('streams the answer, then refreshes the saved conversation list', async () => {
    chats = []
    render(<App />)
    await screen.findByRole('heading', { name: 'AI Tutor' })
    const user = userEvent.setup()
    await user.type(screen.getByPlaceholderText('Ask a follow-up question…'), 'What is a nephron?')
    await user.click(screen.getByRole('button', { name: 'Send message' }))
    expect(await screen.findByText('A nephron filters blood.')).toBeInTheDocument()
    expect(vi.mocked(fetch).mock.calls.some(([url]) => url === '/ask/stream')).toBe(true)
    expect(await screen.findByRole('button', { name: 'Nephron structure' })).toBeInTheDocument()
  })
  it('shows a server-side 409 when creating a chat fails', async () => {
    await openApp()
    fails = 'create'
    await userEvent.setup().click(screen.getByRole('button', { name: 'New Chat' }))
    expect(await screen.findByText('Limit reached')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Nephron structure' })).toBeInTheDocument()
  })

  it('shows rename failure and keeps the original title', async () => {
    await openApp()
    fails = 'rename'
    await userEvent.setup().click(screen.getByRole('button', { name: 'Rename Nephron structure' }))
    await userEvent.setup().click(screen.getByRole('button', { name: 'Save title' }))
    expect(await screen.findByText('Rename denied')).toBeInTheDocument()
    expect(screen.getByRole('textbox', { name: 'Conversation title' })).toBeInTheDocument()
  })

})
