import { beforeEach, describe, expect, it, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { ConversationSidebar } from './ConversationSidebar'
import type { ConversationSummary } from './conversations'

const chat = (number: number): ConversationSummary => ({
  conversation_id: `chat-${number}`, title: `Biology chat ${number}`,
  created_at: '2026-11-13', last_accessed_ns: number, message_count: 2,
})
const props = {
  conversations: [chat(1), chat(2)], activeId: 'chat-1', busy: false, loading: false, error: '',
  onNew: vi.fn(), onOpen: vi.fn(), onRename: vi.fn().mockResolvedValue(undefined),
  onDelete: vi.fn().mockResolvedValue(undefined),
}

beforeEach(() => {
  vi.clearAllMocks()
  vi.stubGlobal('confirm', vi.fn(() => true))
})

describe('Conversation sidebar', () => {
  it('lists chats, active chat and the current count', async () => {
    render(<ConversationSidebar {...props} />)
    expect(screen.getByText('2/30')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Biology chat 1' })).toHaveAttribute('aria-current', 'page')
    await userEvent.setup().click(screen.getByRole('button', { name: 'Biology chat 2' }))
    expect(props.onOpen).toHaveBeenCalledWith('chat-2')
    await userEvent.setup().click(screen.getByRole('button', { name: 'New Chat' }))
    expect(props.onNew).toHaveBeenCalledOnce()
  })

  it('enforces 30-chat limit without deleting existing chats', () => {
    render(<ConversationSidebar {...props} conversations={Array.from({ length: 30 }, (_, i) => chat(i))} />)
    expect(screen.getByText('30/30')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'New Chat' })).toBeDisabled()
    expect(screen.getByText(/30-chat limit reached/)).toBeInTheDocument()
    expect(screen.getAllByLabelText(/^Delete Biology chat/)).toHaveLength(30)
  })

  it('renames a chat via an inline form', async () => {
    render(<ConversationSidebar {...props} />)
    const user = userEvent.setup()
    await user.click(screen.getByRole('button', { name: 'Rename Biology chat 1' }))
    await user.clear(screen.getByRole('textbox', { name: 'Conversation title' }))
    expect(screen.getByRole('button', { name: 'Save title' })).toBeDisabled()
    await user.type(screen.getByRole('textbox', { name: 'Conversation title' }), 'New name')
    await user.click(screen.getByRole('button', { name: 'Save title' }))
    expect(props.onRename).toHaveBeenCalledWith('chat-1', 'New name')
  })

  it('supports cancelling a rename', async () => {
    render(<ConversationSidebar {...props} />)
    await userEvent.setup().click(screen.getByRole('button', { name: 'Rename Biology chat 1' }))
    await userEvent.setup().click(screen.getByRole('button', { name: 'Cancel rename' }))
    expect(screen.queryByRole('textbox', { name: 'Conversation title' })).not.toBeInTheDocument()
  })

  it('requires deletion confirmation', async () => {
    render(<ConversationSidebar {...props} />)
    vi.mocked(window.confirm).mockReturnValueOnce(false)
    await userEvent.setup().click(screen.getByRole('button', { name: 'Delete Biology chat 1' }))
    expect(props.onDelete).not.toHaveBeenCalled()
    await userEvent.setup().click(screen.getByRole('button', { name: 'Delete Biology chat 1' }))
    expect(props.onDelete).toHaveBeenCalledWith('chat-1')
  })

  it('displays loading, empty state, errors and disables actions when busy', () => {
    render(<ConversationSidebar {...props} conversations={[]} loading error="Cannot load history" busy />)
    expect(screen.getByText('Loading chats…')).toBeInTheDocument()
    expect(screen.getByRole('alert')).toHaveTextContent('Cannot load history')
    expect(screen.getByRole('button', { name: 'New Chat' })).toBeDisabled()
  })
})
