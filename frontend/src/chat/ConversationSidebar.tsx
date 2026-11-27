import { useState } from 'react'
import type { ReactNode } from 'react'
import { LoaderCircle, Pencil, Plus, Trash2, X, Check } from 'lucide-react'
import type { ConversationSummary } from './conversations'

export function ConversationSidebar({
  conversations, activeId, busy, loading, error, onNew, onOpen, onRename, onDelete, actions,
}: {
  conversations: ConversationSummary[]
  activeId: string | null
  busy: boolean
  loading: boolean
  error: string
  onNew: () => void
  onOpen: (id: string) => void
  onRename: (id: string, title: string) => Promise<void>
  onDelete: (id: string) => Promise<void>
  actions?: ReactNode
}) {
  const [editingId, setEditingId] = useState<string | null>(null)
  const [title, setTitle] = useState('')
  const atLimit = conversations.length >= 30

  return (
    <section aria-label="Conversation history" className="pb-2">
      <div className={`mb-5 grid gap-2 ${actions ? 'grid-cols-2' : 'grid-cols-1'}`}>
        <button type="button" onClick={onNew} disabled={busy || atLimit}
          className="side-button side-button--light">
          <Plus size={15} /> New Chat
        </button>
        {actions}
      </div>
      <div className="mb-2 flex items-baseline justify-between px-2">
        <span className="night-label">Recents</span>
        <span className="text-[11.5px] tabular-nums text-[#7E7E88]">{conversations.length}/30</span>
      </div>
      {atLimit && <p role="status" className="mb-2 px-2 text-xs leading-5 text-[#FFB27A]">30-chat limit reached. Delete a conversation to start another.</p>}
      {error && <p role="alert" className="mb-2 break-words px-2 text-xs leading-5 text-[#FF8A8A]">{error}</p>}
      {loading && <p className="flex items-center gap-2 px-2 text-xs text-[#7E7E88]"><LoaderCircle className="animate-spin" size={13} /> Loading chats…</p>}
      <div className="max-h-56 space-y-0.5 overflow-y-auto" aria-label="Recent chats">
        {!loading && conversations.length === 0 && <p className="px-2 py-2 text-xs text-[#7E7E88]">No saved chats yet.</p>}
        {conversations.map(conversation => {
          const active = activeId === conversation.conversation_id
          return (
            <div key={conversation.conversation_id} className={`group flex items-center gap-0.5 rounded-full pl-1 pr-1 transition-colors duration-200 ${active ? 'bg-white text-black' : 'text-night-body hover:bg-night-2'}`}>
              {editingId === conversation.conversation_id ? (
                <form className="flex min-w-0 flex-1 items-center gap-1 py-1" onSubmit={async event => {
                  event.preventDefault()
                  if (!title.trim()) return
                  try { await onRename(conversation.conversation_id, title.trim()); setEditingId(null) }
                  catch { return }
                }}>
                  <input aria-label="Conversation title" autoFocus maxLength={80} value={title} onChange={e => setTitle(e.target.value)} className={`min-w-0 flex-1 rounded-full border px-3 py-1 text-[13px] outline-none ${active ? 'border-black bg-white text-black' : 'border-purple bg-night-2 text-white'}`} />
                  <button type="submit" aria-label="Save title" disabled={!title.trim() || busy} className="grid h-7 w-7 place-items-center rounded-full hover:bg-[rgba(127,127,140,.2)] disabled:opacity-40"><Check size={14} /></button>
                  <button type="button" aria-label="Cancel rename" onClick={() => setEditingId(null)} className="grid h-7 w-7 place-items-center rounded-full opacity-70 hover:bg-[rgba(127,127,140,.2)]"><X size={14} /></button>
                </form>
              ) : (
                <>
                  <button type="button" disabled={busy} aria-current={active ? 'page' : undefined}
                    className={`min-w-0 flex-1 truncate py-2 pl-2.5 pr-1 text-left text-[13.5px] ${active ? 'font-semibold' : ''}`} onClick={() => onOpen(conversation.conversation_id)}>
                    <span title={conversation.title}>{conversation.title}</span>
                  </button>
                  <button type="button" title="Rename chat" aria-label={`Rename ${conversation.title}`} disabled={busy} className="grid h-7 w-7 place-items-center rounded-full opacity-60 transition hover:bg-[rgba(127,127,140,.2)] hover:opacity-100 focus-visible:opacity-100 group-hover:opacity-100" onClick={() => { setEditingId(conversation.conversation_id); setTitle(conversation.title) }}><Pencil size={13} /></button>
                  <button type="button" title="Delete chat" aria-label={`Delete ${conversation.title}`} disabled={busy} className="grid h-7 w-7 place-items-center rounded-full opacity-60 transition hover:bg-[rgba(127,127,140,.2)] hover:text-[#FF6B6B] hover:opacity-100 focus-visible:opacity-100 group-hover:opacity-100" onClick={() => {
                    if (window.confirm(`Delete “${conversation.title}” and its messages?`)) void onDelete(conversation.conversation_id)
                  }}><Trash2 size={13} /></button>
                </>
              )}
            </div>
          )
        })}
      </div>
    </section>
  )
}
