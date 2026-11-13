import { useState } from 'react'
import { LoaderCircle, MessageSquare, Pencil, Plus, Trash2, X, Check } from 'lucide-react'
import type { ConversationSummary } from './conversations'

export function ConversationSidebar({
  conversations, activeId, busy, loading, error, onNew, onOpen, onRename, onDelete,
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
}) {
  const [editingId, setEditingId] = useState<string | null>(null)
  const [title, setTitle] = useState('')
  const atLimit = conversations.length >= 30

  return (
    <section aria-label="Conversation history" className="border-b border-slate-200 pb-4">
      <button type="button" onClick={onNew} disabled={busy || atLimit}
        className="mb-3 flex w-full items-center justify-center gap-2 rounded-xl bg-[#4169FF] px-3 py-2 text-sm font-medium text-white disabled:opacity-50">
        <Plus size={16} /> New Chat
      </button>
      <div className="mb-2 flex items-center justify-between px-1">
        <span className="text-xs font-semibold uppercase tracking-wide text-slate-500">Recents</span>
        <span className="text-xs text-slate-500">{conversations.length}/30</span>
      </div>
      {atLimit && <p role="status" className="mb-2 px-1 text-xs text-amber-700">30-chat limit reached. Delete a conversation to start another.</p>}
      {error && <p role="alert" className="mb-2 break-words px-1 text-xs text-red-600">{error}</p>}
      {loading && <p className="flex items-center gap-2 px-1 text-xs text-slate-500"><LoaderCircle className="animate-spin" size={14} /> Loading chats…</p>}
      <div className="max-h-52 space-y-1 overflow-y-auto" aria-label="Recent chats">
        {!loading && conversations.length === 0 && <p className="px-1 py-2 text-xs text-slate-400">No saved chats yet.</p>}
        {conversations.map(conversation => (
          <div key={conversation.conversation_id} className={`group flex items-center gap-1 rounded-lg px-1 py-1 ${activeId === conversation.conversation_id ? 'bg-blue-100' : 'hover:bg-white'}`}>
            {editingId === conversation.conversation_id ? (
              <form className="flex min-w-0 flex-1 items-center gap-1" onSubmit={async event => {
                event.preventDefault()
                if (!title.trim()) return
                try { await onRename(conversation.conversation_id, title.trim()); setEditingId(null) }
                catch { return }
              }}>
                <input aria-label="Conversation title" autoFocus maxLength={80} value={title} onChange={e => setTitle(e.target.value)} className="min-w-0 flex-1 rounded border border-blue-300 bg-white px-1 py-1 text-xs" />
                <button type="submit" aria-label="Save title" disabled={!title.trim() || busy}><Check size={14} /></button>
                <button type="button" aria-label="Cancel rename" onClick={() => setEditingId(null)}><X size={14} /></button>
              </form>
            ) : (
              <>
                <button type="button" disabled={busy} aria-current={activeId === conversation.conversation_id ? 'page' : undefined}
                  className="flex min-w-0 flex-1 items-center gap-2 py-1 text-left text-xs text-slate-700" onClick={() => onOpen(conversation.conversation_id)}>
                  <MessageSquare size={14} className="shrink-0" />
                  <span className="truncate" title={conversation.title}>{conversation.title}</span>
                </button>
                <button type="button" title="Rename chat" aria-label={`Rename ${conversation.title}`} disabled={busy} className="p-1 text-slate-500 hover:text-blue-600" onClick={() => { setEditingId(conversation.conversation_id); setTitle(conversation.title) }}><Pencil size={13} /></button>
                <button type="button" title="Delete chat" aria-label={`Delete ${conversation.title}`} disabled={busy} className="p-1 text-slate-500 hover:text-red-600" onClick={() => {
                  if (window.confirm(`Delete “${conversation.title}” and its messages?`)) void onDelete(conversation.conversation_id)
                }}><Trash2 size={13} /></button>
              </>
            )}
          </div>
        ))}
      </div>
    </section>
  )
}
