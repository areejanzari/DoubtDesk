import { apiRequest } from '../auth/api'

export type ConversationSummary = {
  conversation_id: string
  title: string
  created_at: string
  last_accessed_ns: number
  message_count: number
}

export type SavedExchange = {
  id: number
  question: string
  answer: string
  formatted_answer: string
  status: string
  sources: unknown
  doc_ids: string[]
  created_at: string
}

export type ConversationDetail = Omit<ConversationSummary, 'message_count'> & {
  messages: SavedExchange[]
}

export const conversationApi = {
  list: async (onUnauthorized: () => void): Promise<ConversationSummary[]> => {
    const data = await apiRequest<unknown>('/conversations', {}, undefined, onUnauthorized)
    if (!Array.isArray(data) || !data.every(item =>
      item !== null && typeof item === 'object' &&
      typeof item.conversation_id === 'string' && typeof item.title === 'string'
    )) {
      throw new Error('Unexpected conversation list response')
    }
    return data as ConversationSummary[]
  },
  create: (csrfToken: string, onUnauthorized: () => void) =>
    apiRequest<ConversationSummary>('/conversations', { method: 'POST' }, csrfToken, onUnauthorized),
  get: (id: string, onUnauthorized: () => void) =>
    apiRequest<ConversationDetail>(`/conversations/${encodeURIComponent(id)}`, {}, undefined, onUnauthorized),
  rename: (id: string, title: string, csrfToken: string, onUnauthorized: () => void) =>
    apiRequest<ConversationSummary>(`/conversations/${encodeURIComponent(id)}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title }),
    }, csrfToken, onUnauthorized),
  remove: (id: string, csrfToken: string, onUnauthorized: () => void) =>
    apiRequest<void>(`/conversations/${encodeURIComponent(id)}`, { method: 'DELETE' }, csrfToken, onUnauthorized),
}
