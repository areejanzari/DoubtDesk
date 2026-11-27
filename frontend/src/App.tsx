import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { ChangeEvent, CSSProperties, KeyboardEvent, ReactNode } from 'react'
import ReactMarkdown from 'react-markdown'
import type { Components } from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import {
  ArrowUp, Check, ChevronLeft,
  ChevronRight, Clipboard, FileText, LoaderCircle,
  Menu, PanelLeftClose, PanelLeftOpen, Paperclip, Upload,
  Search, X, ZoomIn, ZoomOut, Maximize2, LogOut, Moon, Sun,
} from 'lucide-react'
import { LogoMark, Wordmark } from './brand/Logo'
import { Blob3D, Squiggle, Star3D, SunDoodle, Words } from './brand/Shapes'
import 'katex/dist/katex.min.css'
import { apiRequest, authApi } from './auth/api'
import { conversationApi } from './chat/conversations'
import type { ConversationSummary, ConversationDetail } from './chat/conversations'
import { ConversationSidebar } from './chat/ConversationSidebar'
import { streamAsk } from './chat/stream'
import { normalizeRawCitations, linkSourceReferences, extractSources, sourcesFromAnswer } from './chat/citations'
import type { AuthUser } from './auth/api'
import { AuthScreen } from './auth/AuthScreen'

const PAGE_IMAGE_TEMPLATE = import.meta.env.VITE_PAGE_IMAGE_URL_TEMPLATE || '/documents/{doc_id}/pages/{page}/image'

type DocumentInfo = { doc_id: string; filename: string; page_count: number; unreadable_pages?: number }
type UploadResult = { doc_id: string; filename: string; status: string; page_count: number }
type DocumentStatus = { doc_id: string; status: string; page_count: number; unreadable_pages?: number }
type Usage = { plan: string; questions_used: number; questions_limit: number | null; window_resets_at: number | null }
const UPLOAD_POLL_MS = 2000
const UPLOAD_POLL_LIMIT_MS = 15 * 60 * 1000
const stageLabels: Record<string, string> = {
  preparing: 'Preparing…', generating: 'Generating…', verifying: 'Verifying…', checking: 'Verifying…',
  revising: 'Revising…', fallback: 'Revising…',
}
const unreadableNote = (count?: number) => count ? ` ${count} ${count === 1 ? 'page' : 'pages'} couldn't be read.` : ''
const sleep = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms))
const THEME_KEY = 'doubtdesk-theme'
const SUGGESTIONS = ['Summarise the key points', 'Explain this with an example', 'Quiz me on this topic']
type Theme = 'light' | 'dark'
function readTheme(): Theme | null {
  try {
    const saved = window.localStorage.getItem(THEME_KEY)
    return saved === 'light' || saved === 'dark' ? saved : null
  } catch {
    return null
  }
}
function systemPrefersDark() {
  return typeof window.matchMedia === 'function' && window.matchMedia('(prefers-color-scheme: dark)').matches
}
type AnswerStatus = 'grounded' | 'general_knowledge' | 'out_of_scope' | 'verification_failed' | 'unknown'
type Source = { docId?: string; filename: string; page: number; excerpt?: string }
type Message = {
  id: number
  role: 'user' | 'assistant'
  content: string
  status?: AnswerStatus
  sources?: Source[]
  error?: boolean
  pending?: boolean
  revising?: boolean
  settled?: boolean
}
type AskResponse = {
  status: AnswerStatus
  answer: string
  formatted_answer: string
  sources: unknown
  conversation_id: string | null
}

const REMARK_PLUGINS = [remarkGfm, remarkMath]
const REHYPE_PLUGINS = [rehypeKatex]

function MarkdownContent({ content, sources = [], onSourceOpen }: {
  content: string
  sources?: Source[]
  onSourceOpen?: (source: Source) => void
}) {
  const openRef = useRef(onSourceOpen)
  useEffect(() => { openRef.current = onSourceOpen }, [onSourceOpen])
  const components = useMemo<Components>(() => ({
    a: ({ href, children }) => {
      const sourceMatch = /^#doubtdesk-source-(\d+)$/.exec(href || '')
      const source = sourceMatch ? sources[Number(sourceMatch[1]) - 1] : undefined
      if (source) return <button type="button" aria-label={`Open source ${sourceMatch![1]}`} onClick={() => openRef.current?.(source)} className="cite-marker"><span className="cite-bracket">[</span>{sourceMatch![1]}<span className="cite-bracket">]</span></button>
      return <a href={href} target="_blank" rel="noreferrer">{children}</a>
    },
    img: ({ src, alt }) => <img src={src} alt={alt || 'Provided illustration'} loading="lazy" />,
  }), [sources])
  const normalized = sources.length ? normalizeRawCitations(content, sources) : content
  const linked = linkSourceReferences(normalized, sources.length)
  return (
    <div className="markdown-body min-w-0 break-words">
      <ReactMarkdown remarkPlugins={REMARK_PLUGINS} rehypePlugins={REHYPE_PLUGINS} components={components}>{linked}</ReactMarkdown>
    </div>
  )
}

function IconButton({ label, onClick, children, disabled = false }: {
  label: string; onClick: () => void; children: ReactNode; disabled?: boolean
}) {
  return <button type="button" title={label} aria-label={label} onClick={onClick} disabled={disabled}
    className="icon-button">{children}</button>
}

function TutorApp({ user, onUnauthorized, onLogout }: {
  user: AuthUser
  onUnauthorized: () => void
  onLogout: () => Promise<void>
}) {
  const [documents, setDocuments] = useState<DocumentInfo[]>([])
  const [selectedDocIds, setSelectedDocIds] = useState<string[]>([])
  const [documentSearch, setDocumentSearch] = useState('')
  const [subject, setSubject] = useState('All')
  const [question, setQuestion] = useState('')
  const [messages, setMessages] = useState<Message[]>([])
  const [conversationId, setConversationId] = useState<string | null>(null)
  const [conversations, setConversations] = useState<ConversationSummary[]>([])
  const [historyLoading, setHistoryLoading] = useState(true)
  const [historyError, setHistoryError] = useState('')
  const [answerStage, setAnswerStage] = useState('Thinking…')
  const [streamingId, setStreamingId] = useState<number | null>(null)
  const [isAnswering, setIsAnswering] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [uploadMessage, setUploadMessage] = useState('')
  const [documentError, setDocumentError] = useState('')
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false)
  const [viewerSource, setViewerSource] = useState<Source | null>(null)
  const [viewerPage, setViewerPage] = useState(1)
  const [zoom, setZoom] = useState(1)
  const [viewerFullscreen, setViewerFullscreen] = useState(false)
  const [copiedId, setCopiedId] = useState<number | null>(null)
  const [imageUrl, setImageUrl] = useState<string | null>(null)
  const [imageError, setImageError] = useState('')
  const [loggingOut, setLoggingOut] = useState(false)
  const [logoutError, setLogoutError] = useState('')
  const [usage, setUsage] = useState<Usage | null>(null)
  const [usageResetLabel, setUsageResetLabel] = useState('')
  const [theme, setTheme] = useState<Theme | null>(readTheme)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const questionRef = useRef<HTMLTextAreaElement>(null)
  const mountedRef = useRef(true)
  const chatEndRef = useRef<HTMLDivElement>(null)
  const messageIdRef = useRef(0)
  const nextId = () => ++messageIdRef.current

  const loadDocuments = useCallback(async () => {
    try {
      const data = await apiRequest<unknown>('/documents', {}, undefined, onUnauthorized)
      if (!Array.isArray(data)) throw new Error('Unexpected document list response')
      setDocuments(data as DocumentInfo[])
      setDocumentError('')
    } catch (error) {
      setDocumentError(error instanceof Error ? error.message : 'Unable to load documents')
    }
  }, [onUnauthorized])

  const loadConversations = useCallback(async () => {
    try {
      const recent = await conversationApi.list(onUnauthorized)
      setConversations(recent)
      setHistoryError('')
    } catch (error) {
      setHistoryError(error instanceof Error ? error.message : 'Unable to load conversations')
    } finally {
      setHistoryLoading(false)
    }
  }, [onUnauthorized])

  const loadUsage = useCallback(async () => {
    try {
      const data = await apiRequest<Usage>('/usage')
      if (!data || typeof data.plan !== 'string' || typeof data.questions_used !== 'number') return
      const hours = data.window_resets_at ? Math.ceil((data.window_resets_at * 1000 - Date.now()) / 3_600_000) : null
      setUsageResetLabel(hours === null ? '' : hours <= 1 ? 'resets within 1h' : `resets in ${hours}h`)
      setUsage(data)
    } catch {
      setUsage(null)
    }
  }, [])

  function loadSavedMessages(history: ConversationDetail) {
    const restored: Message[] = []
    for (const exchange of history.messages) {
      restored.push({ id: nextId(), role: 'user', content: exchange.question })
      const status: AnswerStatus = ['grounded', 'general_knowledge', 'out_of_scope', 'verification_failed'].includes(exchange.status)
        ? exchange.status as AnswerStatus : 'unknown'
      const sources = status === 'grounded'
        ? extractSources(exchange.sources, documents)
        : []
      restored.push({
        id: nextId(), role: 'assistant', content: exchange.answer, status,
        sources: sources.length ? sources : status === 'grounded'
          ? sourcesFromAnswer(exchange.formatted_answer, documents) : [],
      })
    }
    setMessages(restored)
    setConversationId(history.conversation_id)
    setSelectedDocIds((history.messages.at(-1)?.doc_ids || []).filter(id => documents.some(doc => doc.doc_id === id)))
    setViewerSource(null)
    setMobileSidebarOpen(false)
  }

  async function openConversation(id: string) {
    if (isAnswering) return
    try {
      const detail = await conversationApi.get(id, onUnauthorized)
      loadSavedMessages(detail)
      setHistoryError('')
    } catch (error) {
      setHistoryError(error instanceof Error ? error.message : 'Unable to open this chat')
    }
  }

  async function startConversation() {
    if (isAnswering || conversations.length >= 30) return
    try {
      const created = await conversationApi.create(user.csrf_token, onUnauthorized)
      setConversationId(created.conversation_id)
      setMessages([])
      setQuestion('')
      setViewerSource(null)
      setSelectedDocIds([])
      setConversations(old => [created, ...old])
      setHistoryError('')
      setMobileSidebarOpen(false)
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unable to create a chat'
      await loadConversations()
      setHistoryError(message)
    }
  }

  async function renameConversation(id: string, title: string) {
    try {
      const updated = await conversationApi.rename(id, title, user.csrf_token, onUnauthorized)
      setConversations(old => old.map(item => item.conversation_id === id ? updated : item))
      setHistoryError('')
    } catch (error) {
      setHistoryError(error instanceof Error ? error.message : 'Unable to rename chat')
      throw error
    }
  }

  async function deleteConversation(id: string) {
    try {
      await conversationApi.remove(id, user.csrf_token, onUnauthorized)
      setConversations(old => old.filter(item => item.conversation_id !== id))
      if (conversationId === id) {
        setConversationId(null)
        setMessages([])
        setQuestion('')
        setSelectedDocIds([])
      }
      setHistoryError('')
    } catch (error) {
      setHistoryError(error instanceof Error ? error.message : 'Unable to delete chat')
    }
  }

  useEffect(() => { void loadConversations() }, [loadConversations])
  useEffect(() => { void loadDocuments() }, [loadDocuments])
  useEffect(() => { void loadUsage() }, [loadUsage])
  useEffect(() => { mountedRef.current = true; return () => { mountedRef.current = false } }, [])
  useEffect(() => { chatEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' }) }, [messages, isAnswering])

  async function uploadFile(file: File) {
    if (!file.name.toLowerCase().endsWith('.pdf')) { setUploadMessage('Please choose a PDF file.'); return }
    if (file.size === 0) { setUploadMessage('This PDF is empty.'); return }
    if (file.size > 25 * 1024 * 1024) { setUploadMessage('PDF exceeds the 25 MB limit.'); return }
    setUploading(true)
    setUploadMessage(`Processing ${file.name}…`)
    try {
      const form = new FormData()
      form.append('file', file)
      const result = await apiRequest<UploadResult>(
        '/documents/upload', { method: 'POST', body: form }, user.csrf_token, onUnauthorized,
      )
      if (result.status === 'processing') {
        const final = await waitForDocument(result.doc_id)
        if (!final) return
        if (final.status === 'failed') throw new Error(`${result.filename}: processing failed. Please try uploading again.`)
        setUploadMessage(`${result.filename}: uploaded (${final.page_count} pages).${unreadableNote(final.unreadable_pages)}`)
      } else {
        setUploadMessage(result.status === 'cached' ? `${result.filename}: cached document reused.` : `${result.filename}: uploaded (${result.page_count} pages).`)
      }
      await loadDocuments()
    } catch (error) {
      if (mountedRef.current) setUploadMessage(error instanceof Error ? error.message : 'Upload failed')
    } finally {
      if (mountedRef.current) setUploading(false)
    }
  }

  async function waitForDocument(docId: string): Promise<DocumentStatus | null> {
    const deadline = Date.now() + UPLOAD_POLL_LIMIT_MS
    while (Date.now() < deadline) {
      await sleep(UPLOAD_POLL_MS)
      if (!mountedRef.current) return null
      const status = await apiRequest<DocumentStatus>(`/documents/${encodeURIComponent(docId)}/status`, {}, undefined, onUnauthorized)
      if (status.status === 'completed' || status.status === 'failed') return status
    }
    throw new Error('PDF processing is taking longer than expected. It will appear in your documents when ready.')
  }

  function onFileChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0]
    if (file) void uploadFile(file)
    event.target.value = ''
  }

  async function submitQuestion() {
    const text = question.trim()
    if (!text || isAnswering) return
    if (!conversationId && conversations.length >= 30) {
      setHistoryError('30-chat limit reached. Delete a conversation to start another.')
      return
    }
    const userMessageId = nextId()
    const assistantMessageId = nextId()
    setQuestion('')
    setMessages(old => [...old, { id: userMessageId, role: 'user', content: text }])
    setIsAnswering(true)
    setAnswerStage('Preparing…')
    setStreamingId(assistantMessageId)
    const updateAssistant = (change: (message: Message) => Message) => setMessages(old => {
      const existing = old.find(message => message.id === assistantMessageId)
      return existing
        ? old.map(message => message.id === assistantMessageId ? change(message) : message)
        : [...old, change({ id: assistantMessageId, role: 'assistant', content: '' })]
    })
    try {
      await streamAsk({
        question: text,
        doc_ids: selectedDocIds,
        ...(conversationId ? { conversation_id: conversationId } : {}),
      }, user.csrf_token, onUnauthorized, ({ event, data }) => {
        if (event === 'status') {
          const stage = typeof data.stage === 'string' ? data.stage : 'generating'
          setAnswerStage(stageLabels[stage] || 'Generating…')
          if (stage === 'verifying') updateAssistant(message => ({ ...message, pending: true }))
          if (stage === 'revising' || stage === 'fallback') setMessages(old => old.map(message => message.id === assistantMessageId && message.pending ? { ...message, revising: true } : message))
        }
        if (event === 'reset') updateAssistant(message => ({ ...message, content: '', pending: true, revising: true }))
        if (event === 'token' && typeof data.text === 'string') {
          const token = data.text
          updateAssistant(message => ({ ...message, content: message.content + token }))
        }
        if (event === 'complete') {
          const result = data as unknown as AskResponse
          if (typeof result.answer !== 'string') throw new Error('Unexpected answer response')
          if (result.conversation_id) setConversationId(result.conversation_id)
          const status: AnswerStatus = ['grounded', 'general_knowledge', 'out_of_scope', 'verification_failed'].includes(result.status)
            ? result.status : 'unknown'
          const parsedSources = status === 'grounded' ? extractSources(result.sources, documents) : []
          const sources = status === 'grounded'
            ? parsedSources.length ? parsedSources : sourcesFromAnswer(result.formatted_answer || '', documents)
            : []
          setMessages(old => {
            const previous = old.find(message => message.id === assistantMessageId)
            const saved: Message = { id: assistantMessageId, role: 'assistant', content: result.answer, status, sources, settled: Boolean(previous?.pending) }
            return previous
              ? old.map(message => message.id === assistantMessageId ? saved : message)
              : [...old, saved]
          })
        }
      })
      await loadConversations()
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Something went wrong.'
      setMessages(old => {
        const failure: Message = { id: assistantMessageId, role: 'assistant', content: message, error: true }
        return old.some(item => item.id === assistantMessageId)
          ? old.map(item => item.id === assistantMessageId ? failure : item)
          : [...old, failure]
      })
      await loadConversations()
    } finally {
      setIsAnswering(false)
      setStreamingId(null)
      setAnswerStage('Thinking…')
      void loadUsage()
    }
  }

  function onQuestionKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault()
      void submitQuestion()
    }
  }

  function openSource(source: Source) {
    setViewerSource(source)
    setViewerPage(source.page)
    setZoom(1)
    setViewerFullscreen(false)
  }

  const activeViewerDoc = viewerSource && documents.find(d => d.doc_id === viewerSource.docId)
  const viewerDocId = viewerSource?.docId
  const pageImagePath = viewerSource && viewerDocId && PAGE_IMAGE_TEMPLATE
    ? PAGE_IMAGE_TEMPLATE.replace('{doc_id}', encodeURIComponent(viewerDocId)).replace('{page}', String(viewerPage))
    : null

  useEffect(() => {
    setImageUrl(null)
    setImageError('')
    if (!pageImagePath) return
    const controller = new AbortController()
    let objectUrl: string | null = null
    void apiRequest<Blob>(pageImagePath, { signal: controller.signal }, undefined, onUnauthorized)
      .then((blob) => {
        if (controller.signal.aborted) return
        if (!(blob instanceof Blob)) throw new Error('Invalid PDF page image')
        objectUrl = URL.createObjectURL(blob)
        setImageUrl(objectUrl)
      })
      .catch((error: unknown) => {
        if (!controller.signal.aborted) setImageError(error instanceof Error ? error.message : 'Image unavailable')
      })
    return () => {
      controller.abort()
      if (objectUrl) URL.revokeObjectURL(objectUrl)
    }
  }, [pageImagePath, onUnauthorized])

  async function signOut() {
    if (loggingOut) return
    setLoggingOut(true)
    setLogoutError('')
    try {
      await onLogout()
    } catch (error) {
      setLogoutError(error instanceof Error ? error.message : 'Unable to sign out')
    } finally {
      setLoggingOut(false)
    }
  }
  const darkMode = theme ? theme === 'dark' : systemPrefersDark()
  function toggleTheme() {
    const next: Theme = darkMode ? 'light' : 'dark'
    setTheme(next)
    try { window.localStorage.setItem(THEME_KEY, next) } catch { return }
  }
  function applySuggestion(text: string) {
    setQuestion(text)
    questionRef.current?.focus()
  }
  const visibleDocs = documents.filter((doc) => {
    const matchesSearch = doc.filename.toLowerCase().includes(documentSearch.toLowerCase())
    // Filename-based subjects are only a UI filter; backend has no subject metadata.
    const matchesSubject = subject === 'All' || doc.filename.toLowerCase().includes(subject.toLowerCase())
    return matchesSearch && matchesSubject
  })
  const usagePercent = usage && usage.questions_limit ? Math.min(100, Math.round((usage.questions_used / usage.questions_limit) * 100)) : 0

  const streamingVisible = streamingId !== null && messages.some(message => message.id === streamingId && (message.pending || message.content))
  const statusStamp: Partial<Record<AnswerStatus, { className: string; label: string }>> = {
    grounded: { className: 'stamp--grounded', label: 'From documents' },
    general_knowledge: { className: 'stamp--general', label: 'Not from document' },
    verification_failed: { className: 'stamp--failed', label: 'Verification unavailable' },
    out_of_scope: { className: 'stamp--scope', label: 'Outside academic scope' },
  }

  return (
    <div className="app-shell flex h-dvh min-h-0 gap-3 overflow-hidden p-2 sm:p-3" data-theme={theme ?? undefined}>
      {mobileSidebarOpen && <button aria-label="Close navigation" className="fixed inset-0 z-30 bg-black/40 backdrop-blur-[2px] lg:hidden" onClick={() => setMobileSidebarOpen(false)} />}
      <aside className={`sidebar ${mobileSidebarOpen ? 'sidebar--mobile-open' : ''} ${sidebarOpen ? '' : 'sidebar--collapsed'}`}>
        <div className="flex h-full w-[296px] flex-col overflow-hidden px-4 pb-4 pt-[22px]">
          <div className="mb-[22px] flex items-center justify-between pl-1.5">
            <Wordmark size={32} tone="dark" variant="once" textClassName="text-lg" />
            <IconButton label="Close sidebar" onClick={() => { setSidebarOpen(false); setMobileSidebarOpen(false) }}><PanelLeftClose size={18} /></IconButton>
          </div>
          <input ref={fileInputRef} type="file" accept=".pdf,application/pdf" className="hidden" onChange={onFileChange} />
          <ConversationSidebar conversations={conversations} activeId={conversationId} busy={isAnswering}
            loading={historyLoading} error={historyError} onNew={() => void startConversation()}
            onOpen={id => void openConversation(id)} onRename={renameConversation} onDelete={deleteConversation}
            actions={<button className="side-button side-button--dark" disabled={uploading} onClick={() => fileInputRef.current?.click()}>
              {uploading ? <LoaderCircle className="animate-spin" size={15} /> : <Upload size={15} />}{uploading ? 'Processing…' : 'Upload PDF'}
            </button>} />
          {uploadMessage && <p role="status" className="rise mt-2 rounded-2xl bg-night-2 px-3 py-2.5 text-xs leading-5 text-night-body">{uploadMessage}</p>}
          <div className="mb-2 mt-5 flex items-baseline justify-between px-2"><span className="night-label">Documents</span><span className="text-[11.5px] tabular-nums text-[#7E7E88]">{documents.length}</span></div>
          <div className="relative mb-2.5"><Search size={15} className="pointer-events-none absolute left-3.5 top-[12px] text-[#7E7E88]" /><input value={documentSearch} onChange={(e) => setDocumentSearch(e.target.value)} placeholder="Search documents" className="night-field h-10 w-full pl-10 pr-4 text-[13.5px]" /></div>
          <div className="mb-2 flex flex-wrap gap-1.5 px-0.5">{['All', 'Biology', 'Chemistry', 'Physics'].map((name) => <button key={name} onClick={() => setSubject(name)} className={`rounded-full border px-3 py-[5px] text-[12.5px] transition-colors duration-200 ${subject === name ? 'border-purple bg-purple font-semibold text-white' : 'border-night-rule text-[#A9A9B2] hover:text-white'}`}>{name}</button>)}</div>
          {documentError && <p role="alert" className="mb-2 px-2 text-xs text-[#FF8A8A]">{documentError} <button className="underline" onClick={() => void loadDocuments()}>Retry</button></p>}
          <div className="min-h-0 flex-1 space-y-0.5 overflow-y-auto pr-1">
            {visibleDocs.length === 0 && <p className="px-2 py-5 text-center text-xs leading-5 text-[#7E7E88]">{documents.length ? 'No matching PDFs.' : 'Upload a PDF to get started.'}</p>}
            {visibleDocs.map((doc) => {
              const selected = selectedDocIds.includes(doc.doc_id)
              return <label key={doc.doc_id} className={`flex cursor-pointer items-start gap-[11px] rounded-[18px] px-2.5 py-2.5 transition-colors duration-200 ${selected ? 'bg-purple-night' : 'hover:bg-night-2'}`}>
                <input type="checkbox" className="doc-check" checked={selected} onChange={() => setSelectedDocIds((old) => selected ? old.filter((id) => id !== doc.doc_id) : [...old, doc.doc_id])} />
                <span className="min-w-0"><span className={`block break-words text-[13.5px] leading-5 ${selected ? 'font-semibold text-white' : 'text-[#E6E6EA]'}`}>{doc.filename}</span><span className="text-[11.5px] text-[#7E7E88]">{doc.page_count} pages</span>{!!doc.unreadable_pages && <span className="block text-[11.5px] text-[#7E7E88]">{doc.unreadable_pages} {doc.unreadable_pages === 1 ? 'page' : 'pages'} couldn't be read</span>}</span>
              </label>
            })}
          </div>
          {usage && <section aria-label="Plan usage" className="mt-3 rounded-[22px] bg-night-2 px-4 py-3.5">
            <div className="flex items-baseline justify-between"><span className="night-label">{usage.plan.charAt(0).toUpperCase() + usage.plan.slice(1)} plan</span>{usage.questions_limit !== null && usageResetLabel && <span className="text-[11.5px] text-[#7E7E88]">{usageResetLabel}</span>}</div>
            {usage.questions_limit === null
              ? <p className="mt-2 text-[13.5px] text-night-body">No daily limit</p>
              : <>
                <p className="mt-2 text-[13.5px] text-night-body"><strong className="font-semibold tabular-nums text-white">{usage.questions_used}</strong> of {usage.questions_limit} questions today</p>
                <span className="mt-2.5 block h-[7px] overflow-hidden rounded-full bg-night-rule" aria-hidden="true"><span className={`block h-full rounded-full transition-[width] duration-700 ${usagePercent >= 90 ? 'bg-danger' : 'bg-[linear-gradient(90deg,#A64DFF,#E28BFF)]'}`} style={{ width: `${usagePercent}%` }} /></span>
              </>}
          </section>}
        </div>
      </aside>

      <section className="rise relative flex min-w-0 flex-1 flex-col overflow-hidden rounded-[24px] bg-surface sm:rounded-[30px]">
        <header className="flex min-h-[70px] items-center justify-between gap-3 border-b border-rule px-3.5 sm:px-5">
          <div className="flex min-w-0 items-center gap-2.5">
            <button type="button" title="Toggle navigation" aria-label="Toggle navigation" className="icon-button ring-button" onClick={() => { if (window.innerWidth < 1024) setMobileSidebarOpen(true); else setSidebarOpen((old) => !old) }}><Menu className="lg:hidden" size={19} /><PanelLeftOpen className="hidden lg:block" size={18} /></button>
            <div className="min-w-0"><h1 className="text-lg font-semibold tracking-[-0.015em]">AI Tutor</h1><p className="hidden truncate text-[12.5px] text-muted sm:block">Ask questions about your uploaded documents.</p></div>
          </div>
          <div className="flex items-center gap-2">
            <span key={selectedDocIds.length} className={`stamp hidden !px-3.5 !py-[7px] !text-[12.5px] sm:inline-flex ${selectedDocIds.length ? 'stamp--grounded' : 'stamp--scope'}`}>{selectedDocIds.length ? `${selectedDocIds.length} PDF${selectedDocIds.length === 1 ? '' : 's'} selected` : 'General knowledge'}</span>
            <span className="hidden items-center gap-2 rounded-full bg-chip py-1 pl-1 pr-3.5 text-[13px] md:flex" title={user.email}><span aria-hidden="true" className="grid h-[30px] w-[30px] place-items-center rounded-full bg-[linear-gradient(135deg,#A64DFF,#FF8A1F)] text-xs font-semibold uppercase text-white">{user.username.slice(0, 2)}</span><span className="max-w-32 truncate">{user.username}</span></span>
            <button type="button" className="icon-button ring-button" title={darkMode ? 'Switch to light mode' : 'Switch to dark mode'} aria-label={darkMode ? 'Switch to light mode' : 'Switch to dark mode'} onClick={toggleTheme}>{darkMode ? <Sun size={17} /> : <Moon size={17} />}</button>
            <button type="button" className="icon-button ring-button" title="Sign out" aria-label="Sign out" onClick={() => void signOut()} disabled={loggingOut}>{loggingOut ? <LoaderCircle size={17} className="animate-spin" /> : <LogOut size={17} />}</button>
            {viewerSource && <IconButton label="Close source viewer" onClick={() => setViewerSource(null)}><X size={18} /></IconButton>}
          </div>
        </header>
        {logoutError && <div role="alert" className="border-b border-rule bg-danger-soft px-5 py-2 text-sm text-danger">{logoutError}</div>}
        <div className="min-h-0 flex-1 overflow-y-auto px-3.5 pb-44 pt-8 sm:px-6" aria-live="polite">
          <div className="mx-auto flex w-full max-w-[760px] flex-col gap-9">
            {messages.length === 0 && <div className="rise relative px-1 pb-8 pt-12 sm:pt-16">
              <Star3D className="float right-0 top-0 hidden w-[110px] sm:block" style={{ '--r': '-12deg' } as CSSProperties} />
              <Blob3D className="float float-2 right-2.5 top-36 hidden w-[64px] sm:block" />
              <SunDoodle className="-left-1.5 -top-2 w-[64px] [animation:dd-pop-in_.8s_.3s_both]" />
              <p className="relative z-[2] inline-flex items-center gap-2 rounded-full bg-chip px-3.5 py-1.5 text-[12.5px] font-medium text-muted">{selectedDocIds.length ? <span className="pill-dot text-lime" /> : null}{selectedDocIds.length ? `${selectedDocIds.length} document${selectedDocIds.length === 1 ? '' : 's'} ticked` : 'No documents ticked'}</p>
              <h2 className="relative z-[2] mt-5 max-w-[560px] text-balance text-[clamp(36px,5vw,60px)] font-semibold leading-[1.06] tracking-[-0.04em]"><Words text="What would you like to" start={100} step={80} /> <span className="relative inline-block"><Words text="learn?" start={500} /><Squiggle delay={0.8} /></span></h2>
              <p className="relative z-[2] mt-4 max-w-[470px] text-base leading-relaxed text-muted">Select one or more PDFs to ask document-grounded questions, or ask anything without selecting a document.</p>
              <div className="relative z-[2] mt-6 flex flex-wrap gap-2">{SUGGESTIONS.map((text, index) => <button key={text} type="button" onClick={() => applySuggestion(text)} className="rise rounded-full border border-rule px-4 py-2.5 text-[13.5px] text-body transition hover:-translate-y-0.5 hover:border-ink hover:text-ink" style={{ animationDelay: `${600 + index * 90}ms` }}>{text}</button>)}</div>
            </div>}
            {messages.map((message) => <article key={message.id} className={message.role === 'user' ? 'flex flex-col items-end' : 'rise'} aria-busy={message.pending ? true : undefined}>
              {message.role === 'user' && <div className="you-in max-w-[90%] rounded-[26px_26px_8px_26px] bg-you px-5 py-3.5 text-[15.5px] leading-relaxed text-white sm:max-w-[78%]"><p className="whitespace-pre-wrap break-words">{message.content}</p></div>}
              {message.role === 'assistant' && <div className="min-w-0 max-w-full">
                <div className="mb-3 flex flex-wrap items-center gap-2.5">
                  <span className="flex items-center gap-2 text-[13px] font-semibold"><LogoMark size={24} tone={darkMode ? 'dark' : 'light'} />DoubtDesk</span>
                  {message.pending && <span className="stamp stamp--pending"><span className="stamp-spinner" aria-hidden="true" />{message.revising ? 'Revising' : 'Verifying'}</span>}
                  {!message.pending && message.status && statusStamp[message.status] && <span className={`stamp ${statusStamp[message.status]!.className}`}>{message.status === 'grounded' && <Check size={11} strokeWidth={3} />}{statusStamp[message.status]!.label}</span>}
                  {message.error && <span className="stamp stamp--failed">Request failed</span>}
                </div>
                <div className={message.pending ? 'answer-pending' : message.settled ? 'answer-settled' : undefined}>
                  <MarkdownContent content={message.content} sources={message.sources} onSourceOpen={openSource} />
                </div>
                {!message.pending && <div className="mt-4">
                  {!!message.sources?.length && <div className="mb-1">
                    <p className="mono-label mb-2.5">Sources</p>
                    <div>{message.sources.map((source, index) => <button key={`${source.filename}-${source.page}-${index}`} className="src-in mb-2 flex w-full items-center gap-3.5 rounded-[20px] bg-chip px-3.5 py-3 text-left transition hover:translate-x-1" style={{ animationDelay: `${index * 110}ms` }} onClick={() => openSource(source)}>
                      <span className="grid h-[30px] w-[30px] shrink-0 place-items-center rounded-full bg-purple text-xs font-semibold text-white">{index + 1}</span>
                      <span className="min-w-0 flex-1"><span className="block break-all text-[13.5px] font-semibold text-ink">{source.filename} <span className="text-[13px] font-normal text-muted">· Page {source.page}</span></span>{source.excerpt && <span className="mt-0.5 block line-clamp-2 text-[12.5px] text-muted">{source.excerpt}</span>}</span>
                    </button>)}</div>
                  </div>}
                  <button className="inline-flex items-center gap-1.5 rounded-full px-3 py-1.5 text-[12.5px] text-muted transition hover:bg-chip hover:text-ink" onClick={() => { void navigator.clipboard.writeText(message.sources?.length ? normalizeRawCitations(message.content, message.sources) : message.content).then(() => { setCopiedId(message.id); window.setTimeout(() => setCopiedId(null), 1500) }).catch(() => {}) }}>{copiedId === message.id ? <Check size={14} /> : <Clipboard size={14} />}{copiedId === message.id ? 'Copied' : 'Copy answer'}</button>
                </div>}
              </div>}
            </article>)}
            {isAnswering && !streamingVisible && <div className="rise flex items-center gap-2.5 text-[13px] text-muted"><LogoMark size={28} variant="think" tone={darkMode ? 'dark' : 'light'} />{answerStage}</div>}
            <div ref={chatEndRef} />
          </div>
        </div>
        <div className="absolute inset-x-0 bottom-0 bg-[linear-gradient(to_top,var(--color-surface)_62%,transparent)] px-3 pb-[max(.9rem,env(safe-area-inset-bottom))] pt-4 sm:px-6"><div className="mx-auto w-full max-w-[760px]">
          <div className="composer-box flex items-end gap-2 rounded-[30px] border border-composer-rule bg-composer p-2 text-white shadow-[0_18px_40px_rgba(0,0,0,.18)] transition-shadow duration-300">
            <IconButton label="Attach PDF" onClick={() => fileInputRef.current?.click()} disabled={uploading}><Paperclip size={18} /></IconButton>
            <textarea ref={questionRef} rows={1} value={question} onChange={(e) => setQuestion(e.target.value)} onKeyDown={onQuestionKeyDown} placeholder="Ask a follow-up question…" className="max-h-36 min-h-10 min-w-0 flex-1 resize-none bg-transparent px-1 py-2 text-[15px] leading-6 text-white outline-none placeholder:text-[#7E7E88]" />
            <button aria-label="Send message" disabled={!question.trim() || isAnswering} onClick={() => void submitQuestion()} className="send-button grid h-[42px] w-[42px] shrink-0 place-items-center rounded-full bg-white text-black disabled:cursor-not-allowed disabled:opacity-35">{isAnswering ? <LoaderCircle size={18} className="animate-spin" /> : <ArrowUp size={19} />}</button>
          </div>
          <p className="mt-2 text-center text-[11.5px] text-muted">Enter to send · Shift+Enter for a new line · Check important answers</p>
        </div></div>
      </section>

      {viewerSource && <aside className={`${viewerFullscreen ? 'fixed inset-0 z-50 w-full rounded-none' : 'fixed inset-2 z-40 rounded-[24px] shadow-[0_30px_80px_rgba(0,0,0,.35)] sm:inset-3 lg:static lg:z-auto lg:w-[400px] lg:shrink-0 lg:rounded-[30px] lg:shadow-none'} panel-in flex min-h-0 flex-col overflow-hidden bg-surface`}>
        <div className="flex items-center justify-between gap-2 px-4 pb-3 pt-4 sm:pl-[22px]"><div className="min-w-0"><p className="truncate text-sm font-semibold">{viewerSource.filename}</p><p className="text-xs tabular-nums text-muted">Page {viewerPage}{activeViewerDoc ? ` of ${activeViewerDoc.page_count}` : ''}</p></div><div className="flex items-center"><IconButton label="Fullscreen" onClick={() => setViewerFullscreen((v) => !v)}><Maximize2 size={16} /></IconButton><IconButton label="Close viewer" onClick={() => { setViewerSource(null); setViewerFullscreen(false) }}><X size={18} /></IconButton></div></div>
        <div className="mx-3.5 mb-3 flex items-center justify-between gap-2 rounded-full bg-chip p-1 text-[12.5px] tabular-nums text-body"><div className="flex items-center"><IconButton label="Previous page" disabled={viewerPage <= 1} onClick={() => setViewerPage((p) => Math.max(1, p - 1))}><ChevronLeft size={18} /></IconButton><span className="px-1.5">{viewerPage}{activeViewerDoc ? ` / ${activeViewerDoc.page_count}` : ''}</span><IconButton label="Next page" disabled={!activeViewerDoc || viewerPage >= activeViewerDoc.page_count} onClick={() => setViewerPage((p) => p + 1)}><ChevronRight size={18} /></IconButton></div><div className="flex items-center"><IconButton label="Zoom out" disabled={zoom <= 0.5} onClick={() => setZoom((z) => Math.max(0.5, z - 0.25))}><ZoomOut size={16} /></IconButton><span className="w-11 text-center">{Math.round(zoom * 100)}%</span><IconButton label="Zoom in" disabled={zoom >= 2.5} onClick={() => setZoom((z) => Math.min(2.5, z + 0.25))}><ZoomIn size={16} /></IconButton></div></div>
        <div className="mx-3.5 mb-3.5 min-h-0 flex-1 overflow-auto rounded-[22px] bg-viewer p-4 text-center sm:p-5">{imageUrl ? <img key={viewerPage} src={imageUrl} alt={`${viewerSource.filename} page ${viewerPage}`} style={{ width: `${zoom * 100}%`, maxWidth: 'none' }} className="page-turn mx-auto rounded-[10px] bg-white shadow-[0_10px_30px_rgba(0,0,0,.08)]" /> : <div className="mx-auto mt-12 max-w-xs rounded-[22px] bg-surface p-6 text-left"><FileText size={24} className="mb-3 text-muted" /><h3 className="mb-2 text-sm font-semibold">Page image unavailable</h3><p className="text-sm leading-6 text-muted">{imageError || (pageImagePath ? "Loading page image…" : `Unable to locate the image for ${viewerSource.filename}, page ${viewerPage}.`)}</p>{viewerSource.excerpt && <blockquote className="mt-4 rounded-2xl bg-purple-soft px-3.5 py-2.5 text-sm leading-6 text-ink">{viewerSource.excerpt}</blockquote>}</div>}</div>
      </aside>}
    </div>
  )
}

function App() {
  const [user, setUser] = useState<AuthUser | null>(null)
  const [checkingSession, setCheckingSession] = useState(true)

  const onUnauthorized = useCallback(() => setUser(null), [])
  useEffect(() => {
    if (/\/(verify-email|reset-password)\/?$/.test(window.location.pathname)) {
      setCheckingSession(false)
      return
    }
    let active = true
    void authApi.me()
      .then((profile) => { if (active) setUser(profile) })
      .catch(() => { if (active) setUser(null) })
      .finally(() => { if (active) setCheckingSession(false) })
    return () => { active = false }
  }, [])

  async function logout() {
    if (!user) return
    try {
      await authApi.logout(user.csrf_token)
      setUser(null)
    } catch (error) {
      if (error instanceof Error && 'status' in error && error.status === 401) {
        setUser(null)
        return
      }
      throw error
    }
  }

  if (checkingSession) {
    return <div role="status" className="flex min-h-dvh items-center justify-center gap-3 bg-night text-sm text-night-body"><LogoMark size={40} variant="think" tone="dark" /> Checking session…</div>
  }
  if (!user) return <AuthScreen onLogin={setUser} />
  return <TutorApp key={user.user_id} user={user} onUnauthorized={onUnauthorized} onLogout={logout} />
}

export default App
