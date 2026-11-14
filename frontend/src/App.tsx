import { useCallback, useEffect, useRef, useState } from 'react'
import type { ChangeEvent, KeyboardEvent, ReactNode } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import {
  ArrowUp, BookOpen, Bot, Check, ChevronLeft,
  ChevronRight, Clipboard, FileText, GraduationCap, LoaderCircle,
  Menu, PanelLeftClose, PanelLeftOpen, Paperclip, Plus,
  Search, Settings, X, ZoomIn, ZoomOut, Maximize2, LogOut,
} from 'lucide-react'
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

type DocumentInfo = { doc_id: string; filename: string; page_count: number }
type AnswerStatus = 'grounded' | 'general_knowledge' | 'out_of_scope' | 'verification_failed' | 'unknown'
type Source = { docId?: string; filename: string; page: number; excerpt?: string }
type Message = {
  id: number
  role: 'user' | 'assistant'
  content: string
  status?: AnswerStatus
  sources?: Source[]
  error?: boolean
}
type AskResponse = {
  status: AnswerStatus
  answer: string
  formatted_answer: string
  sources: unknown
  conversation_id: string | null
}

function MarkdownContent({ content, sources = [], onSourceOpen }: {
  content: string
  sources?: Source[]
  onSourceOpen?: (source: Source) => void
}) {
  const normalized = sources.length ? normalizeRawCitations(content, sources) : content
  const linked = linkSourceReferences(normalized, sources.length)
  return (
    <div className="markdown-body min-w-0 break-words">
      <ReactMarkdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeKatex]}
        components={{
          a: ({ href, children }) => {
            const sourceMatch = /^#doubtdesk-source-(\d+)$/.exec(href || '')
            const source = sourceMatch ? sources[Number(sourceMatch[1]) - 1] : undefined
            if (source) return <button type="button" aria-label={`Open source ${sourceMatch![1]}`} onClick={() => onSourceOpen?.(source)} className="rounded text-[#3155E7] underline underline-offset-2 hover:text-[#153fc9]">{`[${sourceMatch![1]}]`}</button>
            return <a href={href} target="_blank" rel="noreferrer">{children}</a>
          },
          img: ({ src, alt }) => <img src={src} alt={alt || 'Provided illustration'} loading="lazy" />,
        }}
      >{linked}</ReactMarkdown>
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
  const fileInputRef = useRef<HTMLInputElement>(null)
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
      const result = await apiRequest<{ filename: string; status: string; page_count: number }>(
        '/documents/upload', { method: 'POST', body: form }, user.csrf_token, onUnauthorized,
      )
      setUploadMessage(result.status === 'cached' ? `${result.filename}: cached document reused.` : `${result.filename}: uploaded (${result.page_count} pages).`)
      await loadDocuments()
    } catch (error) {
      setUploadMessage(error instanceof Error ? error.message : 'Upload failed')
    } finally {
      setUploading(false)
    }
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
    try {
      await streamAsk({
        question: text,
        doc_ids: selectedDocIds,
        ...(conversationId ? { conversation_id: conversationId } : {}),
      }, user.csrf_token, onUnauthorized, ({ event, data }) => {
        if (event === 'status') {
          const stage = typeof data.stage === 'string' ? data.stage : 'generating'
          setAnswerStage(stage === 'checking' ? 'Verifying…' : stage === 'preparing' ? 'Preparing…' : 'Generating…')
        }
        if (event === 'token' && typeof data.text === 'string') {
          const token = data.text
          setMessages(old => {
            const existing = old.find(message => message.id === assistantMessageId)
            return existing
              ? old.map(message => message.id === assistantMessageId ? { ...message, content: message.content + token } : message)
              : [...old, { id: assistantMessageId, role: 'assistant', content: token }]
          })
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
            const saved = { id: assistantMessageId, role: 'assistant' as const, content: result.answer, status, sources }
            return old.some(message => message.id === assistantMessageId)
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
      setAnswerStage('Thinking…')
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
  const visibleDocs = documents.filter((doc) => {
    const matchesSearch = doc.filename.toLowerCase().includes(documentSearch.toLowerCase())
    // Filename-based subjects are only a UI filter; backend has no subject metadata.
    const matchesSubject = subject === 'All' || doc.filename.toLowerCase().includes(subject.toLowerCase())
    return matchesSearch && matchesSubject
  })

  return (
    <div className="flex h-dvh min-h-0 overflow-hidden bg-[#F8FAFC] text-[#0F172A]">
      {mobileSidebarOpen && <button aria-label="Close navigation" className="fixed inset-0 z-30 bg-slate-900/30 lg:hidden" onClick={() => setMobileSidebarOpen(false)} />}
      <aside className={`sidebar ${mobileSidebarOpen ? 'sidebar--mobile-open' : ''} ${sidebarOpen ? '' : 'sidebar--collapsed'}`}>
        <div className="flex h-full w-[260px] flex-col overflow-hidden p-4">
          <div className="mb-7 flex items-center justify-between px-1 pt-1">
            <div className="flex items-center gap-2.5"><div className="flex h-9 w-9 items-center justify-center rounded-xl bg-[#4169FF] text-white"><GraduationCap size={21} /></div><span className="text-xl font-bold tracking-tight">DoubtDesk</span></div>
            <IconButton label="Close sidebar" onClick={() => { setSidebarOpen(false); setMobileSidebarOpen(false) }}><PanelLeftClose size={18} /></IconButton>
          </div>
          <input ref={fileInputRef} type="file" accept=".pdf,application/pdf" className="hidden" onChange={onFileChange} />
          <button className="primary-button mb-3 w-full justify-center" disabled={uploading} onClick={() => fileInputRef.current?.click()}>
            {uploading ? <LoaderCircle className="animate-spin" size={17} /> : <Plus size={17} />}{uploading ? 'Processing PDF…' : 'Upload PDF'}
          </button>
          {uploadMessage && <p role="status" className="mb-3 rounded-lg bg-white px-2.5 py-2 text-xs leading-5 text-slate-600">{uploadMessage}</p>}
          <ConversationSidebar conversations={conversations} activeId={conversationId} busy={isAnswering}
            loading={historyLoading} error={historyError} onNew={() => void startConversation()}
            onOpen={id => void openConversation(id)} onRename={renameConversation} onDelete={deleteConversation} />
          <div className="mb-3 mt-4 flex items-center justify-between px-1"><span className="text-xs font-semibold uppercase tracking-wider text-slate-500">All Documents</span><span className="rounded-md bg-white px-2 py-0.5 text-xs text-slate-500">{documents.length}</span></div>
          <div className="relative mb-3"><Search size={16} className="absolute left-3 top-2.5 text-slate-400" /><input value={documentSearch} onChange={(e) => setDocumentSearch(e.target.value)} placeholder="Search documents" className="field w-full pl-9 text-sm" /></div>
          <div className="mb-3 flex flex-wrap gap-1.5">{['All', 'Biology', 'Chemistry', 'Physics'].map((name) => <button key={name} onClick={() => setSubject(name)} className={`rounded-lg px-2 py-1 text-xs transition ${subject === name ? 'bg-blue-100 font-semibold text-[#3155E7]' : 'bg-white text-slate-500 hover:bg-slate-100'}`}>{name}</button>)}</div>
          {documentError && <p role="alert" className="mb-2 text-xs text-red-600">{documentError} <button className="underline" onClick={() => void loadDocuments()}>Retry</button></p>}
          <div className="min-h-0 flex-1 space-y-1 overflow-y-auto pr-1">
            {visibleDocs.length === 0 && <p className="px-2 py-5 text-center text-xs leading-5 text-slate-400">{documents.length ? 'No matching PDFs.' : 'Upload a PDF to get started.'}</p>}
            {visibleDocs.map((doc) => {
              const selected = selectedDocIds.includes(doc.doc_id)
              return <label key={doc.doc_id} className={`flex cursor-pointer items-start gap-2.5 rounded-xl border p-2.5 transition ${selected ? 'border-blue-200 bg-blue-50' : 'border-transparent hover:bg-white'}`}>
                <input type="checkbox" className="mt-1 accent-[#4169FF]" checked={selected} onChange={() => setSelectedDocIds((old) => selected ? old.filter((id) => id !== doc.doc_id) : [...old, doc.doc_id])} />
                <FileText size={18} className="mt-0.5 shrink-0 text-[#4169FF]" />
                <span className="min-w-0"><span className="block break-all text-sm font-medium leading-5">{doc.filename}</span><span className="text-xs text-slate-400">{doc.page_count} pages</span></span>
              </label>
            })}
          </div>
          <div className="mt-4 border-t border-slate-200 pt-4"><div className="flex items-center gap-2 px-2 text-sm text-slate-500"><BookOpen size={17} /> Saved conversations</div><p className="mt-1 px-2 text-xs text-slate-400">Up to 30 chats per account.</p><div className="mt-3 flex items-center gap-2 px-2 text-sm text-slate-400"><Settings size={17} /> Settings unavailable</div></div>
        </div>
      </aside>

      <section className="flex min-w-0 flex-1 flex-col bg-white">
        <header className="flex min-h-[76px] items-center justify-between gap-3 border-b border-slate-200 px-4 sm:px-6">
          <div className="flex min-w-0 items-center gap-3">
            <IconButton label="Toggle navigation" onClick={() => { if (window.innerWidth < 1024) setMobileSidebarOpen(true); else setSidebarOpen((old) => !old) }}><Menu className="lg:hidden" size={20} /><PanelLeftOpen className="hidden lg:block" size={20} /></IconButton>
            <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-blue-50 text-[#4169FF]"><Bot size={20} /></div>
            <div className="min-w-0"><h1 className="text-base font-semibold sm:text-lg">AI Tutor</h1><p className="hidden truncate text-xs text-slate-500 sm:block">Ask questions about your uploaded documents.</p></div>
          </div>
          <div className="flex items-center gap-2"><span className="hidden max-w-32 truncate text-sm text-slate-600 md:inline" title={user.email}>{user.username}</span><button type="button" className="icon-button" title="Sign out" aria-label="Sign out" onClick={() => void signOut()} disabled={loggingOut}>{loggingOut ? <LoaderCircle size={18} className="animate-spin" /> : <LogOut size={18} />}</button><span className="hidden rounded-full bg-slate-100 px-3 py-1.5 text-xs text-slate-600 sm:inline-flex">{selectedDocIds.length ? `${selectedDocIds.length} PDF${selectedDocIds.length === 1 ? '' : 's'} selected` : 'General knowledge'}</span>{viewerSource && <IconButton label="Close source viewer" onClick={() => setViewerSource(null)}><X size={18} /></IconButton>}</div>
        </header>
        {logoutError && <div role="alert" className="bg-red-50 px-5 py-2 text-sm text-red-700">{logoutError}</div>}
        <div className="min-h-0 flex-1 overflow-y-auto px-4 py-7 sm:px-7" aria-live="polite">
          <div className="mx-auto w-full max-w-4xl space-y-7">
            {messages.length === 0 && <div className="flex flex-col items-center justify-center px-3 py-16 text-center sm:py-24"><div className="mb-5 rounded-2xl bg-blue-50 p-5 text-[#4169FF]"><GraduationCap size={35} /></div><h2 className="text-2xl font-semibold tracking-tight">What would you like to learn?</h2><p className="mt-3 max-w-md text-sm leading-6 text-slate-500">Select one or more PDFs to ask document-grounded questions, or ask anything without selecting a document.</p></div>}
            {messages.map((message) => <article key={message.id} className={message.role === 'user' ? 'flex justify-end' : 'flex items-start gap-3'}>
              {message.role === 'assistant' && <div className="mt-1 flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-blue-50 text-[#4169FF]"><GraduationCap size={17} /></div>}
              <div className={message.role === 'user' ? 'max-w-[88%] rounded-2xl rounded-tr-md bg-[#EEF2FF] px-4 py-3 text-sm leading-7 sm:max-w-[78%]' : 'min-w-0 max-w-full flex-1'}>
                {message.role === 'assistant' && <div className="mb-2 flex flex-wrap items-center gap-2"><span className="text-sm font-semibold">DoubtDesk</span>{message.status === 'grounded' && <span className="rounded-full bg-emerald-50 px-2 py-0.5 text-xs text-emerald-700">From documents</span>}{message.status === 'general_knowledge' && <span className="rounded-full bg-amber-50 px-2 py-0.5 text-xs text-amber-700">Not from document</span>}{message.status === 'verification_failed' && <span className="rounded-full bg-red-50 px-2 py-0.5 text-xs text-red-700">Verification unavailable</span>}{message.status === 'out_of_scope' && <span className="rounded-full bg-slate-100 px-2 py-0.5 text-xs text-slate-600">Outside academic scope</span>}{message.error && <span className="text-xs text-red-600">Request failed</span>}</div>}
                {message.role === 'assistant' ? <MarkdownContent content={message.content} sources={message.sources} onSourceOpen={openSource} /> : <p className="whitespace-pre-wrap break-words">{message.content}</p>}
                {message.role === 'assistant' && <div className="mt-4 space-y-2"><button className="inline-flex items-center gap-1.5 rounded-lg px-2 py-1 text-xs text-slate-500 hover:bg-slate-100" onClick={() => { void navigator.clipboard.writeText(message.sources?.length ? normalizeRawCitations(message.content, message.sources) : message.content).then(() => { setCopiedId(message.id); window.setTimeout(() => setCopiedId(null), 1500) }).catch(() => {}) }}>{copiedId === message.id ? <Check size={14} /> : <Clipboard size={14} />}{copiedId === message.id ? 'Copied' : 'Copy answer'}</button>
                  {!!message.sources?.length && <div className="border-t border-slate-100 pt-3"><p className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">Sources</p><div className="flex flex-wrap gap-2">{message.sources.map((source, index) => <button key={`${source.filename}-${source.page}-${index}`} className="flex max-w-full items-start gap-2 rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 text-left transition hover:border-blue-300 hover:bg-blue-50" onClick={() => openSource(source)}><FileText size={16} className="mt-0.5 shrink-0 text-[#4169FF]" /><span className="min-w-0"><span className="block break-all text-xs font-medium">[{index + 1}] {source.filename}</span><span className="text-xs text-slate-500">Page {source.page}</span>{source.excerpt && <span className="mt-1 block line-clamp-2 text-xs text-slate-500">{source.excerpt}</span>}</span></button>)}</div></div>}
                </div>}
              </div>
            </article>)}
            {isAnswering && <div className="flex items-center gap-3 text-sm text-slate-500"><div className="flex h-8 w-8 items-center justify-center rounded-lg bg-blue-50 text-[#4169FF]"><Bot size={17} /></div><LoaderCircle className="animate-spin text-[#4169FF]" size={17} /> {answerStage}</div>}
            <div ref={chatEndRef} />
          </div>
        </div>
        <div className="shrink-0 border-t border-slate-100 bg-white px-4 pb-[max(1rem,env(safe-area-inset-bottom))] pt-3 sm:px-7"><div className="mx-auto w-full max-w-4xl"><div className="flex items-end gap-2 rounded-2xl border border-slate-200 bg-white p-2 shadow-sm focus-within:border-blue-300 focus-within:ring-2 focus-within:ring-blue-50"><IconButton label="Attach PDF" onClick={() => fileInputRef.current?.click()} disabled={uploading}><Paperclip size={19} /></IconButton><textarea rows={1} value={question} onChange={(e) => setQuestion(e.target.value)} onKeyDown={onQuestionKeyDown} placeholder="Ask a follow-up question…" className="max-h-36 min-h-10 flex-1 resize-none bg-transparent px-1 py-2 text-sm leading-6 outline-none placeholder:text-slate-400" /><button aria-label="Send message" disabled={!question.trim() || isAnswering} onClick={() => void submitQuestion()} className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-[#4169FF] text-white transition hover:bg-[#3155E7] disabled:cursor-not-allowed disabled:opacity-40">{isAnswering ? <LoaderCircle size={18} className="animate-spin" /> : <ArrowUp size={19} />}</button></div><p className="mt-2 text-center text-[11px] text-slate-400">Enter to send · Shift+Enter for a new line · Check important answers</p></div></div>
      </section>

      {viewerSource && <aside className={`${viewerFullscreen ? 'fixed inset-0 z-50 w-full' : 'fixed inset-0 z-40 w-full lg:static lg:z-auto lg:w-[380px] lg:shrink-0'} flex min-h-0 flex-col border-l border-slate-200 bg-white shadow-xl lg:shadow-none`}>
        <div className="flex items-center justify-between gap-2 border-b border-slate-200 p-4"><div className="min-w-0"><p className="truncate text-sm font-semibold">{viewerSource.filename}</p><p className="text-xs text-slate-500">Page {viewerPage}{activeViewerDoc ? ` of ${activeViewerDoc.page_count}` : ''}</p></div><div className="flex items-center"><IconButton label="Fullscreen" onClick={() => setViewerFullscreen((v) => !v)}><Maximize2 size={17} /></IconButton><IconButton label="Close viewer" onClick={() => { setViewerSource(null); setViewerFullscreen(false) }}><X size={18} /></IconButton></div></div>
        <div className="flex items-center justify-between gap-2 border-b border-slate-100 px-3 py-2"><div className="flex items-center"><IconButton label="Previous page" disabled={viewerPage <= 1} onClick={() => setViewerPage((p) => Math.max(1, p - 1))}><ChevronLeft size={18} /></IconButton><span className="px-2 text-xs text-slate-600">{viewerPage}{activeViewerDoc ? ` / ${activeViewerDoc.page_count}` : ''}</span><IconButton label="Next page" disabled={!activeViewerDoc || viewerPage >= activeViewerDoc.page_count} onClick={() => setViewerPage((p) => p + 1)}><ChevronRight size={18} /></IconButton></div><div className="flex items-center"><IconButton label="Zoom out" disabled={zoom <= 0.5} onClick={() => setZoom((z) => Math.max(0.5, z - 0.25))}><ZoomOut size={17} /></IconButton><span className="w-11 text-center text-xs text-slate-500">{Math.round(zoom * 100)}%</span><IconButton label="Zoom in" disabled={zoom >= 2.5} onClick={() => setZoom((z) => Math.min(2.5, z + 0.25))}><ZoomIn size={17} /></IconButton></div></div>
        <div className="min-h-0 flex-1 overflow-auto bg-slate-100 p-5 text-center">{imageUrl ? <img src={imageUrl} alt={`${viewerSource.filename} page ${viewerPage}`} style={{ width: `${zoom * 100}%`, maxWidth: 'none' }} className="mx-auto bg-white shadow-lg" /> : <div className="mx-auto mt-12 max-w-xs rounded-2xl border border-slate-200 bg-white p-6 text-left shadow-sm"><FileText size={28} className="mb-3 text-[#4169FF]" /><h3 className="mb-2 text-sm font-semibold">Page image unavailable</h3><p className="text-sm leading-6 text-slate-500">{imageError || (pageImagePath ? "Loading page image…" : `Unable to locate the image for ${viewerSource.filename}, page ${viewerPage}.`)}</p>{viewerSource.excerpt && <blockquote className="mt-4 border-l-2 border-blue-300 pl-3 text-xs leading-5 text-slate-600">{viewerSource.excerpt}</blockquote>}</div>}</div>
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
    return <div role="status" className="flex min-h-dvh items-center justify-center gap-3 bg-[#F8FAFC] text-sm text-slate-500"><LoaderCircle size={20} className="animate-spin text-[#4169FF]"/> Checking session…</div>
  }
  if (!user) return <AuthScreen onLogin={setUser} />
  return <TutorApp key={user.user_id} user={user} onUnauthorized={onUnauthorized} onLogout={logout} />
}

export default App
