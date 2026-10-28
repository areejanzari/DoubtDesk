import { useEffect, useRef, useState } from 'react'
import type { ChangeEvent, KeyboardEvent, ReactNode } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import {
  ArrowUp, BookOpen, Bot, Check, ChevronLeft,
  ChevronRight, Clipboard, FileText, GraduationCap, LoaderCircle,
  Menu, PanelLeftClose, PanelLeftOpen, Paperclip, Plus,
  Search, Settings, X, ZoomIn, ZoomOut, Maximize2,
} from 'lucide-react'
import 'katex/dist/katex.min.css'

const API_BASE = (import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000').replace(/\/$/, '')
const PAGE_IMAGE_TEMPLATE = import.meta.env.VITE_PAGE_IMAGE_URL_TEMPLATE || '/documents/{doc_id}/pages/{page}/image'

type DocumentInfo = { doc_id: string; filename: string; page_count: number }
type AnswerStatus = 'grounded' | 'general_knowledge' | 'out_of_scope' | 'unknown'
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
  status: 'grounded' | 'general_knowledge' | 'out_of_scope'
  answer: string
  formatted_answer: string
  sources: unknown
  conversation_id: string | null
}

function extractSources(raw: unknown, documents: DocumentInfo[]): Source[] {
  if (!Array.isArray(raw)) return []
  const result: Source[] = []
  for (const entry of raw) {
    if (!entry || typeof entry !== 'object') continue
    const item = entry as Record<string, unknown>
    const docId = typeof item.doc_id === 'string' ? item.doc_id : undefined
    const match = documents.find((d) => d.doc_id === docId)
    const filename = typeof item.filename === 'string' ? item.filename : match?.filename
    const excerpt = typeof item.excerpt === 'string' ? item.excerpt :
      typeof item.text === 'string' ? item.text : undefined
    const rawPages = item.page_numbers ?? item.pages ?? item.page_number ?? item.page
    const pages = Array.isArray(rawPages) ? rawPages : [rawPages]
    if (!filename) continue
    for (const rawPage of pages) {
      const page = typeof rawPage === 'number' ? rawPage :
        typeof rawPage === 'string' && /^\d+$/.test(rawPage) ? Number(rawPage) : NaN
      if (Number.isInteger(page) && page > 0) result.push({ docId, filename, page, excerpt })
    }
  }
  return result.filter((s, i, all) => all.findIndex((x) => x.filename === s.filename && x.page === s.page) === i)
}

// Fallback for the backend's existing formatted_answer citation convention.
// Extracts only references that the backend actually returned; never invents them.
function sourcesFromAnswer(answer: string, documents: DocumentInfo[]): Source[] {
  const result: Source[] = []
  const pattern = /([^\n:]+?\.pdf)\s*:\s*PDF page\(s\)\s*([\d,\s-]+)/gi
  for (const match of answer.matchAll(pattern)) {
    const filename = match[1].trim().replace(/^[-*\s]+/, '')
    const doc = documents.find((d) => d.filename.toLowerCase() === filename.toLowerCase())
    const numbers = match[2].match(/\d+/g) || []
    for (const number of numbers) result.push({ filename, docId: doc?.doc_id, page: Number(number) })
  }
  return result.filter((s, i, all) => all.findIndex((x) => x.filename === s.filename && x.page === s.page) === i)
}

function MarkdownContent({ content }: { content: string }) {
  return (
    <div className="markdown-body min-w-0 break-words">
      <ReactMarkdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeKatex]}
        components={{
          a: ({ href, children }) => <a href={href} target="_blank" rel="noreferrer">{children}</a>,
          img: ({ src, alt }) => <img src={src} alt={alt || 'Provided illustration'} loading="lazy" />,
        }}
      >{content}</ReactMarkdown>
    </div>
  )
}

function IconButton({ label, onClick, children, disabled = false }: {
  label: string; onClick: () => void; children: ReactNode; disabled?: boolean
}) {
  return <button type="button" title={label} aria-label={label} onClick={onClick} disabled={disabled}
    className="icon-button">{children}</button>
}

function App() {
  const [documents, setDocuments] = useState<DocumentInfo[]>([])
  const [selectedDocIds, setSelectedDocIds] = useState<string[]>([])
  const [documentSearch, setDocumentSearch] = useState('')
  const [subject, setSubject] = useState('All')
  const [question, setQuestion] = useState('')
  const [messages, setMessages] = useState<Message[]>([])
  const [conversationId, setConversationId] = useState<string | null>(null)
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
  const fileInputRef = useRef<HTMLInputElement>(null)
  const chatEndRef = useRef<HTMLDivElement>(null)
  const messageIdRef = useRef(0)
  const nextId = () => ++messageIdRef.current

  async function loadDocuments() {
    try {
      const response = await fetch(`${API_BASE}/documents`)
      if (!response.ok) throw new Error(`Document list failed (${response.status})`)
      const data: unknown = await response.json()
      if (!Array.isArray(data)) throw new Error('Unexpected document list response')
      setDocuments(data as DocumentInfo[])
      setDocumentError('')
    } catch (error) {
      setDocumentError(error instanceof Error ? error.message : 'Unable to load documents')
    }
  }

  useEffect(() => { void loadDocuments() }, [])
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
      const response = await fetch(`${API_BASE}/documents/upload`, { method: 'POST', body: form })
      const result = await response.json()
      if (!response.ok) throw new Error(typeof result.detail === 'string' ? result.detail : 'Upload failed')
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
    setQuestion('')
    setMessages((old) => [...old, { id: nextId(), role: 'user', content: text }])
    setIsAnswering(true)
    try {
      const response = await fetch(`${API_BASE}/ask`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: text,
          doc_ids: selectedDocIds,
          ...(conversationId ? { conversation_id: conversationId } : {}),
        }),
      })
      const data: AskResponse | { detail?: unknown } | null = await response.json().catch(() => null)
      if (!response.ok) {
        const detail = data && 'detail' in data && typeof data.detail === 'string' ? data.detail : null
        throw new Error(detail || `Unable to answer (${response.status})`)
      }
      if (!data || !('answer' in data) || typeof data.answer !== 'string') throw new Error('Unexpected answer response')
      const answerData = data as AskResponse
      if (answerData.conversation_id) setConversationId(answerData.conversation_id)
      const content = answerData.answer || answerData.formatted_answer || 'No answer was returned.'
      const status: AnswerStatus = answerData.status === 'grounded' || answerData.status === 'general_knowledge' || answerData.status === 'out_of_scope' ? answerData.status : 'unknown'
      const parsedSources = status === 'grounded' ? extractSources(answerData.sources, documents) : []
      const sources = status === 'grounded'
        ? (parsedSources.length ? parsedSources : sourcesFromAnswer(answerData.formatted_answer || '', documents))
        : []
      setMessages((old) => [...old, { id: nextId(), role: 'assistant', content, status, sources }])
    } catch (error) {
      setMessages((old) => [...old, { id: nextId(), role: 'assistant',
        content: error instanceof Error ? error.message : 'Something went wrong.', error: true }])
    } finally {
      setIsAnswering(false)
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

  const activeViewerDoc = viewerSource && documents.find((d) =>
    (viewerSource.docId && d.doc_id === viewerSource.docId) || d.filename === viewerSource.filename)
  const pageImageUrl = viewerSource && activeViewerDoc && PAGE_IMAGE_TEMPLATE
    ? `${API_BASE}${PAGE_IMAGE_TEMPLATE.replace('{doc_id}', encodeURIComponent(activeViewerDoc.doc_id)).replace('{page}', String(viewerPage))}`
    : null
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
          <div className="mb-3 flex items-center justify-between px-1"><span className="text-xs font-semibold uppercase tracking-wider text-slate-500">All Documents</span><span className="rounded-md bg-white px-2 py-0.5 text-xs text-slate-500">{documents.length}</span></div>
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
          <div className="mt-4 border-t border-slate-200 pt-4"><div className="flex items-center gap-2 px-2 text-sm font-medium text-slate-600"><BookOpen size={17} /> Current chat</div><p className="mt-1 px-2 text-xs text-slate-400">History lasts until this page is refreshed.</p><div className="mt-4 flex items-center gap-2 px-2 text-sm text-slate-400"><Settings size={17} /> Settings unavailable</div></div>
        </div>
      </aside>

      <section className="flex min-w-0 flex-1 flex-col bg-white">
        <header className="flex min-h-[76px] items-center justify-between gap-3 border-b border-slate-200 px-4 sm:px-6">
          <div className="flex min-w-0 items-center gap-3">
            <IconButton label="Toggle navigation" onClick={() => { if (window.innerWidth < 1024) setMobileSidebarOpen(true); else setSidebarOpen((old) => !old) }}><Menu className="lg:hidden" size={20} /><PanelLeftOpen className="hidden lg:block" size={20} /></IconButton>
            <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-blue-50 text-[#4169FF]"><Bot size={20} /></div>
            <div className="min-w-0"><h1 className="text-base font-semibold sm:text-lg">AI Tutor</h1><p className="hidden truncate text-xs text-slate-500 sm:block">Ask questions about your uploaded documents.</p></div>
          </div>
          <div className="flex items-center gap-2"><span className="hidden rounded-full bg-slate-100 px-3 py-1.5 text-xs text-slate-600 sm:inline-flex">{selectedDocIds.length ? `${selectedDocIds.length} PDF${selectedDocIds.length === 1 ? '' : 's'} selected` : 'General knowledge'}</span>{viewerSource && <IconButton label="Close source viewer" onClick={() => setViewerSource(null)}><X size={18} /></IconButton>}</div>
        </header>
        <div className="min-h-0 flex-1 overflow-y-auto px-4 py-7 sm:px-7" aria-live="polite">
          <div className="mx-auto w-full max-w-4xl space-y-7">
            {messages.length === 0 && <div className="flex flex-col items-center justify-center px-3 py-16 text-center sm:py-24"><div className="mb-5 rounded-2xl bg-blue-50 p-5 text-[#4169FF]"><GraduationCap size={35} /></div><h2 className="text-2xl font-semibold tracking-tight">What would you like to learn?</h2><p className="mt-3 max-w-md text-sm leading-6 text-slate-500">Select one or more PDFs to ask document-grounded questions, or ask anything without selecting a document.</p></div>}
            {messages.map((message) => <article key={message.id} className={message.role === 'user' ? 'flex justify-end' : 'flex items-start gap-3'}>
              {message.role === 'assistant' && <div className="mt-1 flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-blue-50 text-[#4169FF]"><GraduationCap size={17} /></div>}
              <div className={message.role === 'user' ? 'max-w-[88%] rounded-2xl rounded-tr-md bg-[#EEF2FF] px-4 py-3 text-sm leading-7 sm:max-w-[78%]' : 'min-w-0 max-w-full flex-1'}>
                {message.role === 'assistant' && <div className="mb-2 flex flex-wrap items-center gap-2"><span className="text-sm font-semibold">DoubtDesk</span>{message.status === 'grounded' && <span className="rounded-full bg-emerald-50 px-2 py-0.5 text-xs text-emerald-700">From documents</span>}{message.status === 'general_knowledge' && <span className="rounded-full bg-amber-50 px-2 py-0.5 text-xs text-amber-700">Not from document</span>}{message.status === 'out_of_scope' && <span className="rounded-full bg-slate-100 px-2 py-0.5 text-xs text-slate-600">Outside academic scope</span>}{message.error && <span className="text-xs text-red-600">Request failed</span>}</div>}
                {message.role === 'assistant' ? <MarkdownContent content={message.content} /> : <p className="whitespace-pre-wrap break-words">{message.content}</p>}
                {message.role === 'assistant' && <div className="mt-4 space-y-2"><button className="inline-flex items-center gap-1.5 rounded-lg px-2 py-1 text-xs text-slate-500 hover:bg-slate-100" onClick={() => { void navigator.clipboard.writeText(message.content).then(() => { setCopiedId(message.id); window.setTimeout(() => setCopiedId(null), 1500) }).catch(() => {}) }}>{copiedId === message.id ? <Check size={14} /> : <Clipboard size={14} />}{copiedId === message.id ? 'Copied' : 'Copy answer'}</button>
                  {!!message.sources?.length && <div className="border-t border-slate-100 pt-3"><p className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">Sources</p><div className="flex flex-wrap gap-2">{message.sources.map((source, index) => <button key={`${source.filename}-${source.page}-${index}`} className="flex max-w-full items-start gap-2 rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 text-left transition hover:border-blue-300 hover:bg-blue-50" onClick={() => openSource(source)}><FileText size={16} className="mt-0.5 shrink-0 text-[#4169FF]" /><span className="min-w-0"><span className="block break-all text-xs font-medium">[{index + 1}] {source.filename}</span><span className="text-xs text-slate-500">Page {source.page}</span>{source.excerpt && <span className="mt-1 block line-clamp-2 text-xs text-slate-500">{source.excerpt}</span>}</span></button>)}</div></div>}
                </div>}
              </div>
            </article>)}
            {isAnswering && <div className="flex items-center gap-3 text-sm text-slate-500"><div className="flex h-8 w-8 items-center justify-center rounded-lg bg-blue-50 text-[#4169FF]"><Bot size={17} /></div><LoaderCircle className="animate-spin text-[#4169FF]" size={17} /> Thinking…</div>}
            <div ref={chatEndRef} />
          </div>
        </div>
        <div className="shrink-0 border-t border-slate-100 bg-white px-4 pb-[max(1rem,env(safe-area-inset-bottom))] pt-3 sm:px-7"><div className="mx-auto w-full max-w-4xl"><div className="flex items-end gap-2 rounded-2xl border border-slate-200 bg-white p-2 shadow-sm focus-within:border-blue-300 focus-within:ring-2 focus-within:ring-blue-50"><IconButton label="Attach PDF" onClick={() => fileInputRef.current?.click()} disabled={uploading}><Paperclip size={19} /></IconButton><textarea rows={1} value={question} onChange={(e) => setQuestion(e.target.value)} onKeyDown={onQuestionKeyDown} placeholder="Ask a follow-up question…" className="max-h-36 min-h-10 flex-1 resize-none bg-transparent px-1 py-2 text-sm leading-6 outline-none placeholder:text-slate-400" /><button aria-label="Send message" disabled={!question.trim() || isAnswering} onClick={() => void submitQuestion()} className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-[#4169FF] text-white transition hover:bg-[#3155E7] disabled:cursor-not-allowed disabled:opacity-40">{isAnswering ? <LoaderCircle size={18} className="animate-spin" /> : <ArrowUp size={19} />}</button></div><p className="mt-2 text-center text-[11px] text-slate-400">Enter to send · Shift+Enter for a new line · Check important answers</p></div></div>
      </section>

      {viewerSource && <aside className={`${viewerFullscreen ? 'fixed inset-0 z-50 w-full' : 'fixed inset-0 z-40 w-full lg:static lg:z-auto lg:w-[380px] lg:shrink-0'} flex min-h-0 flex-col border-l border-slate-200 bg-white shadow-xl lg:shadow-none`}>
        <div className="flex items-center justify-between gap-2 border-b border-slate-200 p-4"><div className="min-w-0"><p className="truncate text-sm font-semibold">{viewerSource.filename}</p><p className="text-xs text-slate-500">Page {viewerPage}{activeViewerDoc ? ` of ${activeViewerDoc.page_count}` : ''}</p></div><div className="flex items-center"><IconButton label="Fullscreen" onClick={() => setViewerFullscreen((v) => !v)}><Maximize2 size={17} /></IconButton><IconButton label="Close viewer" onClick={() => { setViewerSource(null); setViewerFullscreen(false) }}><X size={18} /></IconButton></div></div>
        <div className="flex items-center justify-between gap-2 border-b border-slate-100 px-3 py-2"><div className="flex items-center"><IconButton label="Previous page" disabled={viewerPage <= 1} onClick={() => setViewerPage((p) => Math.max(1, p - 1))}><ChevronLeft size={18} /></IconButton><span className="px-2 text-xs text-slate-600">{viewerPage}{activeViewerDoc ? ` / ${activeViewerDoc.page_count}` : ''}</span><IconButton label="Next page" disabled={!activeViewerDoc || viewerPage >= activeViewerDoc.page_count} onClick={() => setViewerPage((p) => p + 1)}><ChevronRight size={18} /></IconButton></div><div className="flex items-center"><IconButton label="Zoom out" disabled={zoom <= 0.5} onClick={() => setZoom((z) => Math.max(0.5, z - 0.25))}><ZoomOut size={17} /></IconButton><span className="w-11 text-center text-xs text-slate-500">{Math.round(zoom * 100)}%</span><IconButton label="Zoom in" disabled={zoom >= 2.5} onClick={() => setZoom((z) => Math.min(2.5, z + 0.25))}><ZoomIn size={17} /></IconButton></div></div>
        <div className="min-h-0 flex-1 overflow-auto bg-slate-100 p-5 text-center">{pageImageUrl ? <img src={pageImageUrl} alt={`${viewerSource.filename} page ${viewerPage}`} style={{ width: `${zoom * 100}%`, maxWidth: 'none' }} className="mx-auto bg-white shadow-lg" /> : <div className="mx-auto mt-12 max-w-xs rounded-2xl border border-slate-200 bg-white p-6 text-left shadow-sm"><FileText size={28} className="mb-3 text-[#4169FF]" /><h3 className="mb-2 text-sm font-semibold">Page image unavailable</h3><p className="text-sm leading-6 text-slate-500">Unable to locate the image for {viewerSource.filename}, page {viewerPage}.</p>{viewerSource.excerpt && <blockquote className="mt-4 border-l-2 border-blue-300 pl-3 text-xs leading-5 text-slate-600">{viewerSource.excerpt}</blockquote>}</div>}</div>
      </aside>}
    </div>
  )
}

export default App
