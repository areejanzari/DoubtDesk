export type CitationSource = { docId?: string; page: number }
export type DocumentInfo = { doc_id: string; filename: string; page_count: number }
export type Source = CitationSource & { filename: string; excerpt?: string }

const RAW_PDF_CITATION = /\(\s*([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\s*,\s*p(?:age)?\.?\s*(\d+)\s*\)/gi
const PDF_PAGE_CITATION = /\(\s*PDF\s+p(?:age)?\.?\s*(\d+)\s*\)/gi

export function normalizeRawCitations(content: string, sources: CitationSource[]): string {
  return content.replace(RAW_PDF_CITATION, (_match, docId: string, page: string) => {
    const index = sources.findIndex(source => source.docId?.toLowerCase() === docId.toLowerCase() && source.page === Number(page))
    return index < 0 ? '[unverified source]' : `[${index + 1}]`
  }).replace(PDF_PAGE_CITATION, (_match, page: string) => {
    const matches = sources.flatMap((source, index) => source.page === Number(page) ? [index + 1] : [])
    if (matches.length === 1) return `[${matches[0]}]`
    return matches.length ? '[ambiguous source]' : '[unverified source]'
  })
}

export function linkSourceReferences(content: string, sourceCount: number): string {
  if (sourceCount <= 0) return content
  return content.replace(/\[(\d{1,3})\](?!\()/g, (match, value: string) => {
    const index = Number(value)
    return index > 0 && index <= sourceCount ? `[${index}](#doubtdesk-source-${index})` : match
  })
}

export function extractSources(raw: unknown, documents: DocumentInfo[]): Source[] {
  if (!Array.isArray(raw)) return []
  const result: Source[] = []
  const seen = new Set<string>()
  for (const entry of raw) {
    if (!entry || typeof entry !== 'object') continue
    const item = entry as Record<string, unknown>
    const docId = typeof item.doc_id === 'string' && item.doc_id ? item.doc_id : undefined
    const match = documents.find(d => d.doc_id === docId)
    const filename = match?.filename || (typeof item.filename === 'string' ? item.filename : undefined)
    const candidates = filename ? documents.filter(d => d.filename.toLowerCase() === filename.toLowerCase()) : []
    const resolvedId = docId || (candidates.length === 1 ? candidates[0].doc_id : undefined)
    if (!resolvedId) continue
    const excerpt = typeof item.excerpt === 'string' ? item.excerpt : typeof item.text === 'string' ? item.text : undefined
    const rawPages = item.page_numbers ?? item.pages ?? item.page_number ?? item.page
    for (const rawPage of Array.isArray(rawPages) ? rawPages : [rawPages]) {
      const page = typeof rawPage === 'number' ? rawPage : typeof rawPage === 'string' && /^\d+$/.test(rawPage) ? Number(rawPage) : NaN
      const key = `${resolvedId.toLowerCase()}:${page}`
      if (!Number.isInteger(page) || page <= 0 || seen.has(key)) continue
      seen.add(key)
      result.push({ docId: resolvedId, filename: filename || 'PDF document', page, excerpt })
    }
  }
  return result
}

export function sourcesFromAnswer(answer: string, documents: DocumentInfo[]): Source[] {
  const raw: { doc_id: string; page_numbers: number[] }[] = []
  const pattern = /([^\n:]+?\.pdf)\s*:\s*PDF page\(s\)\s*([\d,\s-]+)/gi
  for (const match of answer.matchAll(pattern)) {
    const filename = match[1].trim().replace(/^[-*\s]+/, '')
    const candidates = documents.filter(d => d.filename.toLowerCase() === filename.toLowerCase())
    if (candidates.length !== 1) continue
    raw.push({ doc_id: candidates[0].doc_id, page_numbers: (match[2].match(/\d+/g) || []).map(Number) })
  }
  return extractSources(raw, documents)
}
