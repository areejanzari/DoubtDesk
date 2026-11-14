import { describe, expect, it } from 'vitest'
import { extractSources, linkSourceReferences, normalizeRawCitations, sourcesFromAnswer } from './citations'

const id = 'e8be2d4d-9094-43e8-8cec-7fe4e7853cce'
const other = '1c2f3e40-1111-2222-3333-444455556666'
const docs = [{ doc_id: id, filename: 'biology.pdf', page_count: 7 }, { doc_id: other, filename: 'biology.pdf', page_count: 7 }]

describe('PDF page citation regression', () => {
  it.each(['(PDF p. 1)', '(PDF page 1)', '(pdf p 1)', `(${id}, p. 1)`])('normalizes and links %s', raw => {
    const normalized = normalizeRawCitations(`Ureotelic animals excrete urea. ${raw}`, [{ docId: id, page: 1 }])
    expect(normalized).toBe('Ureotelic animals excrete urea. [1]')
    expect(linkSourceReferences(normalized, 1)).toBe('Ureotelic animals excrete urea. [1](#doubtdesk-source-1)')
  })
  it('uses the source index rather than the PDF page number', () => {
    expect(normalizeRawCitations('A (PDF p. 2) B (PDF p. 7)', [{ docId: id, page: 7 }, { docId: id, page: 2 }])).toBe('A [2] B [1]')
  })
  it('never guesses which PDF a page belongs to', () => {
    expect(normalizeRawCitations('A (PDF p. 1) B (PDF p. 2)', [{ docId: id, page: 1 }, { docId: other, page: 1 }])).toBe('A [ambiguous source] B [unverified source]')
  })
  it('keeps same-filename documents distinct and preserves deduplicated source order', () => {
    const sources = extractSources([{ doc_id: id, page_numbers: [7, 1, 7] }, { doc_id: other, page_numbers: ['1', 0, 'bad'], text: 'Source text' }], docs)
    expect(sources.map(s => [s.docId, s.page])).toEqual([[id, 7], [id, 1], [other, 1]])
    expect(sources[2].excerpt).toBe('Source text')
    expect(normalizeRawCitations(`A (${other}, p. 1)`, sources)).toBe('A [3]')
  })
  it('retains source IDs even before the document list loads', () => {
    expect(extractSources([{ doc_id: id, page_number: 2 }], [])).toEqual([{ docId: id, filename: 'PDF document', page: 2, excerpt: undefined }])
  })
  it('handles invalid metadata and legacy filenames without guessing', () => {
    expect(extractSources({}, docs)).toEqual([])
    expect(extractSources([null, 'bad', { filename: 'biology.pdf', page: 1 }], docs)).toEqual([])
    expect(extractSources([{ filename: 'biology.pdf', pages: [1], excerpt: 'text' }], [docs[0]])[0]).toMatchObject({ docId: id, page: 1, excerpt: 'text' })
    expect(sourcesFromAnswer('biology.pdf: PDF page(s) 1', docs)).toEqual([])
    expect(sourcesFromAnswer('biology.pdf: PDF page(s) 1, 2', [docs[0]]).map(s => s.page)).toEqual([1, 2])
  })
})
