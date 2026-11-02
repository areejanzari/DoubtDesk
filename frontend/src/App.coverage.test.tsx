import { beforeEach, describe, expect, it, vi } from 'vitest'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import App from './App'

const ACCOUNT = {
  user_id: 'test-user', username: 'student_a', email: 'student@example.com', csrf_token: 'csrf-test',
}
const BIOLOGY = { doc_id: 'bio-doc', filename: 'biology.pdf', page_count: 8 }
const CHEMISTRY = { doc_id: 'chem-doc', filename: 'chemistry.pdf', page_count: 3 }

function response(body: unknown, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    headers: new Headers({ 'content-type': 'application/json' }),
    json: async () => body,
  } as Response
}

let documents = [BIOLOGY, CHEMISTRY]
let answer: unknown

beforeEach(() => {
  documents = [BIOLOGY, CHEMISTRY]
  answer = {
    status: 'general_knowledge', answer: 'Water is H2O.', formatted_answer: '',
    sources: [], conversation_id: null,
  }
  window.history.replaceState({}, '', '/')
  vi.mocked(fetch).mockImplementation(async input => {
    const url = String(input)
    if (url.endsWith('/auth/me')) return response(ACCOUNT)
    if (url.endsWith('/documents')) return response(documents)
    if (url.endsWith('/documents/upload')) return response({ filename: 'sample.pdf', status: 'completed', page_count: 2 })
    if (url.endsWith('/ask')) return response(answer)
    if (url.endsWith('/auth/logout')) return response({ message: 'Logged out.' })
    return response({ detail: 'Not found' }, 404)
  })
})

async function openTutor() {
  render(<App />)
  await screen.findByRole('heading', { name: 'AI Tutor' })
  await screen.findByText('biology.pdf')
}

async function askQuestion(text = 'Explain the topic') {
  const user = userEvent.setup()
  await user.type(screen.getByPlaceholderText('Ask a follow-up question…'), text)
  await user.click(screen.getByRole('button', { name: 'Send message' }))
}

describe('additional tutor behavior and failure paths', () => {
  it('rejects empty PDFs before calling the upload API', async () => {
    await openTutor()
    fireEvent.change(document.querySelector('input[type="file"]')!, {
      target: { files: [new File([], 'empty.pdf', { type: 'application/pdf' })] },
    })
    expect(await screen.findByRole('status')).toHaveTextContent('This PDF is empty.')
    expect(vi.mocked(fetch).mock.calls.some(([url]) => String(url).endsWith('/documents/upload'))).toBe(false)
  })

  it('rejects PDFs exceeding the file-size limit', async () => {
    await openTutor()
    const file = new File(['%PDF'], 'oversize.pdf', { type: 'application/pdf' })
    Object.defineProperty(file, 'size', { value: 26 * 1024 * 1024 })
    fireEvent.change(document.querySelector('input[type="file"]')!, { target: { files: [file] } })
    expect(await screen.findByRole('status')).toHaveTextContent('PDF exceeds the 25 MB limit.')
    expect(vi.mocked(fetch).mock.calls.some(([url]) => String(url).endsWith('/documents/upload'))).toBe(false)
  })

  it('shows a server-side upload error and keeps the tutor usable', async () => {
    await openTutor()
    const previous = vi.mocked(fetch).getMockImplementation()!
    vi.mocked(fetch).mockImplementation(async (url, options) => String(url).endsWith('/documents/upload')
      ? response({ detail: 'PDF is corrupted or unreadable.' }, 400)
      : previous(url, options))
    fireEvent.change(document.querySelector('input[type="file"]')!, {
      target: { files: [new File(['%PDF-1.4'], 'corrupted.pdf', { type: 'application/pdf' })] },
    })
    expect(await screen.findByRole('status')).toHaveTextContent('PDF is corrupted or unreadable.')
    expect(screen.getByRole('heading', { name: 'AI Tutor' })).toBeInTheDocument()
  })

  it('shows a document-load error and recovers after retry', async () => {
    let attempts = 0
    const previous = vi.mocked(fetch).getMockImplementation()!
    vi.mocked(fetch).mockImplementation(async (url, options) => {
      if (String(url).endsWith('/documents')) {
        attempts++
        if (attempts === 1) return response({ detail: 'Temporary unavailable' }, 503)
      }
      return previous(url, options)
    })
    render(<App />)
    expect(await screen.findByRole('alert')).toHaveTextContent('Temporary unavailable')
    await userEvent.setup().click(screen.getByRole('button', { name: 'Retry' }))
    expect(await screen.findByText('biology.pdf')).toBeInTheDocument()
    expect(attempts).toBe(2)
  })

  it('rejects an unexpected document-list response instead of rendering it', async () => {
    const previous = vi.mocked(fetch).getMockImplementation()!
    vi.mocked(fetch).mockImplementation(async (url, options) => String(url).endsWith('/documents')
      ? response({ results: documents })
      : previous(url, options))
    render(<App />)
    expect(await screen.findByRole('alert')).toHaveTextContent('Unexpected document list response')
  })

  it('filters documents by subject and handles an empty search result', async () => {
    await openTutor()
    const user = userEvent.setup()
    await user.click(screen.getByRole('button', { name: 'Chemistry' }))
    expect(screen.getByText('chemistry.pdf')).toBeInTheDocument()
    expect(screen.queryByText('biology.pdf')).not.toBeInTheDocument()
    await user.type(screen.getByPlaceholderText('Search documents'), 'no-match')
    expect(screen.getByText('No matching PDFs.')).toBeInTheDocument()
    await user.clear(screen.getByPlaceholderText('Search documents'))
    await user.click(screen.getByRole('button', { name: /^All$/ }))
    expect(screen.getByText('biology.pdf')).toBeInTheDocument()
  })

  it('handles malformed answer responses without silently losing the question', async () => {
    answer = { status: 'grounded', sources: [], conversation_id: null }
    await openTutor()
    await askQuestion('Explain photosynthesis')
    expect(await screen.findByText('Unexpected answer response')).toBeInTheDocument()
    expect(screen.getByText('Explain photosynthesis')).toBeInTheDocument()
  })

  it('shows the out-of-scope status without exposing document citations', async () => {
    answer = {
      status: 'out_of_scope', answer: 'Please ask an academic question.',
      formatted_answer: '', sources: [{ doc_id: 'bio-doc', page_number: 1 }],
      conversation_id: null,
    }
    await openTutor()
    await askQuestion('Suggest a movie')
    expect(await screen.findByText('Please ask an academic question.')).toBeInTheDocument()
    expect(screen.getByText('Outside academic scope')).toBeInTheDocument()
    expect(screen.queryByText('Sources')).not.toBeInTheDocument()
  })

  it('extracts citations from the backend formatted-answer convention', async () => {
    answer = {
      status: 'grounded', answer: 'Photosynthesis stores chemical energy.',
      formatted_answer: 'biology.pdf: PDF page(s) 1, 3', sources: [],
      conversation_id: null,
    }
    await openTutor()
    await askQuestion('Explain photosynthesis')
    expect(await screen.findByRole('button', { name: /biology\.pdf.*Page 1/ })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /biology\.pdf.*Page 3/ })).toBeInTheDocument()
  })

  it('accepts source page arrays and ignores invalid page numbers', async () => {
    answer = {
      status: 'grounded', answer: 'Details in the PDF.', formatted_answer: '',
      sources: [{ doc_id: 'bio-doc', page_numbers: ['2', 3, 0, 'invalid'] }],
      conversation_id: null,
    }
    await openTutor()
    await askQuestion('Show the source')
    expect(await screen.findByRole('button', { name: /biology\.pdf.*Page 2/ })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /biology\.pdf.*Page 3/ })).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /biology\.pdf.*Page 0/ })).not.toBeInTheDocument()
  })

  it('supports viewer page navigation, zoom and fullscreen', async () => {
    answer = {
      status: 'grounded', answer: 'Plants capture energy.', formatted_answer: '',
      sources: [{ doc_id: 'bio-doc', page_number: 2 }], conversation_id: null,
    }
    const createUrl = vi.fn().mockReturnValue('blob:http://localhost/page')
    URL.createObjectURL = createUrl
    URL.revokeObjectURL = vi.fn()
    const previous = vi.mocked(fetch).getMockImplementation()!
    vi.mocked(fetch).mockImplementation(async (url, options) => {
      if (String(url).includes('/pages/') && String(url).endsWith('/image')) {
        return {
          ok: true, status: 200, headers: new Headers({ 'content-type': 'image/png' }),
          blob: async () => new Blob(['image'], { type: 'image/png' }),
        } as Response
      }
      return previous(url, options)
    })
    await openTutor()
    await askQuestion('Show the document')
    await userEvent.setup().click(await screen.findByRole('button', { name: /biology\.pdf.*Page 2/ }))
    expect(screen.getByText('Page 2 of 8')).toBeInTheDocument()
    await userEvent.setup().click(screen.getByRole('button', { name: 'Previous page' }))
    expect(screen.getByText('Page 1 of 8')).toBeInTheDocument()
    await userEvent.setup().click(screen.getByRole('button', { name: 'Next page' }))
    expect(screen.getByText('Page 2 of 8')).toBeInTheDocument()
    await userEvent.setup().click(screen.getByRole('button', { name: 'Zoom in' }))
    expect(screen.getByText('125%')).toBeInTheDocument()
    await userEvent.setup().click(screen.getByRole('button', { name: 'Fullscreen' }))
    expect(screen.getByRole('button', { name: 'Close viewer' })).toBeInTheDocument()
    await userEvent.setup().click(screen.getByRole('button', { name: 'Close viewer' }))
    expect(screen.queryByRole('button', { name: 'Close viewer' })).not.toBeInTheDocument()
    await waitFor(() => expect(createUrl).toHaveBeenCalled())
  })

  it('shows an image error without leaking a public image URL', async () => {
    answer = {
      status: 'grounded', answer: 'See the PDF.', formatted_answer: '',
      sources: [{ doc_id: 'bio-doc', page_number: 1 }], conversation_id: null,
    }
    const previous = vi.mocked(fetch).getMockImplementation()!
    vi.mocked(fetch).mockImplementation(async (url, options) => String(url).includes('/pages/1/image')
      ? response({ detail: 'Page image not found.' }, 404)
      : previous(url, options))
    await openTutor()
    await askQuestion()
    await userEvent.setup().click(await screen.findByRole('button', { name: /biology\.pdf.*Page 1/ }))
    expect(await screen.findByText('Page image not found.')).toBeInTheDocument()
    expect(screen.getByText('Page image unavailable')).toBeInTheDocument()
  })

  it('displays logout failures without discarding the current workspace', async () => {
    const previous = vi.mocked(fetch).getMockImplementation()!
    vi.mocked(fetch).mockImplementation(async (url, options) => String(url).endsWith('/auth/logout')
      ? response({ detail: 'Logout temporarily unavailable.' }, 503)
      : previous(url, options))
    await openTutor()
    await userEvent.setup().click(screen.getByRole('button', { name: 'Sign out' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('Logout temporarily unavailable.')
    expect(screen.getByRole('heading', { name: 'AI Tutor' })).toBeInTheDocument()
  })

  it('treats a revoked session during logout as successful sign-out', async () => {
    const previous = vi.mocked(fetch).getMockImplementation()!
    vi.mocked(fetch).mockImplementation(async (url, options) => String(url).endsWith('/auth/logout')
      ? response({ detail: 'Session expired.' }, 401)
      : previous(url, options))
    await openTutor()
    await userEvent.setup().click(screen.getByRole('button', { name: 'Sign out' }))
    expect(await screen.findByRole('heading', { name: 'Welcome back' })).toBeInTheDocument()
  })

  it('submits the question when Enter is pressed', async () => {
    await openTutor()
    const textarea = screen.getByPlaceholderText('Ask a follow-up question…')
    fireEvent.change(textarea, { target: { value: 'How are stars formed?' } })
    fireEvent.keyDown(textarea, { key: 'Enter', shiftKey: false })
    expect(await screen.findByText('Water is H2O.')).toBeInTheDocument()
    expect(vi.mocked(fetch).mock.calls.some(([url]) => String(url).endsWith('/ask'))).toBe(true)
  })
})
