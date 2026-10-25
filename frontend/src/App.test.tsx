import { render, screen, waitFor } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import userEvent from '@testing-library/user-event'
import App from './App'

describe('DoubtDesk App', () => {
  it('renders the application', async () => {
    render(<App />)

    expect(screen.getByRole('heading', { name: 'AI Tutor' })).toBeInTheDocument()
    expect(screen.getByText('What would you like to learn?')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Upload PDF' })).toBeInTheDocument()

    await waitFor(() => {
      expect(fetch).toHaveBeenCalledWith('http://localhost:8000/documents')
    })
  })

  it('displays an empty document message', async () => {
    render(<App />)

    expect(
      await screen.findByText('Upload a PDF to get started.')
    ).toBeInTheDocument()
  })

  it('disables sending an empty question', () => {
    render(<App />)

    expect(
      screen.getByRole('button', { name: 'Send message' })
    ).toBeDisabled()
  })

  it('enables sending when a question is entered', async () => {
    const user = userEvent.setup()

    render(<App />)

    await user.type(
      screen.getByPlaceholderText('Ask a follow-up question…'),
      'What is photosynthesis?'
    )

    expect(
      screen.getByRole('button', { name: 'Send message' })
    ).toBeEnabled()
  })

  it('displays documents returned by the API', async () => {
    vi.mocked(fetch).mockResolvedValueOnce({
      ok: true,
      json: async () => [
        {
          doc_id: 'doc1',
          filename: 'biology.pdf',
          page_count: 8,
        },
      ],
    } as Response)

    render(<App />)

    expect(await screen.findByText('biology.pdf')).toBeInTheDocument()
    expect(screen.getByText('8 pages')).toBeInTheDocument()
  })

  it('filters documents by filename', async () => {
    const user = userEvent.setup()

    vi.mocked(fetch).mockResolvedValueOnce({
      ok: true,
      json: async () => [
        { doc_id: 'doc1', filename: 'biology.pdf', page_count: 8 },
        { doc_id: 'doc2', filename: 'chemistry.pdf', page_count: 5 },
      ],
    } as Response)

    render(<App />)

    await screen.findByText('biology.pdf')

    await user.type(
      screen.getByPlaceholderText('Search documents'),
      'chemistry'
    )

    expect(screen.getByText('chemistry.pdf')).toBeInTheDocument()
    expect(screen.queryByText('biology.pdf')).not.toBeInTheDocument()
  })

  it('displays an error when document loading fails', async () => {
    vi.mocked(fetch).mockRejectedValueOnce(
      new Error('Network unavailable')
    )

    render(<App />)

    expect(
      await screen.findByRole('alert')
    ).toHaveTextContent('Network unavailable')

    expect(screen.getByRole('button', { name: 'Retry' })).toBeInTheDocument()
  })
})

describe('DoubtDesk interactions', () => {
  it('selects a document for grounded questions', async () => {
    const user = userEvent.setup()

    vi.mocked(fetch).mockResolvedValueOnce({
      ok: true,
      json: async () => [
        { doc_id: 'doc1', filename: 'biology.pdf', page_count: 8 },
      ],
    } as Response)

    render(<App />)

    await user.click(await screen.findByRole('checkbox'))

    expect(screen.getByText('1 PDF selected')).toBeInTheDocument()
  })

  it('filters documents by subject', async () => {
    const user = userEvent.setup()

    vi.mocked(fetch).mockResolvedValueOnce({
      ok: true,
      json: async () => [
        { doc_id: 'doc1', filename: 'biology.pdf', page_count: 8 },
        { doc_id: 'doc2', filename: 'chemistry.pdf', page_count: 5 },
      ],
    } as Response)

    render(<App />)

    await screen.findByText('biology.pdf')
    await user.click(screen.getByRole('button', { name: 'Biology' }))

    expect(screen.getByText('biology.pdf')).toBeInTheDocument()
    expect(screen.queryByText('chemistry.pdf')).not.toBeInTheDocument()
  })

  it('submits a question without documents', async () => {
    const user = userEvent.setup()

    vi.mocked(fetch).mockResolvedValueOnce({
      ok: true,
      json: async () => [],
    } as Response)

    vi.mocked(fetch).mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        status: 'general_knowledge',
        answer: 'Photosynthesis converts light into chemical energy.',
        sources: [],
      }),
    } as Response)

    render(<App />)

    await user.type(
      screen.getByPlaceholderText('Ask a follow-up question…'),
      'What is photosynthesis?'
    )

    await user.click(screen.getByRole('button', { name: 'Send message' }))

    expect(
      await screen.findByText('Photosynthesis converts light into chemical energy.')
    ).toBeInTheDocument()

    expect(screen.getByText('Not from document')).toBeInTheDocument()

    expect(fetch).toHaveBeenCalledWith(
      'http://localhost:8000/ask',
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({
          question: 'What is photosynthesis?',
          doc_ids: [],
        }),
      })
    )
  })

  it('sends selected document IDs to the backend', async () => {
    const user = userEvent.setup()

    vi.mocked(fetch)
      .mockResolvedValueOnce({
        ok: true,
        json: async () => [
          { doc_id: 'doc1', filename: 'biology.pdf', page_count: 8 },
        ],
      } as Response)
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          status: 'grounded',
          answer: 'Plants use sunlight.',
          sources: [],
        }),
      } as Response)

    render(<App />)

    await user.click(await screen.findByRole('checkbox'))

    await user.type(
      screen.getByPlaceholderText('Ask a follow-up question…'),
      'Explain photosynthesis'
    )

    await user.click(screen.getByRole('button', { name: 'Send message' }))

    expect(await screen.findByText('Plants use sunlight.')).toBeInTheDocument()
    expect(screen.getByText('From documents')).toBeInTheDocument()

    expect(fetch).toHaveBeenCalledWith(
      'http://localhost:8000/ask',
      expect.objectContaining({
        body: JSON.stringify({
          question: 'Explain photosynthesis',
          doc_ids: ['doc1'],
        }),
      })
    )
  })

  it('submits a question using Enter', async () => {
    const user = userEvent.setup()

    vi.mocked(fetch)
      .mockResolvedValueOnce({
        ok: true,
        json: async () => [],
      } as Response)
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          status: 'general_knowledge',
          answer: 'Water is H2O.',
          sources: [],
        }),
      } as Response)

    render(<App />)

    const input = screen.getByPlaceholderText('Ask a follow-up question…')

    await user.type(input, 'What is water?{Enter}')

    expect(await screen.findByText('Water is H2O.')).toBeInTheDocument()
  })

  it('allows Shift+Enter without submitting', async () => {
    const user = userEvent.setup()

    render(<App />)

    const input = screen.getByPlaceholderText('Ask a follow-up question…')

    await user.type(input, 'First line{Shift>}{Enter}{/Shift}')

    expect(input).toHaveValue('First line\n')
    expect(fetch).not.toHaveBeenCalledWith(
      'http://localhost:8000/ask',
      expect.anything()
    )
  })

  it('displays an error when the answer request fails', async () => {
    const user = userEvent.setup()

    vi.mocked(fetch)
      .mockResolvedValueOnce({
        ok: true,
        json: async () => [],
      } as Response)
      .mockRejectedValueOnce(new Error('Connection failed'))

    render(<App />)

    await user.type(
      screen.getByPlaceholderText('Ask a follow-up question…'),
      'What is DNA?'
    )

    await user.click(screen.getByRole('button', { name: 'Send message' }))

    expect(await screen.findByText('Connection failed')).toBeInTheDocument()
    expect(screen.getByText('Request failed')).toBeInTheDocument()
  })
})

import { fireEvent, within } from '@testing-library/react'

describe('DoubtDesk PDF uploads', () => {
  function upload(file: File) {
    const input = document.querySelector<HTMLInputElement>('input[type="file"]')!
    fireEvent.change(input, { target: { files: [file] } })
  }

  it('rejects non-PDF files', async () => {
    render(<App />)

    upload(new File(['hello'], 'notes.txt', { type: 'text/plain' }))

    expect(await screen.findByRole('status')).toHaveTextContent(
      'Please choose a PDF file.'
    )
    expect(fetch).toHaveBeenCalledTimes(1)
  })

  it('rejects empty PDF files', async () => {
    render(<App />)

    upload(new File([], 'empty.pdf', { type: 'application/pdf' }))

    expect(await screen.findByRole('status')).toHaveTextContent(
      'This PDF is empty.'
    )
    expect(fetch).toHaveBeenCalledTimes(1)
  })

  it('rejects PDFs larger than 25 MB', async () => {
    render(<App />)

    const file = new File(['x'], 'large.pdf', { type: 'application/pdf' })
    Object.defineProperty(file, 'size', { value: 26 * 1024 * 1024 })

    upload(file)

    expect(await screen.findByRole('status')).toHaveTextContent(
      'PDF exceeds the 25 MB limit.'
    )
    expect(fetch).toHaveBeenCalledTimes(1)
  })

  it('uploads a valid PDF and refreshes documents', async () => {
    vi.mocked(fetch)
      .mockResolvedValueOnce({
        ok: true,
        json: async () => [],
      } as Response)
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          doc_id: 'doc1',
          filename: 'biology.pdf',
          page_count: 8,
          status: 'completed',
        }),
      } as Response)
      .mockResolvedValueOnce({
        ok: true,
        json: async () => [
          { doc_id: 'doc1', filename: 'biology.pdf', page_count: 8 },
        ],
      } as Response)

    render(<App />)

    await screen.findByText('Upload a PDF to get started.')

    upload(new File(['%PDF-1.4'], 'biology.pdf', {
      type: 'application/pdf',
    }))

    expect(await screen.findByRole('status')).toHaveTextContent(
      'biology.pdf: uploaded (8 pages).'
    )

    expect(await screen.findByText('biology.pdf')).toBeInTheDocument()

    expect(fetch).toHaveBeenCalledWith(
      'http://localhost:8000/documents/upload',
      expect.objectContaining({
        method: 'POST',
        body: expect.any(FormData),
      })
    )

    const uploadCall = vi.mocked(fetch).mock.calls.find(
      ([url]) => url === 'http://localhost:8000/documents/upload'
    )

    expect(uploadCall).toBeDefined()

    const form = uploadCall![1]!.body as FormData
    expect((form.get('file') as File).name).toBe('biology.pdf')
  })

  it('displays cached document status', async () => {
    vi.mocked(fetch)
      .mockResolvedValueOnce({
        ok: true,
        json: async () => [],
      } as Response)
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          doc_id: 'doc1',
          filename: 'biology.pdf',
          page_count: 8,
          status: 'cached',
        }),
      } as Response)
      .mockResolvedValueOnce({
        ok: true,
        json: async () => [
          { doc_id: 'doc1', filename: 'biology.pdf', page_count: 8 },
        ],
      } as Response)

    render(<App />)

    await screen.findByText('Upload a PDF to get started.')

    upload(new File(['%PDF-1.4'], 'biology.pdf', {
      type: 'application/pdf',
    }))

    expect(await screen.findByRole('status')).toHaveTextContent(
      'biology.pdf: cached document reused.'
    )
  })

  it('displays backend upload errors', async () => {
    vi.mocked(fetch)
      .mockResolvedValueOnce({
        ok: true,
        json: async () => [],
      } as Response)
      .mockResolvedValueOnce({
        ok: false,
        status: 400,
        json: async () => ({ detail: 'PDF is corrupted or unreadable.' }),
      } as Response)

    render(<App />)

    await screen.findByText('Upload a PDF to get started.')

    upload(new File(['%PDF-invalid'], 'broken.pdf', {
      type: 'application/pdf',
    }))

    expect(await screen.findByRole('status')).toHaveTextContent(
      'PDF is corrupted or unreadable.'
    )
  })

  it('allows retrying a failed document-list request', async () => {
    const user = userEvent.setup()

    vi.mocked(fetch)
      .mockRejectedValueOnce(new Error('Network unavailable'))
      .mockResolvedValueOnce({
        ok: true,
        json: async () => [
          { doc_id: 'doc1', filename: 'biology.pdf', page_count: 8 },
        ],
      } as Response)

    render(<App />)

    await screen.findByRole('alert')
    await user.click(within(screen.getByRole('alert')).getByRole('button', {
      name: 'Retry',
    }))

    expect(await screen.findByText('biology.pdf')).toBeInTheDocument()
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
  })
})

describe('DoubtDesk citations and PDF viewer', () => {
  async function renderGroundedAnswer() {
    const user = userEvent.setup()

    vi.mocked(fetch)
      .mockResolvedValueOnce({
        ok: true,
        json: async () => [
          { doc_id: 'doc1', filename: 'biology.pdf', page_count: 8 },
        ],
      } as Response)
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          status: 'grounded',
          answer: 'Photosynthesis converts light energy.',
          sources: [
            {
              doc_id: 'doc1',
              page_number: 3,
              excerpt: 'Plants absorb sunlight.',
            },
          ],
        }),
      } as Response)

    render(<App />)

    await user.click(await screen.findByRole('checkbox'))

    await user.type(
      screen.getByPlaceholderText('Ask a follow-up question…'),
      'Explain photosynthesis'
    )

    await user.click(screen.getByRole('button', { name: 'Send message' }))

    await screen.findByText('Photosynthesis converts light energy.')

    return user
  }

  it('displays citations for grounded answers', async () => {
    await renderGroundedAnswer()

    expect(screen.getByText('From documents')).toBeInTheDocument()
    expect(screen.getByText('Sources')).toBeInTheDocument()
    expect(screen.getByText('Page 3')).toBeInTheDocument()
    expect(screen.getByText('Plants absorb sunlight.')).toBeInTheDocument()
  })

  it('opens the source viewer at the cited page', async () => {
    const user = await renderGroundedAnswer()

    await user.click(screen.getByRole('button', {
      name: /biology\.pdf.*Page 3/,
    }))

    expect(screen.getByText('Page 3 of 8')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Next page' })).toBeEnabled()
    expect(screen.getByRole('button', { name: 'Previous page' })).toBeEnabled()
  })

  it('navigates between PDF pages', async () => {
    const user = await renderGroundedAnswer()

    await user.click(screen.getByRole('button', {
      name: /biology\.pdf.*Page 3/,
    }))

    await user.click(screen.getByRole('button', { name: 'Next page' }))
    expect(screen.getByText('Page 4 of 8')).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'Previous page' }))
    expect(screen.getByText('Page 3 of 8')).toBeInTheDocument()
  })

  it('adjusts the viewer zoom', async () => {
    const user = await renderGroundedAnswer()

    await user.click(screen.getByRole('button', {
      name: /biology\.pdf.*Page 3/,
    }))

    expect(screen.getByText('100%')).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'Zoom in' }))
    expect(screen.getByText('125%')).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'Zoom out' }))
    expect(screen.getByText('100%')).toBeInTheDocument()
  })

  it('closes the source viewer', async () => {
    const user = await renderGroundedAnswer()

    await user.click(screen.getByRole('button', {
      name: /biology\.pdf.*Page 3/,
    }))

    await user.click(screen.getByRole('button', { name: 'Close viewer' }))

    expect(screen.queryByText('Page 3 of 8')).not.toBeInTheDocument()
  })

  it('does not show citations for general knowledge answers', async () => {
    const user = userEvent.setup()

    vi.mocked(fetch)
      .mockResolvedValueOnce({
        ok: true,
        json: async () => [],
      } as Response)
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          status: 'general_knowledge',
          answer: 'Water is H2O.',
          sources: [
            { filename: 'biology.pdf', page_number: 3 },
          ],
        }),
      } as Response)

    render(<App />)

    await user.type(
      screen.getByPlaceholderText('Ask a follow-up question…'),
      'What is water?'
    )

    await user.click(screen.getByRole('button', { name: 'Send message' }))

    expect(await screen.findByText('Water is H2O.')).toBeInTheDocument()
    expect(screen.getByText('Not from document')).toBeInTheDocument()
    expect(screen.queryByText('Sources')).not.toBeInTheDocument()
  })
})