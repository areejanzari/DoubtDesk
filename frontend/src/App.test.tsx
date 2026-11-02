import { beforeEach, describe, expect, it, vi } from 'vitest'
import { render, screen, waitFor, fireEvent } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import App from './App'

const ACCOUNT = {
  user_id: 'user_A', username: 'student_a', email: 'student@example.com', csrf_token: 'csrf_A',
}
const DOC = { doc_id: 'pdf_101', filename: 'biology.pdf', page_count: 8 }
const response = (body: unknown, status = 200) => ({
  ok: status >= 200 && status < 300,
  status,
  headers: new Headers({ 'content-type': 'application/json' }),
  json: async () => body,
} as Response)

let documents: typeof DOC[] = []
let askResult: unknown = {
  status: 'general_knowledge', answer: 'Water is H2O.', formatted_answer: '',
  sources: [], conversation_id: '00000000-0000-4000-8000-000000000001',
}

beforeEach(() => {
  documents = []
  askResult = {
    status: 'general_knowledge', answer: 'Water is H2O.', formatted_answer: '',
    sources: [], conversation_id: '00000000-0000-4000-8000-000000000001',
  }
  vi.mocked(fetch).mockImplementation(async (input) => {
    const url = String(input)
    if (url.endsWith('/auth/me')) return response(ACCOUNT)
    if (url.endsWith('/documents')) return response(documents)
    if (url.endsWith('/ask')) return response(askResult)
    if (url.endsWith('/documents/upload')) return response({ status: 'completed', filename: 'sample.pdf', page_count: 2, doc_id: 'pdf_202' })
    if (url.endsWith('/auth/logout')) return response({ message: 'Logged out successfully.' })
    return response({ detail: 'Unknown route' }, 404)
  })
  window.history.replaceState({}, '', '/')
})

async function openTutor() {
  render(<App />)
  await screen.findByRole('heading', { name: 'AI Tutor' })
}

describe('Authenticated tutor', () => {
  it('restores session and loads only the account documents', async () => {
    documents = [DOC]
    await openTutor()
    expect(await screen.findByText('biology.pdf')).toBeInTheDocument()
    expect(screen.getByText('student_a')).toBeInTheDocument()
    

expect(fetch).toHaveBeenCalledWith(
  '/auth/me',
  expect.objectContaining({
    credentials: 'include',
  }),
)

expect(fetch).toHaveBeenCalledWith(
  '/documents',
  expect.objectContaining({
    credentials: 'include',
  }),
)

  })

  it('allows PDF filtering and selection', async () => {
    documents = [DOC, { doc_id: 'pdf_202', filename: 'chemistry.pdf', page_count: 4 }]
    const user = userEvent.setup()
    await openTutor()
    await user.type(screen.getByPlaceholderText('Search documents'), 'chemistry')
    expect(screen.getByText('chemistry.pdf')).toBeInTheDocument()
    expect(screen.queryByText('biology.pdf')).not.toBeInTheDocument()
    await user.click(screen.getByRole('checkbox'))
    expect(screen.getByText('1 PDF selected')).toBeInTheDocument()
  })

  it('sends the CSRF token, cookies, and selected PDF IDs for chat', async () => {
    documents = [DOC]
    const user = userEvent.setup()
    await openTutor()
    await user.click(await screen.findByRole('checkbox'))
    await user.type(screen.getByPlaceholderText('Ask a follow-up question…'), 'Explain biology')
    await user.click(screen.getByRole('button', { name: 'Send message' }))
    expect(await screen.findByText('Water is H2O.')).toBeInTheDocument()
    const askCall = vi.mocked(fetch).mock.calls.find(([url]) => String(url).endsWith('/ask'))!
    const opts = askCall[1]!
    expect(opts.credentials).toBe('include')
    expect(new Headers(opts.headers).get('X-CSRF-Token')).toBe('csrf_A')
    expect(JSON.parse(String(opts.body))).toEqual({ question: 'Explain biology', doc_ids: ['pdf_101'] })
  })

  it('reuses the returned conversation ID for follow-ups', async () => {
    const user = userEvent.setup()
    await openTutor()
    const field = screen.getByPlaceholderText('Ask a follow-up question…')
    await user.type(field, 'First question')
    await user.click(screen.getByRole('button', { name: 'Send message' }))
    await screen.findByText('Water is H2O.')
    await user.type(field, 'Follow up')
    await user.click(screen.getByRole('button', { name: 'Send message' }))
    await waitFor(() => expect(vi.mocked(fetch).mock.calls.filter(([url]) => String(url).endsWith('/ask'))).toHaveLength(2))
    const calls = vi.mocked(fetch).mock.calls.filter(([url]) => String(url).endsWith('/ask'))
    expect(JSON.parse(String(calls[1][1]!.body)).conversation_id).toBe(askResult && (askResult as { conversation_id: string }).conversation_id)
  })

  it('uploads a PDF with CSRF and refreshes the list', async () => {
    await openTutor()
    fireEvent.change(document.querySelector('input[type="file"]')!, { target: { files: [new File(['%PDF-1.4'], 'sample.pdf', { type: 'application/pdf' })] } })
    expect(await screen.findByRole('status')).toHaveTextContent('sample.pdf: uploaded (2 pages).')
    const call = vi.mocked(fetch).mock.calls.find(([url]) => String(url).endsWith('/documents/upload'))!
    expect(new Headers(call[1]!.headers).get('X-CSRF-Token')).toBe('csrf_A')
    expect(call[1]!.credentials).toBe('include')
    expect(call[1]!.body).toBeInstanceOf(FormData)
  })

  it('blocks invalid uploads before calling the API', async () => {
    await openTutor()
    fireEvent.change(document.querySelector('input[type="file"]')!, { target: { files: [new File(['hi'], 'invalid.txt')] } })
    expect(await screen.findByRole('status')).toHaveTextContent('Please choose a PDF file.')
    expect(vi.mocked(fetch).mock.calls.find(([url]) => String(url).endsWith('/documents/upload'))).toBeUndefined()
  })

  it('logs out, revokes UI access, and clears the chat', async () => {
    const user = userEvent.setup()
    await openTutor()
    await user.type(screen.getByPlaceholderText('Ask a follow-up question…'), 'What is water?')
    await user.click(screen.getByRole('button', { name: 'Send message' }))
    await screen.findByText('Water is H2O.')
    await user.click(screen.getByRole('button', { name: 'Sign out' }))
    expect(await screen.findByRole('heading', { name: 'Welcome back' })).toBeInTheDocument()
    expect(screen.queryByText('Water is H2O.')).not.toBeInTheDocument()
    const call = vi.mocked(fetch).mock.calls.find(([url]) => String(url).endsWith('/auth/logout'))!
    expect(new Headers(call[1]!.headers).get('X-CSRF-Token')).toBe('csrf_A')
  })

  it('returns to sign-in when a protected request gets 401', async () => {
    const user = userEvent.setup()
    await openTutor()
    vi.mocked(fetch).mockImplementation(async (input) => String(input).endsWith('/ask')
      ? response({ detail: 'Invalid or expired session.' }, 401)
      : response([]))
    await user.type(screen.getByPlaceholderText('Ask a follow-up question…'), 'Is the session active?')
    await user.click(screen.getByRole('button', { name: 'Send message' }))
    expect(await screen.findByRole('heading', { name: 'Welcome back' })).toBeInTheDocument()
  })

  it('shows backend validation errors in chat', async () => {
    const user = userEvent.setup()
    await openTutor()
    vi.mocked(fetch).mockImplementation(async (input) => String(input).endsWith('/ask')
      ? response({ detail: 'One or more documents are unavailable.' }, 400)
      : response([]))
    await user.type(screen.getByPlaceholderText('Ask a follow-up question…'), 'Explain?')
    await user.click(screen.getByRole('button', { name: 'Send message' }))
    expect(await screen.findByText('One or more documents are unavailable.')).toBeInTheDocument()
  })

  it('shows source citations without public image URLs', async () => {
    documents = [DOC]
    askResult = {
      status: 'grounded', answer: 'Plants use light.', formatted_answer: '',
      sources: [{ doc_id: 'pdf_101', page_number: 3 }], conversation_id: null,
    }
    const user = userEvent.setup()
    const createUrl = vi.fn().mockReturnValue('blob:http://localhost/secure-page')
    const revokeUrl = vi.fn()
    URL.createObjectURL = createUrl
    URL.revokeObjectURL = revokeUrl
    const request = vi.mocked(fetch)
    const baseImpl = request.getMockImplementation()!
    request.mockImplementation(async (input, init) => {
      if (String(input).endsWith('/documents/pdf_101/pages/3/image')) {
        return { ok: true, status: 200, headers: new Headers({ 'content-type': 'image/png' }), blob: async () => new Blob(['image'], { type: 'image/png' }) } as Response
      }
      return baseImpl(input, init)
    })
    await openTutor()
    await user.click(await screen.findByRole('checkbox'))
    await user.type(screen.getByPlaceholderText('Ask a follow-up question…'), 'Explain photosynthesis')
    await user.click(screen.getByRole('button', { name: 'Send message' }))
    await user.click(await screen.findByRole('button', { name: /biology\.pdf.*Page 3/ }))
    await waitFor(() => expect(screen.getByAltText('biology.pdf page 3')).toHaveAttribute('src', 'blob:http://localhost/secure-page'))
    const call = vi.mocked(fetch).mock.calls.find(([url]) => String(url).endsWith('/documents/pdf_101/pages/3/image'))!
    expect(call[1]!.credentials).toBe('include')
    await user.click(screen.getByRole('button', { name: 'Close viewer' }))
    await waitFor(() => expect(revokeUrl).toHaveBeenCalledWith('blob:http://localhost/secure-page'))
  })
})

describe('Authentication screens', () => {
  beforeEach(() => {
    vi.mocked(fetch).mockImplementation(async (input) => String(input).endsWith('/auth/me')
      ? response({ detail: 'Not authenticated.' }, 401)
      : response({ message: 'If this account is eligible, an email will be sent.' }))
  })

  it('shows sign-in instead of private tutor to guests', async () => {
    render(<App />)
    expect(await screen.findByRole('heading', { name: 'Welcome back' })).toBeInTheDocument()
    expect(screen.queryByRole('heading', { name: 'AI Tutor' })).not.toBeInTheDocument()
  })

  it('logs in and sends cookies via the browser', async () => {
    const user = userEvent.setup()
    const baseImpl = vi.mocked(fetch).getMockImplementation()!
    vi.mocked(fetch).mockImplementation(async (input, init) =>
      String(input).endsWith('/auth/login') ? response(ACCOUNT) :
        String(input).endsWith('/documents') ? response([]) : baseImpl(input, init))
    render(<App />)
    await screen.findByRole('heading', { name: 'Welcome back' })
    await user.type(screen.getByRole('textbox', { name: 'Username or email' }), 'student_a')
    await user.type(screen.getByLabelText('Password'), 'password123')
    await user.click(screen.getByRole('button', { name: /Sign in/ }))
    expect(await screen.findByRole('heading', { name: 'AI Tutor' })).toBeInTheDocument()
    const call = vi.mocked(fetch).mock.calls.find(([url]) => String(url).endsWith('/auth/login'))!
    expect(call[1]!.credentials).toBe('include')
    expect(JSON.parse(String(call[1]!.body))).toEqual({ identifier: 'student_a', password: 'password123' })
  })

  it('signs up with the FastAPI field names', async () => {
    const user = userEvent.setup()
    const baseImpl = vi.mocked(fetch).getMockImplementation()!
    vi.mocked(fetch).mockImplementation(async (input, init) => String(input).endsWith('/auth/signup')
      ? response({ user_id: 'id', username: 'learner', email: 'learner@example.com', email_verified: false }, 201)
      : baseImpl(input, init))
    render(<App />)
    await screen.findByRole('heading', { name: 'Welcome back' })
    await user.click(screen.getByRole('button', { name: 'Create an account' }))
    await user.type(screen.getByRole('textbox', { name: 'Username' }), 'learner')
    await user.type(screen.getByRole('textbox', { name: 'Email' }), 'learner@example.com')
    await user.type(screen.getByLabelText('Password', { exact: true }), 'password123')
    await user.type(screen.getByLabelText('Confirm password'), 'password123')
    await user.click(screen.getByRole('button', { name: /Create account/ }))
    expect(await screen.findByRole('heading', { name: 'Check your inbox' })).toBeInTheDocument()
    const call = vi.mocked(fetch).mock.calls.find(([url]) => String(url).endsWith('/auth/signup'))!
    expect(JSON.parse(String(call[1]!.body))).toEqual({ username: 'learner', email: 'learner@example.com', password: 'password123' })
  })

  it('supports password reset without requiring login', async () => {
    const user = userEvent.setup()
    render(<App />)
    await screen.findByRole('heading', { name: 'Welcome back' })
    await user.click(screen.getByRole('button', { name: 'Forgot password?' }))
    await user.type(screen.getByRole('textbox', { name: 'Email' }), 'learner@example.com')
    await user.click(screen.getByRole('button', { name: /Send reset link/ }))
    expect(await screen.findByRole('status')).toHaveTextContent('email will be sent')
    expect(vi.mocked(fetch).mock.calls.some(([url]) => String(url).endsWith('/auth/forgot-password'))).toBe(true)
  })

  it('resends a verification email', async () => {
    const user = userEvent.setup()
    render(<App />)
    await screen.findByRole('heading', { name: 'Welcome back' })
    await user.click(screen.getByRole('button', { name: 'Resend verification email' }))
    await user.type(screen.getByRole('textbox', { name: 'Email' }), 'student@example.com')
    await user.click(screen.getByRole('button', { name: 'Resend verification email' }))
    const call = vi.mocked(fetch).mock.calls.find(([url]) => String(url).endsWith('/auth/resend-verification'))!
    expect(JSON.parse(String(call[1]!.body))).toEqual({ email: 'student@example.com' })
  })

  it('resets password using the token from a link', async () => {
    const user = userEvent.setup()
    window.history.replaceState({}, '', '/reset-password?token=reset-private-token')
    render(<App />)
    await screen.findByRole('heading', { name: 'Choose a new password' })
    expect(window.location.search).toBe('')
    await user.type(screen.getByLabelText('New password'), 'newpassword123')
    await user.type(screen.getByLabelText('Confirm password'), 'newpassword123')
    await user.click(screen.getByRole('button', { name: 'Reset password' }))
    expect(await screen.findByRole('heading', { name: 'Welcome back' })).toBeInTheDocument()
    const call = vi.mocked(fetch).mock.calls.find(([url]) => String(url).endsWith('/auth/reset-password'))!
    expect(JSON.parse(String(call[1]!.body))).toEqual({ token: 'reset-private-token', new_password: 'newpassword123' })
  })

  it('verifies from an email-link token and removes it from the address bar', async () => {
    const user = userEvent.setup()
    window.history.replaceState({}, '', '/verify-email?token=private-value')
    render(<App />)
    expect(await screen.findByRole('heading', { name: 'Verify your email' })).toBeInTheDocument()
    expect(window.location.search).toBe('')
    await user.click(screen.getByRole('button', { name: 'Verify email' }))
    expect(await screen.findByRole('heading', { name: 'Welcome back' })).toBeInTheDocument()
    const call = vi.mocked(fetch).mock.calls.find(([url]) => String(url).endsWith('/auth/verify-email'))!
    expect(JSON.parse(String(call[1]!.body))).toEqual({ token: 'private-value' })
  })
})
