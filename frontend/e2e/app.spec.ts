import { test, expect } from '@playwright/test'

const user = {
  user_id: 'user_A', username: 'student_a', email: 'student@example.com', csrf_token: 'csrf_A',
}

test('guest sees sign-in; login opens private tutor', async ({ page }) => {
  await page.route('**/auth/me', route => route.fulfill({ status: 401, contentType: 'application/json', body: JSON.stringify({ detail: 'Not authenticated.' }) }))
  await page.route('**/auth/login', route => route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(user) }))
  await page.route('**/conversations', route => route.fulfill({ status: 200, contentType: 'application/json', body: '[]' }))
  await page.route('**/documents', route => route.fulfill({ status: 200, contentType: 'application/json', body: '[]' }))
  await page.goto('/')
  await expect(page.getByRole('heading', { name: 'Welcome back' })).toBeVisible()
  await page.getByRole('textbox', { name: 'Username or email' }).fill('student_a')
  await page.getByLabel('Password').fill('password123')
  await page.getByRole('button', { name: 'Sign in' }).click()
  await expect(page.getByRole('heading', { name: 'AI Tutor' })).toBeVisible()
})

test.describe('Authenticated tutor UI', () => {
  test.beforeEach(async ({ page }) => {
    await page.route('**/auth/me', route => route.fulfill({
      status: 200, contentType: 'application/json', body: JSON.stringify(user),
    }))
    await page.route('**/conversations', route => route.fulfill({ status: 200, contentType: 'application/json', body: '[]' }))
    await page.route('**/documents', route => route.fulfill({
      status: 200, contentType: 'application/json',
      body: JSON.stringify([{ doc_id: 'pdf_101', filename: 'biology.pdf', page_count: 8 }]),
    }))
    await page.goto('/')
    await expect(page.getByRole('heading', { name: 'AI Tutor' })).toBeVisible()
  })

  test('loads and selects documents', async ({ page }) => {
    await expect(page.getByText('biology.pdf')).toBeVisible()
    await page.getByRole('checkbox').check()
    await expect(page.getByText('1 PDF selected')).toBeVisible()
  })

  test('sends a question and displays the answer', async ({ page }) => {
    await page.route('**/ask/stream', async route => {
      expect(route.request().headers()['x-csrf-token']).toBe('csrf_A')
      await route.fulfill({ status: 200, contentType: 'text/event-stream',
        body: ['event: status\ndata: {"stage":"checking"}\n\n', 'event: token\ndata: {"text":"Water has formula H2O."}\n\n', `event: complete\ndata: ${JSON.stringify({ status: 'general_knowledge', answer: 'Water has formula H2O.', formatted_answer: '', sources: [], conversation_id: '00000000-0000-4000-8000-000000000001' })}\n\n`].join(''),
      })
    })
    await page.getByPlaceholder('Ask a follow-up question…').fill('What is water?')
    await page.getByRole('button', { name: 'Send message' }).click()
    await expect(page.getByText('Water has formula H2O.')).toBeVisible()
    await expect(page.getByText('Not from document')).toBeVisible()
  })

  test('uploads a PDF with CSRF header', async ({ page }) => {
    await page.route('**/documents/upload', async route => {
      expect(route.request().headers()['x-csrf-token']).toBe('csrf_A')
      await route.fulfill({ status: 200, contentType: 'application/json',
        body: JSON.stringify({ doc_id: 'pdf_202', filename: 'sample.pdf', page_count: 2, status: 'completed' }),
      })
    })
    await page.locator('input[type="file"]').setInputFiles({
      name: 'sample.pdf', mimeType: 'application/pdf', buffer: Buffer.from('%PDF-1.4\n'),
    })
    await expect(page.getByRole('status')).toContainText('sample.pdf: uploaded (2 pages).')
  })

  test('fetches PDF page image with credentials', async ({ page }) => {
    await page.route('**/ask/stream', route => route.fulfill({ status: 200, contentType: 'text/event-stream',
      body: `event: token\ndata: {"text":"Plants absorb sunlight."}\n\nevent: complete\ndata: ${JSON.stringify({ status: 'grounded', answer: 'Plants absorb sunlight.', formatted_answer: '', sources: [{ doc_id: 'pdf_101', page_number: 3 }], conversation_id: null })}\n\n`,
    }))
    await page.route('**/documents/pdf_101/pages/3/image', async route => {
      await route.fulfill({ status: 200, contentType: 'image/png', body: Buffer.from(
        'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIHWP4z8DwHwAFgAI/ScL/nwAAAABJRU5ErkJggg==', 'base64') })
    })
    await page.getByRole('checkbox').check()
    await page.getByPlaceholder('Ask a follow-up question…').fill('Explain photosynthesis')
    await page.getByRole('button', { name: 'Send message' }).click()
    await page.getByRole('button', { name: /biology\.pdf.*Page 3/ }).click()
    await expect(page.getByText('Page 3 of 8')).toBeVisible()
    await expect(page.locator('img[src^="blob:"]')).toBeVisible()
  })

  test('signs out and hides the tutor', async ({ page }) => {
    await page.route('**/auth/logout', route => route.fulfill({ status: 200, contentType: 'application/json', body: '{"message":"Logged out successfully."}' }))
    await page.getByRole('button', { name: 'Sign out' }).click()
    await expect(page.getByRole('heading', { name: 'Welcome back' })).toBeVisible()
    await expect(page.getByRole('heading', { name: 'AI Tutor' })).not.toBeVisible()
  })
})
