import { test, expect } from '@playwright/test'

const id = 'e8be2d4d-9094-43e8-8cec-7fe4e7853cce'
const other = '1c2f3e40-1111-2222-3333-444455556666'
const chatId = '00000000-0000-4000-8000-000000000001'
const chat = { conversation_id: chatId, title: 'Ureotelic animals?', message_count: 1, created_at: '2026-11-14', last_accessed_ns: 1 }
const answer = { status: 'grounded', answer: 'Ureotelic animals mainly excrete urea. (PDF p. 1)', formatted_answer: 'biology.pdf: PDF page(s) 1', sources: [{ doc_id: other, page_numbers: [1] }], conversation_id: chatId }
const png = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/l9sAAAAASUVORK5CYII=', 'base64')

test('page-only citations open the correct document in a streamed answer and after restoring history', async ({ page }) => {
  await page.route('**/auth/me', route => route.fulfill({ json: { user_id: 'owner', username: 'student', email: 'student@example.com', csrf_token: 'csrf' } }))
  await page.route('**/documents', route => route.fulfill({ json: [{ doc_id: id, filename: 'biology.pdf', page_count: 5 }, { doc_id: other, filename: 'biology.pdf', page_count: 5 }] }))
  await page.route('**/conversations', route => route.fulfill({ json: [chat] }))
  await page.route(`**/conversations/${chatId}`, route => route.fulfill({ json: { ...chat, messages: [{ id: 1, question: chat.title, ...answer, doc_ids: [other], created_at: '2026-11-14' }] } }))
  await page.route('**/ask/stream', route => route.fulfill({ contentType: 'text/event-stream', body: `event: token\ndata: ${JSON.stringify({ text: answer.answer })}\n\nevent: complete\ndata: ${JSON.stringify(answer)}\n\n` }))
  await page.route('**/documents/*/pages/*/image', route => route.fulfill({ contentType: 'image/png', body: png }))
  await page.goto('/')
  await expect(page.getByRole('heading', { name: 'AI Tutor' })).toBeVisible()
  await page.getByRole('checkbox').nth(1).check()
  await page.getByPlaceholder('Ask a follow-up question…').fill(chat.title)
  await page.getByRole('button', { name: 'Send message' }).click()
  const citation = page.getByRole('button', { name: 'Open source 1', exact: true })
  await expect(citation).toHaveText('[1]')
  await expect(page.getByText('(PDF p. 1)', { exact: false })).toHaveCount(0)
  const first = page.waitForRequest(`**/documents/${other}/pages/1/image`)
  await citation.click()
  await first
  await expect(page.getByText('Page 1 of 5')).toBeVisible()
  await page.reload()
  await page.getByRole('button', { name: chat.title, exact: true }).click()
  await expect(citation).toBeVisible()
  const restored = page.waitForRequest(`**/documents/${other}/pages/1/image`)
  await citation.click()
  await restored
  await expect(page.getByText('Page 1 of 5')).toBeVisible()
})
