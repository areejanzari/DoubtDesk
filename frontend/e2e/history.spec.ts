import { test, expect } from '@playwright/test'

const account = { user_id: 'owner', username: 'student', email: 'student@example.com', csrf_token: 'csrf' }
const id = '00000000-0000-4000-8000-000000000001'
const item = { conversation_id: id, title: 'Nephron structure', created_at: '2026-11-13', last_accessed_ns: 1, message_count: 1 }

test.beforeEach(async ({ page }) => {
  await page.route('**/auth/me', route => route.fulfill({ json: account }))
  await page.route('**/documents', route => route.fulfill({ json: [{ doc_id: 'doc-1', filename: 'biology.pdf', page_count: 5 }] }))
})

test('restores existing conversations and allows rename/delete', async ({ page }) => {
  let chats = [item]
  await page.route('**/conversations', route => route.fulfill({ json: chats }))
  await page.route(`**/conversations/${id}`, async route => {
    const method = route.request().method()
    if (method === 'GET') return route.fulfill({ json: { ...item, messages: [{ id: 1, question: 'Describe a nephron', answer: 'It is a kidney unit.', formatted_answer: 'biology.pdf: PDF page(s) 2', sources: [{ doc_id: 'doc-1', page_numbers: [2] }], doc_ids: ['doc-1'], status: 'grounded', created_at: '2026-11-13' }] } })
    if (method === 'PATCH') {
      const title = route.request().postDataJSON().title
      chats = chats.map(chat => ({ ...chat, title }))
      return route.fulfill({ json: chats[0] })
    }
    chats = []
    return route.fulfill({ status: 204 })
  })
  await page.goto('/')
  await page.getByRole('button', { name: 'Nephron structure', exact: true }).click()
  await expect(page.getByText('It is a kidney unit.')).toBeVisible()
  await expect(page.getByRole('button', { name: /biology\.pdf.*Page 2/ })).toBeVisible()
  await page.getByRole('button', { name: 'Rename Nephron structure' }).click()
  await page.getByRole('textbox', { name: 'Conversation title' }).fill('Kidney revision')
  await page.getByRole('button', { name: 'Save title' }).click()
  await expect(page.getByRole('button', { name: 'Kidney revision', exact: true })).toBeVisible()
  page.once('dialog', dialog => dialog.accept())
  await page.getByRole('button', { name: 'Delete Kidney revision' }).click()
  await expect(page.getByText('No saved chats yet.')).toBeVisible()
})

test('prevents creating a 31st saved conversation', async ({ page }) => {
  await page.route('**/conversations', route => route.fulfill({ json: Array.from({ length: 30 }, (_, n) => ({ ...item, conversation_id: `chat-${n}`, title: `Study ${n}` })) }))
  await page.goto('/')
  await expect(page.getByText('30/30')).toBeVisible()
  await expect(page.getByRole('button', { name: 'New Chat' })).toBeDisabled()
  await expect(page.getByText(/30-chat limit reached/)).toBeVisible()
})
