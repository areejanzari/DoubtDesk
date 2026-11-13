import { test, expect } from '@playwright/test'
import { resolve } from 'node:path'

test('uploads a PDF, asks a question, and opens its citation', async ({ page }) => {
  test.setTimeout(180_000)

  const pdfPath = resolve('e2e/fixtures/test-document.pdf')

  await page.goto('/')
  await expect(page.getByRole('heading', { name: 'Welcome back' })).toBeVisible()

  await page.getByRole('textbox', { name: 'Username or email' }).fill('e2e_student')
  await page.getByLabel('Password').fill('e2e-password-123')
  await page.getByRole('button', { name: 'Sign in' }).click()

  await expect(page.getByRole('heading', { name: 'AI Tutor' })).toBeVisible()
  await expect(page.getByRole('button', { name: 'Upload PDF' })).toBeVisible()

  await page.locator('input[type="file"]').setInputFiles(pdfPath)

  await expect(page.getByRole('status')).toContainText(
    /test-document\.pdf: (uploaded|cached document reused)/,
    { timeout: 90_000 },
  )

  const documentRow = page.getByText('test-document.pdf').first()
  await expect(documentRow).toBeVisible()
  await documentRow.locator('xpath=..').getByRole('checkbox').check()
  await expect(page.getByText('1 PDF selected')).toBeVisible()

  await page.getByPlaceholder('Ask a follow-up question…').fill(
    'What does photosynthesis convert?',
  )
  await page.getByRole('button', { name: 'Send message' }).click()

  await expect(
    page.getByText('Photosynthesis converts light energy into chemical energy.'),
  ).toBeVisible({ timeout: 60_000 })

  await page.getByRole('button', {
    name: /test-document\.pdf.*Page 1/,
  }).click()

  await expect(page.getByText('Page 1 of 1')).toBeVisible()
  const pageImage = page.locator('img[src^="blob:"]')
  await expect(pageImage).toBeVisible()
  await expect.poll(
    () => pageImage.evaluate(img => (img as HTMLImageElement).naturalWidth),
  ).toBeGreaterThan(0)
  await expect(page.getByRole('button', { name: 'What does photosynthesis convert?', exact: true })).toBeVisible()
  await page.reload()
  await page.getByRole('button', { name: 'What does photosynthesis convert?', exact: true }).click()
  await expect(page.getByText('Photosynthesis converts light energy into chemical energy.')).toBeVisible()
})
