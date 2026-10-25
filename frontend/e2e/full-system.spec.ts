
import { test, expect } from '@playwright/test'
import { resolve } from 'node:path'

test('uploads a PDF, asks a question, and opens its citation', async ({ page }) => {
  test.setTimeout(120_000)

  const pdfPath = resolve('e2e/fixtures/test-document.pdf')

  await page.goto('/')

  await page.locator('input[type="file"]').setInputFiles(pdfPath)

  await expect(page.getByRole('status')).toContainText(
    /test-document\.pdf: (uploaded|cached document reused)/,
    { timeout: 90_000 }
  )

  await expect(page.getByText('test-document.pdf').first()).toBeVisible()

  const documentRow = page.getByText('test-document.pdf').first()
  await documentRow.locator('xpath=..').getByRole('checkbox').check()

  await expect(page.getByText('1 PDF selected')).toBeVisible()

  await page.getByPlaceholder('Ask a follow-up question…').fill(
    'What does photosynthesis convert?'
  )

  await page.getByRole('button', { name: 'Send message' }).click()

  await expect(
    page.getByText('Photosynthesis converts light energy into chemical energy.')
  ).toBeVisible({ timeout: 60_000 })

  await page.getByRole('button', {
    name: /test-document\.pdf.*Page 1/
  }).click()

  await expect(page.getByText('Page 1 of 1')).toBeVisible()

  const pageImage = page.locator('img[src*="/documents/"][src*="/image"]')

  await expect(pageImage).toBeVisible()

  await expect.poll(
    () => pageImage.evaluate(img => (img as HTMLImageElement).naturalWidth)
  ).toBeGreaterThan(0)
})