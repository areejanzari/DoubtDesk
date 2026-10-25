import { test, expect } from '@playwright/test'

test.beforeEach(async ({ page }) => {
  await page.route('**/documents', async route => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify([
        {
          doc_id: 'doc1',
          filename: 'biology.pdf',
          page_count: 8,
        },
      ]),
    })
  })

  await page.goto('/')
})

test('loads the DoubtDesk interface', async ({ page }) => {
  await expect(page.getByRole('heading', { name: 'AI Tutor' })).toBeVisible()
  await expect(page.getByText('What would you like to learn?')).toBeVisible()
  await expect(page.getByRole('button', { name: 'Upload PDF' })).toBeVisible()
  await expect(page.getByText('biology.pdf')).toBeVisible()
})

test('selects a document', async ({ page }) => {
  await page.getByRole('checkbox').check()

  await expect(page.getByText('1 PDF selected')).toBeVisible()
})

test('submits a question and displays the answer', async ({ page }) => {
  await page.route('**/ask', async route => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        status: 'general_knowledge',
        answer: 'Water has the chemical formula H2O.',
        sources: [],
      }),
    })
  })

  await page.getByPlaceholder('Ask a follow-up question…').fill('What is water?')
  await page.getByRole('button', { name: 'Send message' }).click()

  await expect(page.getByText('What is water?')).toBeVisible()
  await expect(page.getByText('Water has the chemical formula H2O.')).toBeVisible()
  await expect(page.getByText('Not from document')).toBeVisible()
})

test('filters documents', async ({ page }) => {
  await page.getByPlaceholder('Search documents').fill('physics')

  await expect(page.getByText('No matching PDFs.')).toBeVisible()
  await expect(page.getByText('biology.pdf')).not.toBeVisible()
})

test('uploads a PDF', async ({ page }) => {
  await page.route('**/documents/upload', async route => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        doc_id: 'doc2',
        filename: 'sample.pdf',
        page_count: 2,
        status: 'completed',
      }),
    })
  })

  await page.locator('input[type="file"]').setInputFiles({
    name: 'sample.pdf',
    mimeType: 'application/pdf',
    buffer: Buffer.from('%PDF-1.4\n'),
  })

  await expect(page.getByRole('status')).toContainText(
    'sample.pdf: uploaded (2 pages).'
  )
})

test('opens the source viewer', async ({ page }) => {
  await page.route('**/ask', async route => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        status: 'grounded',
        answer: 'Plants absorb sunlight.',
        sources: [
          {
            doc_id: 'doc1',
            page_number: 3,
          },
        ],
      }),
    })
  })

  await page.getByRole('checkbox').check()
  await page.getByPlaceholder('Ask a follow-up question…').fill('Explain photosynthesis')
  await page.getByRole('button', { name: 'Send message' }).click()

  await page.getByRole('button', { name: /biology\.pdf.*Page 3/ }).click()

  await expect(page.getByText('Page 3 of 8')).toBeVisible()

  await page.getByRole('button', { name: 'Next page' }).click()

  await expect(page.getByText('Page 4 of 8')).toBeVisible()
})