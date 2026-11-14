import { describe, expect, it } from 'vitest'
import { linkSourceReferences, normalizeRawCitations } from './citations'

const DOC_ID = 'e8be2d4d-9094-43e8-8cec-7fe4e7853cce'

describe('citation display and legacy history', () => {
  it('converts raw document identifiers to numbered references', () => {
    expect(normalizeRawCitations(`Ammonotelic (${DOC_ID}, p. 1)`, [{ docId: DOC_ID, page: 1 }]))
      .toBe('Ammonotelic [1]')
  })

  it('handles a page 2 citation and mismatched citations safely', () => {
    expect(normalizeRawCitations(`First (${DOC_ID}, p. 2); missing (${DOC_ID}, p. 3)`, [{ docId: DOC_ID, page: 2 }]))
      .toBe('First [1]; missing [unverified source]')
  })

  it('creates markdown links to verified source cards only', () => {
    expect(linkSourceReferences('Answer [1] [2] [3]', 2))
      .toBe('Answer [1](#doubtdesk-source-1) [2](#doubtdesk-source-2) [3]')
  })

  it('does not link numbered text without any documents', () => {
    expect(linkSourceReferences('Step [1]', 0)).toBe('Step [1]')
  })

  it('avoids adding a link to an existing markdown link', () => {
    expect(linkSourceReferences('[1](https://example.com)', 1)).toBe('[1](https://example.com)')
  })
})
